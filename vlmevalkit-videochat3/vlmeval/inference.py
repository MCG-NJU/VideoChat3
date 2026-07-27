import torch
import torch.distributed as dist
import glob
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(
    model,
    model_name,
    work_dir,
    dataset,
    out_file,
    verbose=False,
    api_nproc=4,
    use_vllm=False,
    assigned_indices=None,
):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    if assigned_indices is not None:
        data = dataset.data[dataset.data['index'].isin(assigned_indices)]
        data_indices = list(data['index'])
    else:
        sheet_indices = list(range(rank, len(dataset), world_size))
        data = dataset.data.iloc[sheet_indices]
        data_indices = [i for i in data['index']]
    lt = len(data_indices)

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        # For dynamic load-balancing, keep all cached results in this file,
        # since different ranks may take over unfinished samples from others.
        if assigned_indices is None:
            res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    if dist.is_initialized():
        if dist.get_rank() == 0:
            model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
        dist.barrier()
        if dist.get_rank() != 0:
            model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    else:
        model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        if assigned_indices is None:
            res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        dump(res, out_file)

    # For the default (non-dynamic) scheduling strategy, keep only the samples
    # belonging to this rank. When dynamic load balancing is enabled
    # (assigned_indices is not None), we keep all cached results in this file,
    # since different ranks may take over unfinished samples from others.
    if assigned_indices is None:
        res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# Add for agent evaluation
def _is_structured_record(v):
    return isinstance(v, dict) and 'prediction' in v and 'extra_records' in v


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            # breakpoint()
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    # Use result file stem (model_name + dataset_name) as job identifier to avoid collisions.
    # Backward compatible: we will also discover legacy part files named by dataset only.
    stem = osp.splitext(osp.basename(result_file))[0]
    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{stem}.pkl')
    out_file = tmpl.format(rank)

    # -------- Dynamic load-balancing for image inference --------
    # Support resuming from previous partial results (per-rank pkl files)
    # while re-balancing the remaining samples evenly across all ranks.
    assigned_indices = None
    existing_part_files = []
    existing_cache = {}
    if rank == 0:
        sample_indices = list(dataset.data['index'])
        done_indices = set()

        # Discover any historical per-rank partial files for this job,
        # regardless of the world_size used to generate them.
        # 1) new naming: *_{stem}.pkl
        # 2) legacy naming: *_{dataset_name}.pkl
        part_files_new = sorted(glob.glob(osp.join(work_dir, f'*_{stem}.pkl')))
        part_files_legacy = sorted(glob.glob(osp.join(work_dir, f'*_{dataset_name}.pkl')))
        # Keep order-stable while de-duplicating
        existing_part_files = []
        seen = set()
        for pf in (part_files_new + part_files_legacy):
            if pf not in seen:
                existing_part_files.append(pf)
                seen.add(pf)

        if len(existing_part_files):
            print(f"===== Found {len(existing_part_files)} existing part files (any world_size) ======")
            for part_file in existing_part_files:
                print(f"===== Part File {part_file} ======")
                if not osp.exists(part_file):
                    continue
                part_res = load(part_file)
                if isinstance(part_res, dict):
                    existing_cache.update(part_res)
                    done_indices.update(part_res.keys())
        else:
            print("===== Found 0 existing part files ======")

        remaining_indices = [idx for idx in sample_indices if idx not in done_indices]
        print(f"===== Done {len(done_indices)} | Remaining {len(remaining_indices)} | Total {len(sample_indices)} ======")

        assigned_indices_all = [[] for _ in range(world_size)]
        for j, idx in enumerate(remaining_indices):
            assigned_indices_all[j % world_size].append(idx)
    else:
        assigned_indices_all = None

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        obj_list = [assigned_indices_all]
        dist.broadcast_object_list(obj_list, src=0)
        assigned_indices_all = obj_list[0]

    if assigned_indices_all is not None:
        assigned_indices = assigned_indices_all[rank]

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm,
        assigned_indices=assigned_indices)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        # Merge results from:
        # 1) any historical partial files (any world_size), and
        # 2) current run's partial files (current world_size).
        data_all = dict(existing_cache)
        part_files_to_cleanup = set(existing_part_files)
        for i in range(world_size):
            part_file = tmpl.format(i)
            if osp.exists(part_file):
                data_all.update(load(part_file))
                part_files_to_cleanup.add(part_file)

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        if os.getenv('SPLIT_THINK', False):
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                prediction = [data_all[x]['prediction'] for x in data['index']]
                extra_records = [data_all[x]['extra_records'] for x in data['index']]
                data['extra_records'] = extra_records
            else:
                prediction = [str(data_all[x]) for x in data['index']]

            def split_thinking(s):
                if '</think>' in s:
                    splits = s.split('</think>')
                    prediction = splits[-1].strip()
                    if len(splits) == 2 and '<think>' in splits[0]:
                        thinking = splits[0].split('<think>')[1].strip()
                    else:
                        thinking = '</think>'.join(splits[:-1])
                        thinking += '</think>'
                        warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
                else:
                    thinking = ''
                    prediction = s
                return (prediction, thinking)
            split_func = model.split_thinking if hasattr(model, 'split_thinking') else split_thinking
            print(f'Prediction format: {os.getenv("SPLIT_THINK")},splitting func: {split_func}')
            tups = [split_func(x) for x in prediction]
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            # data['prediction'] = [str(data_all[x]) for x in data['index']]
            # Add for agent evaluation
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                data['prediction'] = [data_all[x]['prediction'] for x in data['index']]
                data['extra_records'] = [data_all[x]['extra_records'] for x in data['index']]
            else:
                data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        # Clean up all partial files we used for this job (both historical and current).
        for part_file in sorted(part_files_to_cleanup):
            try:
                if osp.exists(part_file):
                    os.remove(part_file)
            except Exception as e:
                print(f"[WARN] Failed to remove partial file {part_file}: {e}")
    if world_size > 1:
        dist.barrier()
    return model
