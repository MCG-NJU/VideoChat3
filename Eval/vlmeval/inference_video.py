import torch
import torch.distributed as dist
import glob
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def _rank_res_keep_keys(sample_indices_sub, initial_keys, assigned_indices):
    """Keys this rank is allowed to persist in its part pkl."""
    keep = set(sample_indices_sub)
    if assigned_indices is not None:
        keep |= initial_keys
    return keep


def _filter_rank_res(res, keep_keys):
    return {k: res[k] for k in keep_keys if k in res}


def _load_historical_part_pkls(work_dir, stem):
    """Load all historical per-rank part pkls for this job (any world_size)."""
    existing_part_files = sorted(glob.glob(osp.join(work_dir, f'*_{stem}.pkl')))
    cache = {}
    done = set()
    for part_file in existing_part_files:
        if not osp.exists(part_file):
            continue
        part_res = load(part_file)
        if isinstance(part_res, dict):
            cache.update(part_res)
            done.update(part_res.keys())
    return existing_part_files, cache, done


def _detect_part_world_sizes(part_files, stem):
    """Infer world_size values from part pkl filenames like 032_ / 08_."""
    suffix = f'_{stem}.pkl'
    world_sizes = set()
    for path in part_files:
        name = osp.basename(path)
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix)]
        for ws in (1, 2, 4, 8, 16, 32, 64, 128):
            ws_s = str(ws)
            if not prefix.endswith(ws_s):
                continue
            rank_s = prefix[: -len(ws_s)]
            if rank_s == '' or rank_s.isdigit():
                rank = int(rank_s) if rank_s else 0
                if 0 <= rank < ws:
                    world_sizes.add(ws)
                    break
    return world_sizes


def _merge_video_part_pkls(work_dir, stem, current_world_size=None):
    """
    Merge per-rank part pkls into one dict.
    Re-scan the directory at merge time (do not rely on a stale snapshot).
    When the same index exists in both historical and current-world_size files,
    prefer the current run's files.
    """
    part_files = sorted(glob.glob(osp.join(work_dir, f'*_{stem}.pkl')))
    current_rank_files = []
    if current_world_size is not None and current_world_size > 0:
        current_rank_files = [
            osp.join(work_dir, f'{i}{current_world_size}_{stem}.pkl')
            for i in range(current_world_size)
        ]
    current_rank_files = [p for p in current_rank_files if osp.exists(p)]

    data_all = {}
    for part_file in part_files:
        if part_file in current_rank_files:
            continue
        part_res = load(part_file)
        if isinstance(part_res, dict):
            data_all.update(part_res)
    for part_file in current_rank_files:
        part_res = load(part_file)
        if isinstance(part_res, dict):
            data_all.update(part_res)
    return data_all, part_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    if getattr(model,'backend', None) == 'genai':
        if dataset.nframe > 0:
            print(
                'Gemini model (with genai backend) does not support nframe, '
                'will set its VIDEO_LLM to False to enable multi-image input for video.'
            )
            setattr(model, 'VIDEO_LLM', False)
        else:
            print('Gemini model (with genai backend) is a video-llm, '
                  'will reset fps setting in model to match the dataset.')
            setattr(model, 'fps', dataset.fps)
            print(f'The fps is set to {dataset.fps} for the model {model_name}.')
    elif getattr(model,'backend', None) == 'vertex':
        print('Gemini model (with vertex backend) does not support video input, '
              'will set its VIDEO_LLM to False to enable multi-image input for video.')
        setattr(model, 'VIDEO_LLM', False)

    packstr = 'pack' if getattr(dataset, 'pack', False) else 'nopack'
    build_prompt_input = [(samples_dict[idx], getattr(model, 'VIDEO_LLM', False)) for idx in indices]
    if dataset.nframe > 0:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_structs.pkl'
    else:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_structs.pkl'
    structs = track_progress_rich(
        dataset.build_prompt,
        tasks=build_prompt_input,
        nproc=api_nproc,
        save=struct_tmp_file,
        keys=indices,
    )

    if dataset.nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    structs = [struct for struct in structs if struct is not None]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
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
        historical_cache=None):
    res = load(out_file) if osp.exists(out_file) else {}

    rank, world_size = get_rank_and_world_size()
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    dataset_name = dataset.dataset_name

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    if assigned_indices is not None:
        sample_indices_sub = assigned_indices
    else:
        sample_indices_sub = sample_indices[rank::world_size]

    # Pre-fill from historical part pkls (possibly produced with a different world_size).
    if historical_cache:
        for idx in sample_indices_sub:
            if idx in historical_cache and idx not in res:
                res[idx] = historical_cache[idx]

    initial_keys = set(res.keys())
    keep_keys = _rank_res_keep_keys(sample_indices_sub, initial_keys, assigned_indices)

    def _dump_rank_res():
        dump(_filter_rank_res(res, keep_keys), out_file)

    if np.all([idx in res for idx in sample_indices_sub]):
        _dump_rank_res()
        return model
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
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
        assert world_size == 1
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        _dump_rank_res()
        return model

    assert not getattr(dataset, 'pack', False), 'Current model not supported pack mode!'
    if 'megabench' in dataset_name.lower() and 'llava_onevision' in model_name:
        print(
            'LLaVA-OneVision does not support Megabench dataset as video dataset, '
            'will set its VIDEO_LLM to False to enable multi-image input for video.'
        )
        setattr(model, 'VIDEO_LLM', False)

    for i, idx in tqdm(enumerate(sample_indices_subrem), total=len(sample_indices_subrem), desc=f"[Rank{rank}]"):
        if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
            if dataset.nframe > 0:
                if getattr(model, 'nframe', 0) != dataset.nframe:
                    print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                    setattr(model, 'nframe', dataset.nframe)
            elif getattr(model, 'fps', 0) == 0:
                raise ValueError(f'fps is not suitable for {model_name}')
            else:
                setattr(model, 'nframe', None)
        if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
            if dataset.fps > 0:
                if getattr(model, 'fps', 0) != dataset.fps:
                    print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                    setattr(model, 'fps', dataset.fps)
            elif getattr(model, 'nframe', 0) == 0:
                raise ValueError(f'nframe is not suitable for {model_name}')
            else:
                setattr(model, 'fps', None)
        if (
            'Qwen2-VL' in model_name
            or 'Qwen2.5-VL' in model_name
            or 'Qwen2.5-Omni' in model_name
        ):
            if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
                print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
            if getattr(model, 'fps', None) is None and dataset.fps > 0:
                print(f'using {model_name} default setting for video, dataset.fps is ommitted')
        if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
            dataset_name = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            if dataset.nframe == 0:
                raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
            struct = model.build_prompt(
                dataset.data.iloc[sample_map[idx]], dataset=dataset, video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        else:
            struct = dataset.build_prompt(
                sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        if struct is None:
            warnings.warn(f'[Rank{rank}] build_prompt returned None for index={idx}, skipped.')
            continue

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.error(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        if i % 10 == 0:
            torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        _dump_rank_res()

    missing = [idx for idx in sample_indices_sub if idx not in res]
    if missing:
        raise RuntimeError(
            f'[Rank{rank}] failed to save {len(missing)}/{len(sample_indices_sub)} predictions; '
            f'examples: {missing[:10]}'
        )
    _dump_rank_res()
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job_video(
        model,
        work_dir,
        model_name,
        dataset,
        result_file_name,
        verbose=False,
        api_nproc=4,
        use_vllm=False):

    dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    result_file = osp.join(work_dir, result_file_name)
    # Dump Predictions to Prev File if result file exists
    if osp.exists(result_file):
        return model

    stem = osp.splitext(result_file_name)[0]
    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{stem}.pkl')
    out_file = tmpl.format(rank)

    # -------- Dynamic load-balancing for video inference --------
    # Resume from historical per-rank part pkls (any world_size), then re-balance
    # the remaining samples evenly across the current world_size.
    assigned_indices = None
    historical_cache = {}
    assigned_indices_all = None
    existing_part_files = []

    if rank == 0:
        sample_indices = (
            list(dataset.videos)
            if getattr(dataset, 'pack', False)
            else list(dataset.data['index'])
        )
        existing_part_files, historical_cache, done_indices = _load_historical_part_pkls(work_dir, stem)
        if existing_part_files:
            hist_ws = sorted(_detect_part_world_sizes(existing_part_files, stem))
            print(f"===== Found {len(existing_part_files)} existing part files (any world_size) ======")
            if hist_ws:
                print(f"===== Historical world_size in part files: {hist_ws}; current world_size: {world_size} ======")
            for part_file in existing_part_files:
                print(f"===== Part File {part_file} ======")
        else:
            print("===== Found 0 existing part files ======")

        remaining_indices = [idx for idx in sample_indices if idx not in done_indices]
        print(f"===== Done {len(done_indices)} | Remaining {len(remaining_indices)} | Total {len(sample_indices)} ======")

        assigned_indices_all = [[] for _ in range(world_size)]
        for j, idx in enumerate(remaining_indices):
            assigned_indices_all[j % world_size].append(idx)

    if dist.is_available() and dist.is_initialized():
        obj_list = [assigned_indices_all, historical_cache]
        dist.broadcast_object_list(obj_list, src=0)
        assigned_indices_all, historical_cache = obj_list[0], obj_list[1]

    if assigned_indices_all is not None:
        assigned_indices = assigned_indices_all[rank]

    model = infer_data(
        model=model,
        model_name=model_name,
        work_dir=work_dir,
        dataset=dataset,
        out_file=out_file,
        verbose=verbose,
        api_nproc=api_nproc,
        use_vllm=use_vllm,
        assigned_indices=assigned_indices,
        historical_cache=historical_cache)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        sample_indices = (
            list(dataset.videos)
            if getattr(dataset, 'pack', False)
            else list(dataset.data['index'])
        )
        data_all, part_files_to_cleanup = _merge_video_part_pkls(
            work_dir, stem, current_world_size=world_size)

        meta = dataset.data
        if dataset_name == 'MMBench-Video' and getattr(dataset, 'pack', False):
            meta, vstats = dataset.load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            missing = [x for x in meta['index'] if x not in data_all]
            if missing:
                raise RuntimeError(
                    f'Incomplete video inference: {len(data_all)}/{len(sample_indices)} saved; '
                    f'missing {len(missing)} indices, examples: {missing[:20]}'
                )
            meta['prediction'] = [str(data_all[x]) for x in meta['index']]
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)
        # Clean up all partial files we used for this job (both historical and current).
        for part_file in sorted(set(part_files_to_cleanup)):
            try:
                if osp.exists(part_file):
                    os.remove(part_file)
            except Exception as e:
                print(f"[WARN] Failed to remove partial file {part_file}: {e}")
    return model
