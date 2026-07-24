import re
import warnings
import huggingface_hub
from huggingface_hub import snapshot_download
from ..smp import *
from .video_concat_dataset import ConcatVideoDataset
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import track_progress_rich
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .utils.tempcompass import *


FAIL_MSG = 'Failed to obtain answer via API.'


def _safe_model_name(x: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', str(x))

def _default_tempcompass_root() -> str:
    return os.environ.get('TEMPCOMPASS_ROOT', '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/tempcompass')


class TempCompass(ConcatVideoDataset):
    def __init__(
        self,
        dataset='TempCompass',
        nframe=0,
        fps=-1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
        **kwargs,
    ):
        self.DATASET_SETS[dataset] = ['TempCompass_MCQ', 'TempCompass_Captioning', 'TempCompass_YorN']
        super().__init__(
            dataset=dataset,
            nframe=nframe,
            fps=fps,
            frames_limit=frames_limit,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            check_extracted_frames=check_extracted_frames,
            **kwargs,
        )

    @classmethod
    def supported_datasets(cls):
        return ['TempCompass']

    def evaluate(self, eval_file, **judge_kwargs):
        result = super().evaluate(eval_file=eval_file, **judge_kwargs)
        result = result.reset_index().rename(columns={'index': 'dim.task_type'})
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        avg_dict = {}
        for idx, item in result.iterrows():
            dim, task_type = item['dim.task_type'].split('. ')
            if dim not in avg_dict:
                avg_dict[dim] = {'success': 0.0, 'overall': 0.0}
            if task_type not in avg_dict:
                avg_dict[task_type] = {'success': 0.0, 'overall': 0.0}
            if 'overall' not in avg_dict:
                avg_dict['overall'] = {'success': 0.0, 'overall': 0.0}
            avg_dict[dim]['success'] += item['success']
            avg_dict[dim]['overall'] += item['overall']
            avg_dict[task_type]['success'] += item['success']
            avg_dict[task_type]['overall'] += item['overall']
            avg_dict['overall']['success'] += item['success']
            avg_dict['overall']['overall'] += item['overall']
            result.loc[idx, 'acc'] = round(item['success'] / item['overall'] * 100, 2)
        for key, value in avg_dict.items():
            # 使用 loc 方法添加新行
            result.loc[len(result)] = {
                'dim.task_type': key,
                'success': value['success'],
                'overall': value['overall'],
                'acc': round(value['success'] / value['overall'] * 100, 2)
            }
        dump(result, score_file)
        return result


class TempCompass_MCQ(VideoBaseDataset):

    MD5 = '7efbb9e6d9dabacd22daf274852691dd'
    TYPE = 'Video-MCQ'

    def __init__(
        self,
        dataset='TempCompass_MCQ',
        nframe=0,
        fps=-1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
    ):
        self.type_data_list = {
            'multi-choice': ('multi-choice.json', './videos', '.mp4'),
            'caption_matching': ('caption_matching.json', './videos', '.mp4'),
        }
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels

    @classmethod
    def supported_datasets(cls):
        return ['TempCompass_MCQ']

    def prepare_dataset(self, dataset_name='TempCompass_MCQ', repo_id=_default_tempcompass_root()):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'] + item['suffix'])):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def read_parquet(pth):
                import pandas as pd
                for task_name in self.type_data_list.keys():
                    if not osp.exists(osp.join(pth, f'{task_name}.json')):
                        data = pd.read_parquet(osp.join(pth, task_name, 'test-00000-of-00001.parquet'))
                        data.to_json(osp.join(pth, f'{task_name}.json'), orient='records', lines=False)

            def unzip_videos(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'videos')):
                    zip_file = osp.join(pth, 'tempcompass_videos.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if osp.exists(data_file) and md5(data_file) == self.MD5:
                    return
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(osp.join(pth, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for data in json_data:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': v[1],
                            'suffix': v[2],
                            'video': data['video_id'],
                            'question': data['question'].split('\n')[0],
                            'answer': data['answer'],
                            'dim': data['dim'],
                            'candidates': data['question'].split('\n')[1:],
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            # 若 repo_id 是已存在的本地路径，直接使用，不下载
            if osp.exists(repo_id):
                dataset_path = repo_id
            elif modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            read_parquet(dataset_path)
            unzip_videos(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def qa_template(self, data):
        question = data['question'] + '\n' + '\n'.join(eval(data['candidates']))
        answer = data['answer']
        return question, answer

    def save_video_frames(self, line, verbose=False):
        vid_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        import decord
        vid = decord.VideoReader(vid_path)
        n_frames = int(len(vid))
        fps_val = float(vid.get_avg_fps())
        duration = (n_frames / fps_val) if fps_val > 0 else 0.0
        video_info = {
            'fps': fps_val,
            'n_frames': n_frames,
            'duration': duration,
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(line['video'])
        elif self.fps > 0:
            total_duration = duration
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames > self.frames_limit:
                warnings.warn(
                    f"Video `{line['video']}` requires {required_frames} frames at {self.fps} fps. "
                    f"Truncating to {self.frames_limit} frames."
                )
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_paths = self.frame_paths_fps(line['video'], self.frames_limit)
            else:
                step_size = fps_val / self.fps if self.fps > 0 else 1.0
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(line['video'], len(indices))

        if len(indices) == 0:
            indices = [max(0, n_frames // 2)]
            frame_paths = self.frame_paths_fps(line['video'], 1) if self.fps > 0 else self.frame_paths(line['video'])[:1]
        elif n_frames > 0:
            max_idx = n_frames - 1
            indices = [min(max(0, int(x)), max_idx) for x in indices]

        need_extract = self.check_extracted_frames and (
            not np.all([osp.exists(p) for p in frame_paths])
        )

        if need_extract:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    try:
                        im.save(pth)
                    except FileExistsError:
                        print(f"Error: {pth} 已经存在")
                        continue  # 如果是FileExistsError，继续处理下一个路径
                    except Exception as e:
                        print(f"Error: {e}")

        return frame_paths, indices, video_info

    def save_video_into_images(self, line):
        frame_paths, _, _ = self.save_video_frames(line)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line)
        question, answer = self.qa_template(line)
        message = []
        if video_llm:
            assert self.fps > 0
            actual_fps = (
                self.frames_limit / video_info['duration']
                if len(frames) == self.frames_limit and video_info['duration'] > 0
                else self.fps
            )
            message.append(dict(
                type='video',
                value=frames,
                sample_fps=actual_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            ))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value=question))
        message.append(dict(type='text', value='\nPlease directly give the best option:'))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model_name = judge_kwargs.get('model', 'exact_matching')

        score_file = get_intermediate_file_path(eval_file, f'_{_safe_model_name(model_name)}_score')
        tmp_file = get_intermediate_file_path(eval_file, f'_{_safe_model_name(model_name)}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 1)

        if not osp.exists(score_file):
            data = load(eval_file)
            if model_name != 'exact_matching':
                try:
                    judge_model = build_judge(system_prompt=sys_prompt, **judge_kwargs)
                except Exception as e:
                    warnings.warn(f'Failed to build judge model ({model_name}), fallback to exact matching: {type(e)}: {e}')
                    warnings.warn(DEBUG_MESSAGE)
                    judge_model = None
            else:
                judge_model = None

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(judge_model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    evaluate_tempcompass_mcq,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            for idx, item in data.iterrows():
                data.loc[idx, 'score'] = ans[idx]['rating']
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating


class TempCompass_Captioning(VideoBaseDataset):

    MD5 = '35be9bf2581ea7767f02e9a8f37ae1ab'
    TYPE = 'Video-VQA'

    def __init__(
        self,
        dataset='TempCompass_Captioning',
        nframe=0,
        fps=-1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
    ):
        self.type_data_list = {
            'captioning': ('captioning.json', './videos', '.mp4'),
        }
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels

    @classmethod
    def supported_datasets(cls):
        return ['TempCompass_Captioning']

    def prepare_dataset(self, dataset_name='TempCompass_Captioning', repo_id=_default_tempcompass_root()):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'] + item['suffix'])):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def read_parquet(pth):
                import pandas as pd
                for task_name in self.type_data_list.keys():
                    if not osp.exists(osp.join(pth, f'{task_name}.json')):
                        data = pd.read_parquet(osp.join(pth, task_name, 'test-00000-of-00001.parquet'))
                        data.to_json(osp.join(pth, f'{task_name}.json'), orient='records', lines=False)

            def unzip_videos(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'videos')):
                    zip_file = osp.join(pth, 'tempcompass_videos.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if osp.exists(data_file) and md5(data_file) == self.MD5:
                    return
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(osp.join(pth, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for data in json_data:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': v[1],
                            'suffix': v[2],
                            'video': data['video_id'],
                            'question': data['question'],
                            'answer': data['answer'],
                            'dim': data['dim'],
                            'mc_question': data['mc_question'],
                            'mc_answer': data['mc_answer'],
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            # 若 repo_id 是已存在的本地路径，直接使用，不下载
            if osp.exists(repo_id):
                dataset_path = repo_id
            elif modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            read_parquet(dataset_path)
            unzip_videos(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def qa_template(self, data):
        question = data['question']
        answer = data['answer']
        return question, answer

    def save_video_frames(self, line, verbose=False):
        vid_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        import decord
        vid = decord.VideoReader(vid_path)
        n_frames = int(len(vid))
        fps_val = float(vid.get_avg_fps())
        duration = (n_frames / fps_val) if fps_val > 0 else 0.0
        video_info = {'fps': fps_val, 'n_frames': n_frames, 'duration': duration}
        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(line['video'])
        elif self.fps > 0:
            total_duration = duration
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames > self.frames_limit:
                warnings.warn(
                    f"Video `{line['video']}` requires {required_frames} frames at {self.fps} fps. "
                    f"Truncating to {self.frames_limit} frames."
                )
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_paths = self.frame_paths_fps(line['video'], self.frames_limit)
            else:
                step_size = fps_val / self.fps if self.fps > 0 else 1.0
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(line['video'], len(indices))
        else:
            indices = []
            frame_paths = []

        if len(indices) == 0:
            indices = [max(0, n_frames // 2)]
            frame_paths = self.frame_paths_fps(line['video'], 1) if self.fps > 0 else self.frame_paths(line['video'])[:1]
        elif n_frames > 0:
            max_idx = n_frames - 1
            indices = [min(max(0, int(x)), max_idx) for x in indices]

        need_extract = self.check_extracted_frames and (
            not np.all([osp.exists(p) for p in frame_paths])
        )

        if need_extract:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    try:
                        im.save(pth)
                    except FileExistsError:
                        print(f"Error: {pth} 已经存在")
                        continue  # 如果是FileExistsError，继续处理下一个路径
                    except Exception as e:
                        print(f"Error: {e}")

        return frame_paths, indices, video_info

    def save_video_into_images(self, line):
        frame_paths, _, _ = self.save_video_frames(line)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line)
        question, answer = self.qa_template(line)
        message = []
        if video_llm:
            assert self.fps > 0
            actual_fps = (
                self.frames_limit / video_info['duration']
                if len(frames) == self.frames_limit and video_info['duration'] > 0
                else self.fps
            )
            message.append(dict(
                type='video',
                value=frames,
                sample_fps=actual_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            ))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value=question))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model_name = judge_kwargs.get('model', 'qwen3-235b-a22b-thinking-2507')

        score_file = get_intermediate_file_path(eval_file, f'_{_safe_model_name(model_name)}_score')
        tmp_file = get_intermediate_file_path(eval_file, f'_{_safe_model_name(model_name)}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 1)
        # pkl 存在时也强制重新测评：--judge-args '{"force_reeval": true}'
        force_reeval = judge_kwargs.pop('force_reeval', False)
        if force_reeval:
            warnings.warn('force_reeval=True: ignoring existing pkl/score cache and re-running captioning judge.')

        if not osp.exists(score_file) or force_reeval:
            data = load(eval_file)
            if model_name not in (None, 'exact_matching'):
                try:
                    print("try to build judge model")
                    judge_model = build_judge(system_prompt=sys_prompt, **judge_kwargs)
                    print("judge model built successfully")
                except Exception as e:
                    warnings.warn(f'Failed to build judge model ({model_name}): {type(e)}: {e}')
                    warnings.warn(DEBUG_MESSAGE)
                    judge_model = None
            else:
                judge_model = None

            if judge_model is None:
                warnings.warn(
                    'TempCompass Captioning requires an LLM judge. '
                    'Pass --judge <model> (e.g. --judge gpt-4o-mini or --judge chatgpt-1106). '
                    'Skipping judge and assigning score 0 to all captioning samples. '
                    'Captioning 全 0 通常是因为未传 --judge 或 build_judge 失败。'
                )

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(judge_model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file) and not force_reeval:
                ans = load(tmp_file)

            def _eval_captioning(model, line):
                return evaluate_tempcompass_captioning(
                    model, line
                )

            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            n_total = len(data)
            n_cached = len(ans)
            print(f'{len(indices)} indices to evaluate (total={n_total}, already cached in pkl={n_cached})')

            if len(indices):
                print("try to evaluate captioning")
                print("judge_model: ", judge_model)
                if judge_model is not None:
                    _ = track_progress_rich(
                        _eval_captioning,
                        tups,
                        nproc=nproc,
                        chunksize=nproc,
                        keys=indices,
                        save=tmp_file,
                    )
                else:
                    for i in indices:
                        ans[i] = {'rating': 0}
                    dump(ans, tmp_file)
            ans = load(tmp_file)
            for idx, item in data.iterrows():
                data.loc[idx, 'score'] = ans[idx]['rating']
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating


class TempCompass_YorN(VideoBaseDataset):

    MD5 = 'c72c046d7fa0e82c8cd7462f2e844ea8'
    TYPE = 'Video-Y/N'

    def __init__(
        self,
        dataset='TempCompass_YorN',
        nframe=0,
        fps=-1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
    ):
        self.type_data_list = {
            'yes_no': ('yes_no.json', './videos', '.mp4'),
        }
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps, check_extracted_frames=check_extracted_frames
        )
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels

    @classmethod
    def supported_datasets(cls):
        return ['TempCompass_YorN']

    def prepare_dataset(self, dataset_name='TempCompass_YorN', repo_id=_default_tempcompass_root()):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'] + item['suffix'])):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def read_parquet(pth):
                import pandas as pd
                for task_name in self.type_data_list.keys():
                    if not osp.exists(osp.join(pth, f'{task_name}.json')):
                        data = pd.read_parquet(osp.join(pth, task_name, 'test-00000-of-00001.parquet'))
                        data.to_json(osp.join(pth, f'{task_name}.json'), orient='records', lines=False)

            def unzip_videos(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'videos')):
                    zip_file = osp.join(pth, 'tempcompass_videos.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if osp.exists(data_file) and md5(data_file) == self.MD5:
                    return
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(osp.join(pth, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for data in json_data:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': v[1],
                            'suffix': v[2],
                            'video': data['video_id'],
                            'question': data['question'].split('\n')[0],
                            'answer': data['answer'],
                            'dim': data['dim']
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            # 若 repo_id 是已存在的本地路径，直接使用，不下载
            if osp.exists(repo_id):
                dataset_path = repo_id
            elif modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            read_parquet(dataset_path)
            unzip_videos(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def qa_template(self, data):
        question = data['question']
        answer = data['answer']
        return question, answer

    def save_video_frames(self, line, verbose=False):
        vid_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        import decord
        vid = decord.VideoReader(vid_path)
        n_frames = int(len(vid))
        fps_val = float(vid.get_avg_fps())
        duration = (n_frames / fps_val) if fps_val > 0 else 0.0
        video_info = {'fps': fps_val, 'n_frames': n_frames, 'duration': duration}
        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(line['video'])
        elif self.fps > 0:
            total_duration = duration
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames > self.frames_limit:
                warnings.warn(
                    f"Video `{line['video']}` requires {required_frames} frames at {self.fps} fps. "
                    f"Truncating to {self.frames_limit} frames."
                )
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_paths = self.frame_paths_fps(line['video'], self.frames_limit)
            else:
                step_size = fps_val / self.fps if self.fps > 0 else 1.0
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(line['video'], len(indices))
        else:
            indices = []
            frame_paths = []

        if len(indices) == 0:
            indices = [max(0, n_frames // 2)]
            frame_paths = self.frame_paths_fps(line['video'], 1) if self.fps > 0 else self.frame_paths(line['video'])[:1]
        elif n_frames > 0:
            max_idx = n_frames - 1
            indices = [min(max(0, int(x)), max_idx) for x in indices]

        need_extract = self.check_extracted_frames and (
            not np.all([osp.exists(p) for p in frame_paths])
        )

        if need_extract:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    try:
                        im.save(pth)
                    except FileExistsError:
                        print(f"Error: {pth} 已经存在")
                        continue  # 如果是FileExistsError，继续处理下一个路径
                    except Exception as e:
                        print(f"Error: {e}")

        return frame_paths, indices, video_info

    def save_video_into_images(self, line):
        frame_paths, _, _ = self.save_video_frames(line)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line)
        question, answer = self.qa_template(line)
        message = []
        if video_llm:
            assert self.fps > 0
            actual_fps = (
                self.frames_limit / video_info['duration']
                if len(frames) == self.frames_limit and video_info['duration'] > 0
                else self.fps
            )
            message.append(dict(
                type='video',
                value=frames,
                sample_fps=actual_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            ))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value=question))
        message.append(dict(type='text', value='\nPlease answer yes or no:'))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model_name = judge_kwargs.get('model', 'exact_matching')
        # judge_kwargs.update({
        #     "max_tokens": 128,
        #     "temperature": 1.0,
        #     "top_p": 1,
        #     "presence_penalty": 1,
        # })

        score_file = get_intermediate_file_path(eval_file, f'_{_safe_model_name(model_name)}_score')
        tmp_file = get_intermediate_file_path(eval_file, f'_{_safe_model_name(model_name)}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 1)

        if not osp.exists(score_file):
            data = load(eval_file)
            if model_name != 'exact_matching':
                try:
                    judge_model = build_judge(system_prompt=sys_prompt, **judge_kwargs)
                except Exception as e:
                    warnings.warn(f'Failed to build judge model ({model_name}), fallback to exact matching: {type(e)}: {e}')
                    warnings.warn(DEBUG_MESSAGE)
                    judge_model = None
            else:
                judge_model = None

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(judge_model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    evaluate_tempcompass_YorN,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            for idx, item in data.iterrows():
                data.loc[idx, 'score'] = ans[idx]['rating']
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating
