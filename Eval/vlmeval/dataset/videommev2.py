import ast
import errno
import json
import math
import os
import os.path as osp
import warnings

from huggingface_hub import snapshot_download
from PIL import Image

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .utils import DEBUG_MESSAGE, build_judge
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


# ──────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────
def cal_relevance(scores):
    score_map_exponential = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    correct_count = sum(scores)
    exp_score = score_map_exponential.get(correct_count, 0.0)
    linear_score = correct_count * 25.0
    return exp_score, linear_score


def cal_logic(scores, group_structure):
    group_structure_list = ast.literal_eval(group_structure)
    last_correct_idx = -1
    for idx, val in enumerate(scores):
        if val:
            last_correct_idx = idx
        else:
            break
    if group_structure_list == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif group_structure_list == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif group_structure_list == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(f'Unknown group_structure_list: {group_structure_list}')
    logic_score = score_map.get(last_correct_idx + 1, 0.0)
    return logic_score


def get_final_rating(score_file):
    data = load(score_file)
    all_groups = [[] for _ in range((len(data) + 1) // 4)]
    final_rating = {
        'level_1': [],
        'level_2': [],
        'level_3': [],
        'relevance_score': [],
        'relevance_linear_score': [],
        'logic_score': [],
        'total': [],
    }
    second_head_rating = {}
    third_head_rating = {}
    for i in range(len(data)):
        level, group_type, group_structure, score, second_head, third_head = (
            data.loc[i, 'level'],
            data.loc[i, 'group_type'],
            data.loc[i, 'group_structure'],
            data.loc[i, 'score'],
            data.loc[i, 'second_head'],
            data.loc[i, 'third_head'],
        )
        all_groups[i // 4].append((level, group_type, group_structure, score, second_head, third_head))
    for group in all_groups:
        level = group[-1][0]
        group_type = group[-1][1]
        group_structure = group[-1][2]
        second_head = group[-1][4]
        third_head = group[-1][5]
        scores = [item[3] for item in group]
        if group_type == 'relevance':
            exp_score, linear_score = cal_relevance(scores)
            final_rating['relevance_score'].append(exp_score)
            final_rating['relevance_linear_score'].append(linear_score)
        elif group_type == 'logic':
            exp_score = cal_logic(scores, group_structure)
            final_rating['logic_score'].append(exp_score)
        else:
            raise ValueError(f'Unknown group_type: {group_type}')
        if level is not None and str(level) != 'None':
            final_rating[f'level_{int(level)}'].append(exp_score)
        final_rating['total'].append(exp_score)
        if second_head not in second_head_rating:
            second_head_rating[second_head] = []
        second_head_rating[second_head].append(exp_score)
        if third_head not in third_head_rating:
            third_head_rating[third_head] = []
        third_head_rating[third_head].append(exp_score)
    for key in final_rating:
        final_rating[key] = sum(final_rating[key]) / len(final_rating[key]) if len(final_rating[key]) > 0 else 0.0
    n_q = len(data)
    if n_q > 0:
        n_correct = int((data['score'] == 1).sum())
        final_rating['average_accuracy'] = n_correct / n_q
    else:
        final_rating['average_accuracy'] = 0.0
    for key in second_head_rating:
        second_head_rating[key] = (
            sum(second_head_rating[key]) / len(second_head_rating[key])
            if len(second_head_rating[key]) > 0 else 0.0
        )
    for key in third_head_rating:
        third_head_rating[key] = (
            sum(third_head_rating[key]) / len(third_head_rating[key])
            if len(third_head_rating[key]) > 0 else 0.0
        )
    return {
        'final_rating': final_rating,
        'second_head_rating': second_head_rating,
        'third_head_rating': third_head_rating,
    }


# ──────────────────────────────────────────────
# Subtitle helpers (JSONL with word-level timestamps)
# ──────────────────────────────────────────────
def load_subtitle_jsonl(subtitle_path):
    """Load a JSONL subtitle file. Each line: {"text": str, "start_time": float, "end_time": float}."""
    if not osp.exists(subtitle_path):
        return None
    entries = []
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def subtitle_concat_all(entries):
    """Concatenate all word texts from subtitle entries into a single string."""
    if not entries:
        return ''
    return ' '.join(e['text'] for e in entries)


def subtitle_between_timestamps(entries, start_time, end_time):
    """Collect words whose time range overlaps [start_time, end_time)."""
    if not entries:
        return ''
    words = []
    for e in entries:
        if e['end_time'] >= start_time and e['start_time'] < end_time:
            words.append(e['text'])
    return ' '.join(words)


class VideoMMEv2(VideoBaseDataset):

    # --- Text prompts (frame description) ---
    WO_SUB_PROMPT = 'These are the frames of a video.'
    WITH_SUB_PROMPT = (
        'These are the frames of a video. '
        "This video's subtitles are listed below:\n{}\n"
    )
    WITH_SUB_PROMPT_INTERLEAVE = (
        'These are the frames of a video with corresponding subtitles shown between frames. '
        'The subtitles indicate what is being said during the time interval between adjacent frames.'
    )

    # --- Response prompts ---
    THINK_PROMPT = (
        'Please perform a detailed reasoning based on the provided video frames to answer the following '
        'multiple-choice question selecting the best option from A through H and providing your final response '
        "strictly in the format: 'Final Answer: <letter>'."
    )
    INSTRUCT_PROMPT = (
        'Select the best answer to the following multiple-choice question based on the video. '
        'Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.'
    )

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='Video-MME-v2', nframe=0, fps=-1,
                 with_subtitle=False, subtitle_interleave=False, reasoning=False,
                 resize_target_area=None, frames_limit=2048,
                 min_pixels=28 * 28, max_pixels=448 * 448,
                 total_pixels=32000 * 2 * 4 * 14 * 14,
                 check_extracted_frames=True):
        super().__init__(
            dataset=dataset, nframe=nframe, fps=fps,
            check_extracted_frames=check_extracted_frames,
        )
        self.dataset_name = dataset
        self.with_subtitle = with_subtitle
        self.subtitle_interleave = subtitle_interleave
        self.reasoning = reasoning
        self.response_prompt = self.THINK_PROMPT if reasoning else self.INSTRUCT_PROMPT
        # resize_target_area: False to disable, or int (e.g. 448*448=200704) to enable
        self.resize_target_area = resize_target_area
        # Following Qwen3-VL: impose a cap of frames per video (mainly for fps sampling).
        self.frames_limit = frames_limit
        # Same as VideoMME: passed to video-native backends (e.g. Qwen3-VL) for token/pixel budgets.
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        if self.resize_target_area:
            # Use a separate frame cache directory for resized frames to avoid conflicts
            self.frame_root_resize = self.frame_root + f'_resize{self.resize_target_area}'
            os.makedirs(self.frame_root_resize, exist_ok=True)

    @classmethod
    def supported_datasets(cls):
        return ['Video-MME-v2']

    def prepare_dataset(
        self,
        dataset_name='Video-MME-v2',
        repo_id=None,
    ):
        """Resolve data root: local path, HF cache, ModelScope, or ``snapshot_download``.

        Override install root with env ``VIDEO_MME_V2_ROOT`` (directory or HF repo id string).
        """
        if repo_id is None:
            repo_id = os.environ.get(
                'VIDEO_MME_V2_ROOT',
                '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/Video-MME-v2',
            )

        def tsv_paths_for(pth):
            """Prefer TSV next to media; mirror under LMUData when benchmark root is read-only."""
            return (
                osp.join(pth, f'{dataset_name}.tsv'),
                osp.join(LMUDataRoot(), f'{dataset_name}.tsv'),
            )

        def valid_tsv_path(pth):
            for data_file in tsv_paths_for(pth):
                if osp.exists(data_file):
                    print('valid_tsv_path: ', data_file)
                    return data_file
            return None

        def check_integrity(pth):
            data_file = valid_tsv_path(pth)
            if data_file is None:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
            data_file = valid_tsv_path(dataset_path)
        else:

            def unzip_hf_zip(pth):
                import zipfile
                base_dir = pth
                video_zip_dir = osp.join(base_dir, 'videos')
                target_dir = osp.join(base_dir, 'video')

                if osp.isdir(video_zip_dir) and not osp.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    zip_files = sorted([
                        osp.join(video_zip_dir, f) for f in os.listdir(video_zip_dir)
                        if f.endswith('.zip')
                    ])
                    for zip_file in zip_files:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                if not member.endswith('/'):
                                    source = zip_ref.open(member)
                                    target = open(osp.join(target_dir, osp.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    print('The video files have been extracted from zip files.')
                elif osp.exists(target_dir):
                    print('The video directory already exists.')

                subtitle_zip = osp.join(base_dir, 'subtitle.zip')
                subtitle_dir = osp.join(base_dir, 'subtitle')
                if osp.isfile(subtitle_zip) and not osp.exists(subtitle_dir):
                    os.makedirs(subtitle_dir, exist_ok=True)
                    with zipfile.ZipFile(subtitle_zip, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            if not member.endswith('/'):
                                source = zip_ref.open(member)
                                target = open(osp.join(subtitle_dir, osp.basename(member)), 'wb')
                                with source, target:
                                    target.write(source.read())
                    print('The subtitle files have been extracted from subtitle.zip.')
                elif osp.exists(subtitle_dir):
                    print('The subtitle directory already exists.')

            def generate_tsv(pth):
                ok = valid_tsv_path(pth)
                if ok is not None:
                    return ok

                in_tree, in_lmu = tsv_paths_for(pth)
                data_file = in_tree if os.access(pth, os.W_OK) else in_lmu

                pq = osp.join(pth, 'test.parquet')
                if not osp.exists(pq):
                    raise FileNotFoundError(
                        f'Expected {pq} under dataset root. '
                        'For HF dataset MME-Benchmarks/Video-MME-v2, run snapshot_download first '
                        'or set VIDEO_MME_V2_ROOT to a prepared directory.'
                    )
                df = pd.read_parquet(pq)
                df = df.assign(index=range(len(df)))

                def _norm_vid(v):
                    s = str(v).strip()
                    try:
                        return str(int(s)).zfill(3)
                    except ValueError:
                        return s

                df['video'] = df['video_id'].apply(_norm_vid)
                df['video_path'] = df['video'].apply(lambda x: f'./video/{x}.mp4')
                df['subtitle_path'] = df['video'].apply(lambda x: f'./subtitle/{x}.jsonl')

                df = df[[
                    'index', 'video', 'video_path', 'subtitle_path',
                    'url', 'group_type', 'group_structure',
                    'question_id', 'question', 'options', 'answer',
                    'level', 'second_head', 'third_head',
                ]]
                try:
                    df.to_csv(data_file, sep='\t', index=False)
                except OSError as e:
                    en = getattr(e, 'errno', None)
                    if data_file == in_tree and en in (errno.EPERM, errno.EACCES, errno.EROFS):
                        df.to_csv(in_lmu, sep='\t', index=False)
                        return in_lmu
                    raise
                return data_file

            if Path(repo_id).exists():
                dataset_path = repo_id
            elif modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_hf_zip(dataset_path)
            data_file = generate_tsv(dataset_path)

        return dict(data_file=data_file, root=dataset_path)

    @staticmethod
    def _vid_str(video):
        """Ensure video id is a zero-padded 3-digit string (e.g. 1 → '001')."""
        return str(int(video)).zfill(3)

    @staticmethod
    def _resize_to_target_area(img, target_area, divisor=16):
        """Resize a PIL Image keeping aspect ratio so that H*W ≈ target_area,
        with both dimensions aligned to `divisor`."""
        w, h = img.size
        scale = math.sqrt(target_area / (w * h))
        new_w = max(divisor, round(w * scale / divisor) * divisor)
        new_h = max(divisor, round(h * scale / divisor) * divisor)
        if new_w == w and new_h == h:
            return img
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _frame_paths_resize(self, video):
        """Frame paths under the resize-specific cache directory."""
        frame_root = osp.join(self.frame_root_resize, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe))
                for i in range(1, self.nframe + 1)]

    def _frame_paths_fps_resize(self, video, num_frames):
        """Frame paths (fps mode) under the resize-specific cache directory."""
        frame_root = osp.join(self.frame_root_resize, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl_fps.format(i, num_frames, self.fps))
                for i in range(1, num_frames + 1)]

    def _frame_paths_count(self, video, num_frames):
        """Frame paths (count mode) under the default cache directory."""
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames))
                for i in range(1, num_frames + 1)]

    def _frame_paths_count_resize(self, video, num_frames):
        """Frame paths (count mode) under the resize-specific cache directory."""
        frame_root = osp.join(self.frame_root_resize, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames))
                for i in range(1, num_frames + 1)]

    def save_video_frames(self, video, video_llm=False, return_info=False, verbose=False):
        """Extract frames to disk (unless ``return_info`` short-circuits).

        If ``return_info`` is True: return ``(frame_paths, indices, resize_target_area)``
        without decoding/writing — used by pre-extract workers; third value is **not**
        ``video_info``. If False: write missing frames and return
        ``(frame_paths, indices, video_info)`` with dict keys fps / n_frames / duration.
        """
        video = self._vid_str(video)
        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
            'duration': len(vid) / vid.get_avg_fps() if vid.get_avg_fps() > 0 else 0.0,
        }
        if verbose:
            print('video_info: ', video_info)
        if self.nframe > 0 and self.fps < 0:
            target_n = self.nframe
            if self.frames_limit and target_n > self.frames_limit:
                warnings.warn(
                    f'Video `{video}` requests nframe={target_n} which exceeds frames_limit={self.frames_limit}. '
                    f'Truncating to {self.frames_limit} frames.'
                )
                target_n = self.frames_limit
            step_size = len(vid) / (target_n + 1)
            indices = [int(i * step_size) for i in range(1, target_n + 1)]
            # Use count-based naming so cache is consistent with the actual number of frames extracted.
            if self.resize_target_area:
                frame_paths = self._frame_paths_count_resize(video, target_n)
            else:
                frame_paths = self._frame_paths_count(video, target_n)
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            if self.frames_limit and required_frames > self.frames_limit:
                warnings.warn(
                    f'Video `{video}` requires {required_frames} frames with fps={self.fps}. '
                    f'Truncating to frames_limit={self.frames_limit} frames.'
                )
                step_size = video_info['n_frames'] / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                # When truncated, use count-based naming (not fps-based), since effective fps changes.
                if self.resize_target_area:
                    frame_paths = self._frame_paths_count_resize(video, self.frames_limit)
                else:
                    frame_paths = self._frame_paths_count(video, self.frames_limit)
            else:
                step_size = video_info['fps'] / self.fps
                indices = [int(i * step_size) for i in range(required_frames)]
                if self.resize_target_area:
                    frame_paths = self._frame_paths_fps_resize(video, len(indices))
                else:
                    frame_paths = self.frame_paths_fps(video, len(indices))

        if return_info:
            return frame_paths, indices, self.resize_target_area

        if not self.check_extracted_frames:
            return frame_paths, indices, video_info

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            if self.resize_target_area:
                images = [self._resize_to_target_area(im, self.resize_target_area)
                          for im in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)
        elif verbose:
            print(f'Video `{video}` frames already exist.')

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video = self._vid_str(line['video'])

        # ── video_llm mode ──
        if video_llm:
            # --- (kept for reference) pass raw video file directly ---
            # vid_path = osp.join(self.data_root, 'video', video + '.mp4')
            #
            # # Compute resized dimensions targeting ~448×448 area, keeping aspect ratio
            # import decord
            # vid = decord.VideoReader(vid_path)
            # h, w = vid[0].shape[:2]
            # del vid
            # target_area = 448 * 448
            # divisor = 16
            # scale = math.sqrt(target_area / (w * h))
            # new_w = max(divisor, round(w * scale / divisor) * divisor)
            # new_h = max(divisor, round(h * scale / divisor) * divisor)
            #
            # video_msg = dict(type='video', value=vid_path, resized_height=new_h, resized_width=new_w)
            # # Pass frame sampling parameters so the model follows dataset settings
            # if self.nframe > 0:
            #     video_msg['nframes'] = self.nframe
            # if self.fps > 0:
            #     video_msg['fps'] = self.fps
            # message = [video_msg]

            # --- pass video frames directly (like videomme.py) ---
            frames, indices, video_info = self.save_video_frames(
                video, video_llm=True, return_info=False)

            video = self._vid_str(video)
            vid_path = osp.join(self.data_root, 'video', video + '.mp4')

            actual_fps = None
            if self.fps > 0 and video_info.get('duration', 0) > 0:
                required_frames = int(video_info['duration'] * self.fps)
                if self.frames_limit and required_frames > self.frames_limit and len(frames) == self.frames_limit:
                    actual_fps = self.frames_limit / video_info['duration']
                else:
                    actual_fps = self.fps
            elif video_info.get('duration', 0) > 0 and len(frames) > 0:
                # nframe-only configs (fps < 0): uniform spacing over full duration
                actual_fps = len(frames) / video_info['duration']
            message = []


            # if os.getenv('VIDEO_INPUT_MODE') == 'frame':
            video_msg = dict(
                type='video',
                value=frames,
                sample_fps=actual_fps,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                total_pixels=self.total_pixels,
            )
            # else:
            #     video_msg = dict(
            #         type='video',
            #         value=vid_path,
            #         fps=self.fps,
            #         max_frames=self.frames_limit
            #     )

            message.append(video_msg)

            # Subtitle handling (non-interleave only, since we don't have frame timestamps)
            if self.with_subtitle:
                sub_path = osp.join(self.data_root, 'subtitle', video + '.jsonl')
                sub_entries = load_subtitle_jsonl(sub_path)
                full_text = subtitle_concat_all(sub_entries)
                text_prompt = self.WITH_SUB_PROMPT.format(full_text)
            else:
                text_prompt = self.WO_SUB_PROMPT

            message.append(dict(type='text', value=text_prompt))
            question_text = line['question'] + '\n' + line['options']
            prompt = 'Question: {}\n'.format(question_text)
            message.append(dict(type='text', value=prompt + self.response_prompt))
            return message

        # ── Frame-based mode: extract frames as images ──
        frames, indices, video_info = self.save_video_frames(
            video, video_llm, return_info=False)

        fps = video_info['fps']
        frame_timestamps = [idx / fps for idx in indices]

        # Load subtitle JSONL if needed
        sub_entries = None
        if self.with_subtitle:
            sub_path = osp.join(self.data_root, 'subtitle', video + '.jsonl')
            sub_entries = load_subtitle_jsonl(sub_path)

        message = []

        if self.with_subtitle and self.subtitle_interleave:
            # ── Interleave mode: image → subtitle_text → image → subtitle_text → ...
            for i, (im, frame_ts) in enumerate(zip(frames, frame_timestamps)):
                if i < len(frame_timestamps) - 1:
                    start_ts, end_ts = frame_ts, frame_timestamps[i + 1]
                else:
                    start_ts = frame_ts
                    end_ts = video_info['n_frames'] / fps

                message.append(dict(type='image', value=im))

                chunk = subtitle_between_timestamps(sub_entries, start_ts, end_ts)
                if chunk:
                    message.append(dict(
                        type='text',
                        value=f'[Subtitle {start_ts:.2f}s - {end_ts:.2f}s]: {chunk}',
                    ))

            text_prompt = self.WITH_SUB_PROMPT_INTERLEAVE

        else:
            # ── Non-interleave: all frames first ──
            for im in frames:
                message.append(dict(type='image', value=im))

            if self.with_subtitle:
                full_text = subtitle_concat_all(sub_entries)
                text_prompt = self.WITH_SUB_PROMPT.format(full_text)
            else:
                text_prompt = self.WO_SUB_PROMPT

        message.append(dict(type='text', value=text_prompt))

        question_text = line['question'] + '\n' + line['options']
        prompt = 'Question: {}\n'.format(question_text)
        message.append(dict(type='text', value=prompt + self.response_prompt))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.videomme import extract_characters_regex_v2
        from .utils.videomme import extract_option_v2 as extract_option

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be a supported format (xlsx/json/tsv) file'
        )

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in [
                'chatgpt-0125', 'exact_matching', 'gpt-4-0125',
                'qwen3-235b-a22b-instruct-2507', 'qwen3-235b-a22b-thinking-2507',
            ]

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None

            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
                row = data.loc[data['index'] == idx].to_dict(orient='records')[0]
                # extract_option_v2 expects multi-line ``question`` (stem + option lines)
                q_full = f"{row['question']}\n{row['options']}"
                row_for_opt = {**row, 'question': q_full}

                if extract_characters_regex_v2(pred) == '':
                    extract_pred = extract_option(model, row_for_opt, 'Video-MME-v2')
                    data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[data['index'] == idx, 'score'] = int(extract_characters_regex_v2(pred) == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for '
                f'{len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, '
                f'and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_final_rating(score_file)
        dump(rating, tgt_file)
        return rating