import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-extract video frames using VLMEvalKit's built-in logic."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="VideoMMMU_2fps_limit_512_768px_80kctx",
        help=(
            "Dataset name used by VLMEvalKit, e.g. "
            "`VideoMMMU_2fps_limit_512_768px_80kctx`. "
            "It should be a key in `supported_video_datasets` or an officially supported dataset name. "
            "VUE_TR_* / VUE_TR_V2_* dedupe by `video_id` (run_vue_tr). "
            "MomentSeeker_* uses `run_moment_seeker` → `MomentSeeker.save_video_frames` (also dedupe by `video_id`). "
            "VideoEval-Pro_* dedupe by `video` (run_videoevalpro)."
        ),
    )
    return parser.parse_args()

def run_videomme(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
    # for i, (_, line) in enumerate(data.iloc[::-1].iterrows()):
        video_pth = line["video"] if "video" in line else None

        print(f"[{i+1}/{len(data)}] Extracting frames for video...")

        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")


def run_videommmu(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
        sample_id = line["id"]
        video_pth = line["video"]

        print(f"[{i+1}/{len(data)}] Extracting frames for video: {sample_id}...")

        dataset.save_video_frames(sample_id, video_pth, verbose=True)

    print("[Done] Frame extraction finished.")


from concurrent.futures import ThreadPoolExecutor, as_completed

def run_longvideobench(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
        video_pth = line["video_path"] if "video" in line else None

        print(f"[{i+1}/{len(data)}] Extracting frames for video...")

        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")

def run_lvbench(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
        video_pth = line["video_path"] if "video" in line else None

        print(f"[{i+1}/{len(data)}] Extracting frames for video...")

        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")


def run_videoevalpro(dataset):
    """
    VideoEval-Pro: multiple QA rows per clip; dedupe by `video` (relative path under data root).
    Same paths as `VideoEvalPro.build_prompt` → `save_video_frames(line['video'], ...)`.
    """
    data = dataset.data
    print(f"[Info] Total rows: {len(data)}")
    if 'video' not in data.columns:
        raise ValueError("[VideoEval-Pro] dataset TSV must contain `video` column.")
    uniq = data.drop_duplicates('video').reset_index(drop=True)
    print(f"[Info] Unique videos to process: {len(uniq)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, (_, line) in enumerate(uniq.iterrows()):
    # for i, (_, line) in enumerate(uniq[::-1].iterrows()):
        video_rel = line['video']
        print(f"[{i+1}/{len(uniq)}] Extracting frames for video: {video_rel}...")
        dataset.save_video_frames(video_rel, video_llm=False, verbose=True)
    print("[Done] Frame extraction finished.")


def run_mmvu(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
        video_pth = line['video']
        print(f"[{i+1}/{len(data)}] Extracting frames for video: {video_pth}...")
        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")

def run_tomato(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    # for i, (_, line) in enumerate(data.iterrows()):
    for i, (_, line) in enumerate(data.iloc[::-1].iterrows()):
        video_pth = line['video']
        # if video_pth != 'videos/object/0217-00.mp4':
        #     print(f"[{i+1}/{len(data)}] Skipping video: {video_pth}...")
        #     continue
        print(f"[{i+1}/{len(data)}] Extracting frames for video: {video_pth}...")
        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")

def run_minerva(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    # for i, (_, line) in enumerate(data.iterrows()):
    for i, (_, line) in enumerate(data.iloc[::-1].iterrows()):
        video_pth = line['video']
        print(f"[{i+1}/{len(data)}] Extracting frames for video: {video_pth}...")
        dataset.save_video_frames(video_pth, key=video_pth, verbose=True)

    print("[Done] Frame extraction finished.")


def run_motionbench(dataset):
    """
    MotionBench may contain multiple QA per video; dedupe by `video` id.
    """
    data = dataset.data
    print(f"[Info] Total samples: {len(data)}")
    if 'video' not in data.columns:
        raise ValueError("[MotionBench] dataset TSV must contain `video` column.")
    uniq = data.drop_duplicates('video').reset_index(drop=True)
    print(f"[Info] Unique videos to process: {len(uniq)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, (_, row) in enumerate(uniq.iterrows()):
        print(f"[{i+1}/{len(uniq)}] Extracting frames for video: {row['video']}...")
        dataset.save_video_frames(row, verbose=True)

    print("[Done] Frame extraction finished.")


def run_vue_tr(dataset):
    """
    VUE_TR / VUE_TR_V2 are query-based; dedupe by `video_id` to avoid repeated decode.
    """
    data = dataset.data
    print(f"[Info] Total samples: {len(data)}")
    if 'video_id' not in data.columns:
        raise ValueError("[VUE_TR/VUE_TR_V2] dataset TSV must contain `video_id` column.")
    uniq = data.drop_duplicates('video_id').reset_index(drop=True)
    print(f"[Info] Unique videos to process: {len(uniq)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    # for i, (_, row) in enumerate(uniq.iterrows()):
    for i, (_, row) in enumerate(uniq.iloc[::-1].iterrows()):
        print(f"[{i+1}/{len(uniq)}] Extracting frames for video_id: {row['video_id']}...")
        try:
            dataset.save_video_frames(row)
        except FileNotFoundError as e:
            # Some youtube videos are not downloadable; skip them and continue.
            print(f"[Warn] {e}")
            continue
        except Exception as e:
            print(f"[Warn] Failed to extract frames for video_id={row.get('video_id')!r}: {type(e).__name__}: {e}")
            continue
    print("[Done] Frame extraction finished.")


def run_ego4d_nlq_v2(dataset):
    """
    Ego4D-NLQ-v2: query-based temporal grounding; dedupe by `video_id`.
    """
    from vlmeval.dataset.ego4d_nlq_v2 import Ego4DNLQv2

    if not isinstance(dataset, Ego4DNLQv2):
        raise TypeError(
            f"[Ego4D-NLQ-v2] expected vlmeval.dataset.ego4d_nlq_v2.Ego4DNLQv2, "
            f"got {type(dataset).__module__}.{type(dataset).__name__}"
        )

    data = dataset.data
    print(f"[Ego4D-NLQ-v2] Total samples: {len(data)}")
    if 'video_id' not in data.columns:
        raise ValueError("[Ego4D-NLQ-v2] dataset TSV must contain `video_id` column.")
    uniq = data.drop_duplicates('video_id').reset_index(drop=True)
    print(f"[Ego4D-NLQ-v2] Unique videos to process: {len(uniq)}")
    print(f"[Ego4D-NLQ-v2] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, (_, row) in enumerate(uniq.iloc[::-1].iterrows()):
        print(f"[Ego4D-NLQ-v2] [{i+1}/{len(uniq)}] video_id={row['video_id']} ...")
        try:
            dataset.save_video_frames(row)
        except FileNotFoundError as e:
            print(f"[Ego4D-NLQ-v2][Warn] {e}")
            continue
        except Exception as e:
            print(
                f"[Ego4D-NLQ-v2][Warn] Failed video_id={row.get('video_id')!r}: "
                f"{type(e).__name__}: {e}"
            )
            continue
    print("[Ego4D-NLQ-v2] Done.")


def run_moment_seeker(dataset):
    """
    MomentSeeker: query-based temporal grounding; dedupe by `video_id`.
    Always calls `MomentSeeker.save_video_frames` (not VUE_TR).
    """
    from vlmeval.dataset.moment_seeker import MomentSeeker

    if not isinstance(dataset, MomentSeeker):
        raise TypeError(
            f"[MomentSeeker] expected vlmeval.dataset.moment_seeker.MomentSeeker, got {type(dataset).__module__}.{type(dataset).__name__}"
        )

    data = dataset.data
    print(f"[MomentSeeker] Total samples: {len(data)}")
    if 'video_id' not in data.columns:
        raise ValueError("[MomentSeeker] dataset TSV must contain `video_id` column.")
    uniq = data.drop_duplicates('video_id').reset_index(drop=True)
    print(f"[MomentSeeker] Unique videos to process: {len(uniq)}")
    print(f"[MomentSeeker] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, (_, row) in enumerate(uniq.iloc[::-1].iterrows()):
        print(f"[MomentSeeker] [{i+1}/{len(uniq)}] video_id={row['video_id']} ...")
        try:
            dataset.save_video_frames(row)
        except FileNotFoundError as e:
            print(f"[MomentSeeker][Warn] {e}")
            continue
        except Exception as e:
            print(
                f"[MomentSeeker][Warn] Failed video_id={row.get('video_id')!r}: "
                f"{type(e).__name__}: {e}"
            )
            continue
    print("[MomentSeeker] Done.")


def run_timelens(dataset):
    """
    TimeLens is span/query-based; easiest is to trigger its internal saving by calling build_prompt(video_llm=False).
    Dedupe by video id to avoid repeated decode.
    """
    data = dataset.data
    print(f"[Info] Total samples: {len(data)}")
    if 'video' not in data.columns:
        raise ValueError("[TimeLens] dataset TSV must contain `video` column.")
    uniq = data.drop_duplicates('video').reset_index(drop=True)
    print(f"[Info] Unique videos to process: {len(uniq)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, (_, row) in enumerate(uniq.iterrows()):
    # for i, (_, row) in enumerate(uniq.iloc[::-1].iterrows()):
        print(f"[{i+1}/{len(uniq)}] Extracting frames for video: {row['video']}...")
        dataset.build_prompt(row, video_llm=False)
    print("[Done] Frame extraction finished.")

def run_tvbench(dataset):
    data = dataset.data
    print(f"[Info] Total samples: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, (_, line) in enumerate(data.iterrows()):
        print(f"[{i+1}/{len(data)}] Extracting frames for video: {line['video']}...")
        dataset.save_video_frames(line, verbose=True)
    print("[Done] Frame extraction finished.")

def run_tempcompass(dataset):
    """
    TempCompass 是 ConcatVideoDataset：顶层类沿用 VideoBaseDataset.save_video_frames（按 data_root/<video>.mp4），
    与真实路径 prefix/suffix 不一致。需按 SUB_DATASET + original_index 把行映射回子数据集
    （TempCompass_MCQ / Captioning / YorN），再调用子类的 save_video_frames(line)。
    """
    data = dataset.data
    print(f"[Info] Total samples (rows): {len(data)}")
    if not hasattr(dataset, "dataset_map"):
        raise ValueError(
            "[TempCompass] Expected a ConcatVideoDataset with `dataset_map`; "
            "build dataset as `TempCompass` (not a single sub-dataset name)."
        )
    if "SUB_DATASET" not in data.columns or "original_index" not in data.columns:
        raise ValueError(
            "[TempCompass] Concatenated TSV must contain SUB_DATASET and original_index."
        )
    uniq = data.drop_duplicates(["SUB_DATASET", "video"]).reset_index(drop=True)
    print(f"[Info] Unique (sub-dataset, video) clips: {len(uniq)}")
    print(
        f"[Info] Frames are written under {os.environ['LMUData']}/images/"
        "<TempCompass_MCQ|TempCompass_Captioning|TempCompass_YorN>/..."
    )
    for i, (_, line) in enumerate(uniq.iterrows()):
        dname = line["SUB_DATASET"]
        idx = int(line["original_index"])
        sub = dataset.dataset_map[dname]
        org_line = sub.data[sub.data["index"] == idx].iloc[0]
        print(f"[{i+1}/{len(uniq)}] [{dname}] video={line['video']} ...")
        sub.save_video_frames(org_line, verbose=True)
    print("[Done] Frame extraction finished.")

def _lightweight_import_build_dataset():
    """Import build_dataset without triggering the heavy vlmeval/__init__.py.

    vlmeval/__init__.py imports ALL 100+ VLM model implementations
    (from .vlm import *), which is extremely slow and unnecessary for
    frame extraction.  We create a minimal package stub so that only
    vlmeval.smp and vlmeval.dataset are loaded.
    """
    import sys
    import types

    vlmeval_pkg_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'vlmeval')
    )

    stub = types.ModuleType('vlmeval')
    stub.__path__ = [vlmeval_pkg_dir]
    stub.__package__ = 'vlmeval'
    stub.__file__ = os.path.join(vlmeval_pkg_dir, '__init__.py')
    sys.modules['vlmeval'] = stub

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    try:
        import torch
    except ImportError:
        pass

    from vlmeval import smp
    smp.load_env()

    for name in dir(smp):
        if not name.startswith('_'):
            setattr(stub, name, getattr(smp, name))

    from vlmeval.dataset import build_dataset
    return build_dataset


def main():
    assert "LMUData" in os.environ
    save_root = os.environ["LMUData"]
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    args = parse_args()

    print(f"[Info] Building dataset: {args.dataset}")
    build_dataset = _lightweight_import_build_dataset()
    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise ValueError(f"Failed to build dataset '{args.dataset}'. Please check the name and data files.")

    if not hasattr(dataset, "save_video_frames"):
        raise ValueError(
            f"Dataset '{args.dataset}' does not provide `save_video_frames`. "
            "Make sure you are using a video dataset that supports frame extraction."
        )

    if 'VideoMMMU' in args.dataset:
        run_videommmu(dataset)
    elif 'Video-MME' in args.dataset:
        run_videomme(dataset)
    elif 'LongVideoBench' in args.dataset:
        run_longvideobench(dataset)
    elif 'LVBench' in args.dataset:
        run_lvbench(dataset)
    elif 'VideoEval-Pro' in args.dataset:
        run_videoevalpro(dataset)
    elif 'MMVU' in args.dataset:
        run_mmvu(dataset)
    elif 'TOMATO' in args.dataset:
        run_tomato(dataset)
    elif 'Minerva' in args.dataset:
        run_minerva(dataset)
    elif 'MotionBench' in args.dataset:
        run_motionbench(dataset)
    elif 'VUE_TR_V2' in args.dataset:
        run_vue_tr(dataset)
    elif 'VUE_TR' in args.dataset:
        run_vue_tr(dataset)
    elif 'Ego4D-NLQ-v2' in args.dataset:
        run_ego4d_nlq_v2(dataset)
    elif 'MomentSeeker' in args.dataset:
        run_moment_seeker(dataset)
    elif 'TimeLens' in args.dataset:
        run_timelens(dataset)
    elif 'TVBench' in args.dataset:
        run_tvbench(dataset)
    elif 'TempCompass' in args.dataset:
        run_tempcompass(dataset)
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")


if __name__ == "__main__":
    main()
