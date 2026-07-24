import os

from .vidi_vue_tr import VUE_TR, _VUE_TR_Paths, _ensure_vue_tr_tsv


def _default_vue_tr_v2_root() -> str:
    return os.environ.get(
        'VUE_TR_V2_ROOT',
        '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/vidi/VUE_TR_V2',
    )


class VUE_TR_V2(VUE_TR):
    """
    VUE-TR v2: same temporal-retrieval protocol as VUE_TR, different annotation release.
    Ground truth: `VUE-TRv2_ground_truth.json` under `VUE_TR_V2_ROOT` (default path above).
    Videos: `<VUE_TR_V2_ROOT>/videos/`. Missing mp4 files are skipped when building the TSV.
    Evaluation metrics match `VUE_TR_V2/qa_eval.py` (same as V1 implementation in `vidi_vue_tr.py`).
    Only `query_modality == "vision"` samples are kept (same as VUE_TR).
    """

    def __init__(
        self,
        dataset='VUE_TR_V2',
        nframe=0,
        fps=-1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
        reason=False,
    ):
        super().__init__(
            dataset=dataset,
            nframe=nframe,
            fps=fps,
            frames_limit=frames_limit,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            check_extracted_frames=check_extracted_frames,
            reason=reason,
        )

    @classmethod
    def supported_datasets(cls):
        return ['VUE_TR_V2']

    def prepare_dataset(self, dataset):
        root = _default_vue_tr_v2_root()
        videos_dir = os.path.join(root, 'videos')
        paths = _VUE_TR_Paths(
            root=root,
            videos_dir=videos_dir,
            gt_json=os.path.join(root, 'VUE-TRv2_ground_truth.json'),
            tsv_path=os.path.join(root, 'VUE_TR_V2.tsv'),
        )
        _ensure_vue_tr_tsv(paths)
        self._videos_dir = paths.videos_dir
        return dict(root=paths.videos_dir, data_file=paths.tsv_path)
