from vlmeval.dataset import *
from functools import partial

vcrbench_dataset = {
    'VCRBench_8frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=8, pack=False),
    'VCRBench_16frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=16, pack=False),
    'VCRBench_32frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=32, pack=False),
    'VCRBench_64frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=64, pack=False),
    'VCRBench_1fps_nopack': partial(VCRBench, dataset='VCR-Bench', fps=1.0, pack=False)
}

mmbench_video_dataset = {
    'MMBench_Video_8frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=False),
    'MMBench_Video_8frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=True),
    'MMBench_Video_16frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=16, pack=False),
    'MMBench_Video_64frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=False),
    'MMBench_Video_64frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=True),
    'MMBench_Video_1fps_nopack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=False),
    'MMBench_Video_1fps_pack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=True)
}

mvbench_dataset = {
    'MVBench_8frame': partial(MVBench, dataset='MVBench', nframe=8),
    'MVBench_64frame': partial(MVBench, dataset='MVBench', nframe=64),
    # MVBench not support fps, but MVBench_MP4 does
    'MVBench_MP4_8frame': partial(MVBench_MP4, dataset='MVBench_MP4', nframe=8),
    'MVBench_MP4_1fps': partial(MVBench_MP4, dataset='MVBench_MP4', fps=1.0),
}

tamperbench_dataset = {
    'MVTamperBench_8frame': partial(MVTamperBench, dataset='MVTamperBench', nframe=8),
    'MVTamperBenchStart_8frame': partial(MVTamperBench, dataset='MVTamperBenchStart', nframe=8),
    'MVTamperBenchEnd_8frame': partial(MVTamperBench, dataset='MVTamperBenchEnd', nframe=8),
}

videomme_dataset = {
    'Video-MME_2fps_limit_1024_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1024, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

videommev2_dataset = {
    'Video-MME-v2_1fps_limit_512_448px_80kctx': partial(VideoMMEv2, dataset='Video-MME-v2', fps=1.0, frames_limit=512, resize_target_area=448 * 448, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

videommmu_dataset = {
    'VideoMMMU_2fps_limit_512_768px_80kctx': partial(VideoMMMU, dataset='VideoMMMU', fps=2.0, frames_limit=512, min_pixels=28*28, max_pixels=768*768, total_pixels=80000*2*4*14*14),
}

longvideobench_dataset = {
    'LongVideoBench_2fps_limit_2048_448px_64kctx': partial(LongVideoBench, dataset='LongVideoBench', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=64000*2*4*14*14),
}

videoevalpro_dataset = {
    'VideoEval-Pro_OpenEnded_2fps_limit_2048_448px_80kctx': partial(VideoEvalPro, dataset='VideoEval-Pro', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'VideoEval-Pro_MCQ_2fps_limit_2048_448px_80kctx': partial(VideoEvalProMCQ, dataset='VideoEval-Pro', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

lvbench_dataset = {
    'LVBench_2fps_limit_2048_448px_80kctx': partial(LVBench, dataset='LVBench', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

mlvu_dataset = {
    'MLVU_8frame': partial(MLVU, dataset='MLVU', nframe=8),
    'MLVU_64frame': partial(MLVU, dataset='MLVU', nframe=64),
    'MLVU_1fps': partial(MLVU, dataset='MLVU', fps=1.0)
}

tempcompass_dataset = {
    'TempCompass_8fps_limit_2048_448px_80kctx': partial(TempCompass, dataset='TempCompass', fps=8.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

# In order to reproduce the experimental results in CGbench paper,
# use_subtitle, use_subtitle_time and use_frame_time need to be set to True.
# When measuring clue-related results, if the number of frames used is greater
# than 32, the frame capture limit will be set to 32.
# We implement the metrics long_acc, clue_acc, miou, CRR, acc@iou and rec@iou
# in the CGBench_MCQ_Grounding_Mini and CGBench_MCQ_Grounding datasets;
# the metric open-ended is implemented in the CGBench_OpenEnded_Mini and CGBench_OpenEnded datasets.
cgbench_dataset = {
    'CGBench_MCQ_Grounding_Mini_8frame_subs_subt': partial(
        CGBench_MCQ_Grounding_Mini,
        dataset='CG-Bench_MCQ_Grounding_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True
    ),
    'CGBench_OpenEnded_Mini_8frame_subs_subt_ft': partial(
        CGBench_OpenEnded_Mini,
        dataset='CG-Bench_OpenEnded_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_MCQ_Grounding_32frame_subs': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=32,
        use_subtitle=True
    ),
    'CGBench_OpenEnded_8frame': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=8
    ),
    'CGBench_MCQ_Grounding_16frame_subs_subt_ft': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_OpenEnded_16frame_subs_subt_ft': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    )
}

megabench_dataset = {
    'MEGABench_core_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="core"),
    'MEGABench_open_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="open"),
    'MEGABench_core_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="core"),
    'MEGABench_open_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="open")
}

moviechat1k_dataset = {
    'moviechat1k_breakpoint_8frame': partial(MovieChat1k, dataset='MovieChat1k', subset='breakpoint', nframe=8),
    'moviechat1k_global_14frame': partial(MovieChat1k, dataset='MovieChat1k', subset='global', nframe=14),
    'moviechat1k_global_8frame_limit0.01': partial(
        MovieChat1k, dataset='MovieChat1k', subset='global', nframe=8, limit=0.01
    )
}

vdc_dataset = {
    'VDC_8frame': partial(VDC, dataset='VDC', nframe=8),
    'VDC_1fps': partial(VDC, dataset='VDC', fps=1.0),
}

worldsense_dataset = {
    'WorldSense_8frame': partial(WorldSense, dataset='WorldSense', nframe=8),
    'WorldSense_8frame_subs': partial(WorldSense, dataset='WorldSense', nframe=8, use_subtitle=True),
    'WorldSense_8frame_audio': partial(WorldSense, dataset='WorldSense', nframe=8, use_audio=True),
    'WorldSense_32frame': partial(WorldSense, dataset='WorldSense', nframe=32),
    'WorldSense_32frame_subs': partial(WorldSense, dataset='WorldSense', nframe=32, use_subtitle=True),
    'WorldSense_32frame_audio': partial(WorldSense, dataset='WorldSense', nframe=32, use_audio=True),
    'WorldSense_1fps': partial(WorldSense, dataset='WorldSense', fps=1.0),
    'WorldSense_1fps_subs': partial(WorldSense, dataset='WorldSense', fps=1.0, use_subtitle=True),
    'WorldSense_1fps_audio': partial(WorldSense, dataset='WorldSense', fps=1.0, use_audio=True),
    'WorldSense_0.5fps': partial(WorldSense, dataset='WorldSense', fps=0.5),
    'WorldSense_0.5fps_subs': partial(WorldSense, dataset='WorldSense', fps=0.5, use_subtitle=True),
    'WorldSense_0.5fps_audio': partial(WorldSense, dataset='WorldSense', fps=0.5, use_audio=True)
}

qbench_video_dataset = {
    'QBench_Video_8frame': partial(QBench_Video, dataset='QBench_Video', nframe=8),
    'QBench_Video_16frame': partial(QBench_Video, dataset='QBench_Video', nframe=16),
}

video_mmlu_dataset = {
    'Video_MMLU_CAP_16frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=16),
    'Video_MMLU_CAP_64frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=64),
    'Video_MMLU_QA_16frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=16),
    'Video_MMLU_QA_64frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=64),
}

video_tt_dataset = {
    'Video_TT_16frame': partial(VideoTT, dataset='Video-TT', nframe=16),
    'Video_TT_32frame': partial(VideoTT, dataset='Video-TT', nframe=32),
    'Video_TT_64frame': partial(VideoTT, dataset='Video-TT', nframe=64),
}

video_holmes_dataset = {
    'Video_Holmes_32frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=32),
    'Video_Holmes_64frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=64),
}

cg_av_counting_dataset = {
    'CG-AV-Counting_32frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=32, use_frame_time=False),
    'CG-AV-Counting_64frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=64, use_frame_time=False)
}

egoexobench_dataset = {
    'EgoExoBench_64frame': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=False),  # noqa: E501
    'EgoExoBench_64frame_skip_EgoExo4D': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=True)  # noqa: E501

}

vsibench_dataset = {
    'vsibench_16frame': partial(VSIBench, dataset='VSIBench', nframe=16),
    'vsibench_32frame': partial(VSIBench, dataset='VSIBench', nframe=32),
    'vsibench_64frame': partial(VSIBench, dataset='VSIBench', nframe=64),
}

mmvu_dataset = {
    'MMVU_2fps_limit_2048_768px_80kctx': partial(MMVU, dataset='MMVU', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=768*768, total_pixels=80000*2*4*14*14),
}

tomato_dataset = {
    'TOMATO_8fps_limit_2048_448px_80kctx': partial(TOMATO, dataset='TOMATO', fps=8.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

minerva_dataset = {
    'Minerva_2fps_limit_2048_448px_80kctx': partial(Minerva, dataset='Minerva', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

timelens_dataset = {
    'TimeLens_Charades_4fps': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=4.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=128000*32*32),
    'TimeLens_ActivityNet_4fps': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=4.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=128000*32*32),
    'TimeLens_QVHighlights_4fps': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=4.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=128000*32*32),
}

tvbench_dataset = {
    'TVBench_8fps_limit_2048_448px_80kctx': partial(TVBench, dataset='TVBench', fps=8.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

motionbench_dataset = {
    'MotionBench_8fps_limit_2048_448px_80kctx': partial(MotionBench, dataset='MotionBench', fps=8.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

vue_tr_dataset = {
    'VUE_TR_1fps_limit_2048_px480_ctx128k': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
}

moment_seeker_dataset = {
    'MomentSeeker_2fps_limit_2048_px480_ctx128k': partial(MomentSeeker, dataset='MomentSeeker', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
}

ego4d_nlq_v2_dataset = {
    'Ego4D-NLQ-v2_2fps_limit_2048_px480_ctx128k': partial(Ego4DNLQv2, dataset='Ego4D-NLQ-v2', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
}

vue_tr_v2_dataset = {
    'VUE_TR_V2_1fps_limit_2048_px480_ctx128k': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
}

dream_1k_dataset = {
    'DREAM-1K_8frame': partial(DREAM, dataset='DREAM-1K', nframe=8),
    'DREAM-1K_64frame': partial(DREAM, dataset='DREAM-1K', nframe=64),
    'DREAM-1K_2fps': partial(DREAM, dataset='DREAM-1K', fps=2.0),
    'DREAM-1K_1fps': partial(DREAM, dataset='DREAM-1K', fps=1.0),
    'DREAM-1K_0.5fps': partial(DREAM, dataset='DREAM-1K', fps=0.5),
}

ovbench_dataset = {
    'OVBench_New_2fps': partial(OVBenchNew, dataset='OVBench_New_2fps', fps=2.0),
    'OVBench_New_4fps': partial(OVBenchNew, dataset='OVBench_New_4fps', fps=4.0),
    'OVBench_BBox1000_New_2fps': partial(OVBenchBBox1000New, dataset='OVBench_BBox1000_New_2fps', fps=2.0),
    'OVBench_BBox1000_New_4fps': partial(OVBenchBBox1000New, dataset='OVBench_BBox1000_New_4fps', fps=4.0),
}

odvbench_dataset = {
    'ODVBench_2fps': partial(ODVBench, dataset='ODVBench_2fps', fps=2.0),
    'ODVBench_4fps': partial(ODVBench, dataset='ODVBench_4fps', fps=4.0),
    'ODVBench_BBox1000_2fps': partial(ODVBenchBBox1000, dataset='ODVBench_BBox1000_2fps', fps=2.0),
    'ODVBench_BBox1000_4fps': partial(ODVBenchBBox1000, dataset='ODVBench_BBox1000_4fps', fps=4.0),
}


ovobench_dataset = {
    'OVOBench_2fps': partial(OVOBench, dataset='OVOBench_2fps', fps=2.0),
    'OVOBench_4fps': partial(OVOBench, dataset='OVOBench_4fps', fps=4.0),
    'OVOBench_2fps_online_max32': partial(OVOBench, dataset='OVOBench_2fps', fps=2.0, online_mode=True, max_nframe=32),
    'OVOBench_4fps_online_max32': partial(OVOBench, dataset='OVOBench_4fps', fps=4.0, online_mode=True, max_nframe=32),
}

streamingbench_dataset = {
    'StreamingBench_2fps': partial(StreamingBench, dataset='StreamingBench_2fps', fps=2.0),
    'StreamingBench_4fps': partial(StreamingBench, dataset='StreamingBench_4fps', fps=4.0),
    'StreamingBench_2fps_online_max32': partial(StreamingBench, dataset='StreamingBench_2fps', fps=2.0, online_mode=True, max_nframe=32),
    'StreamingBench_4fps_online_max32': partial(StreamingBench, dataset='StreamingBench_4fps', fps=4.0, online_mode=True, max_nframe=32),
}

river_dataset = {
    'RIVER_Bench_1fps': partial(RIVER_Bench, dataset='RIVER_Bench', fps=1.0,frames_limit=512),
    'RIVER_Bench_2fps': partial(RIVER_Bench, dataset='RIVER_Bench', fps=2.0,frames_limit=512),
    'RIVER_Bench_4fps': partial(RIVER_Bench, dataset='RIVER_Bench', fps=4.0,frames_limit=512),
    'RIVER_Bench_1fps_online_max32': partial(RIVER_Bench, dataset='RIVER_Bench', fps=1.0, online_mode=True, max_nframe=32),
    'RIVER_Bench_2fps_online_max32': partial(RIVER_Bench, dataset='RIVER_Bench', fps=2.0, online_mode=True, max_nframe=32),
    'RIVER_Bench_4fps_online_max32': partial(RIVER_Bench, dataset='RIVER_Bench', fps=4.0, online_mode=True, max_nframe=32),
}

ovo_timing_dataset = {
    'OVO_Timing_1fps_max128f': partial(OVOTiming, dataset='OVO_Timing', fps=1.0, max_num_frames=128),
    'OVO_Timing_2fps_max128f': partial(OVOTiming, dataset='OVO_Timing', fps=2.0, max_num_frames=128),
    'OVO_Timing_4fps_max128f': partial(OVOTiming, dataset='OVO_Timing', fps=4.0, max_num_frames=128),
}

proactive_videoqa_dataset = {
    k: partial(ProactiveVideoQA, dataset=k, fps=1.0)
    for k in ProactiveVideoQA.supported_datasets()
}
#   i.e.
#   ProactiveVideoQA_WEB
#   ProactiveVideoQA_VAD
#   ProactiveVideoQA_TV
#   ProactiveVideoQA_EGO

supported_video_datasets = {}

dataset_groups = [
    mmbench_video_dataset, mvbench_dataset, videomme_dataset, videommev2_dataset, videommmu_dataset, longvideobench_dataset,
    videoevalpro_dataset, lvbench_dataset,
    mlvu_dataset, tempcompass_dataset, cgbench_dataset, worldsense_dataset, tamperbench_dataset,
    megabench_dataset, qbench_video_dataset, moviechat1k_dataset, vdc_dataset, video_holmes_dataset, vcrbench_dataset,
    cg_av_counting_dataset, video_mmlu_dataset, egoexobench_dataset, dream_1k_dataset, video_tt_dataset,
    vsibench_dataset, mmvu_dataset, tomato_dataset, minerva_dataset, timelens_dataset, motionbench_dataset, vue_tr_dataset, vue_tr_v2_dataset, moment_seeker_dataset, ego4d_nlq_v2_dataset, tvbench_dataset, odvbench_dataset, proactive_videoqa_dataset, ovbench_dataset, ovobench_dataset, streamingbench_dataset, river_dataset, ovo_timing_dataset
]

for grp in dataset_groups:
    supported_video_datasets.update(grp)
