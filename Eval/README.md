<a id="readme-top"></a>

<div align="center">

# VideoChat3 Evaluation Toolkit

**Reproducible image and video evaluation powered by VLMEvalKit.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![Model](https://img.shields.io/badge/🤗%20Model-VideoChat3--4B-ffd21e?style=flat-square)](https://huggingface.co/MCG-NJU/VideoChat3-4B)
[![Upstream](https://img.shields.io/badge/Based%20on-VLMEvalKit-5b67d6?style=flat-square)](https://github.com/open-compass/VLMEvalKit)

[Quick start](#quick-start) · [Evaluation](#evaluation) · [Benchmarks](#benchmarks) ·
[Results](#results)

</div>

This repository provides the evaluation code for
[VideoChat3-4B](https://huggingface.co/MCG-NJU/VideoChat3-4B). It is built on
[OpenCompass VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and keeps its
one-command workflow for image, video, and long-video benchmarks.

## Highlights

- First-class `VideoChat3-4B` model registration with automatic Hugging Face loading.
- Image and video evaluation through the same runner and output format.
- Configurable frame sampling, resolution limits, and reusable frame caches.
- Single-GPU, automatic multi-GPU placement, and `torchrun` data parallelism.
- Exact-match and LLM-judge evaluation inherited from VLMEvalKit.
- Portable commands with no private cluster setup.

## Model

| CLI name | Hugging Face checkpoint | Backend |
| --- | --- | --- |
| `VideoChat3-4B` | [`MCG-NJU/VideoChat3-4B`](https://huggingface.co/MCG-NJU/VideoChat3-4B) | Transformers |

The checkpoint is downloaded from Hugging Face on first use and cached by
`huggingface_hub`.

## Quick start

### 1. Install

Recommended environment:

- Linux
- Python 3.10+
- A CUDA-capable GPU with a compatible PyTorch installation
- FFmpeg/`ffprobe` for video metadata and decoding

```bash
git clone https://github.com/zyuhan1999/VLMEvalKit.git
cd VLMEvalKit

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install flash-attn --no-build-isolation
```

VideoChat3 uses FlashAttention 2. Match the PyTorch, FlashAttention, CUDA, and
GPU-driver versions in your environment.

### 2. Configure Data

Set the environment variables required by each dataset to specify its data root and annotation-file paths. Refer to the corresponding implementation under `vlmeval/dataset` for the exact variables and expected directory layout.

`LMUData` is VLMEvalKit’s shared directory for cached data and intermediate files:

```bash
export LMUData=/absolute/path/to/LMUData
```

You may place environment variables in a local `.env` file instead. `.env` files
are ignored by Git; never commit API keys or storage credentials.

### 3. Evaluate VideoChat3

Video benchmark:

```bash
python run.py \
  --model VideoChat3-4B \
  --data Video-MME_2fps_limit_1024_448px_80kctx \
  --mode all \
  --work-dir outputs
```

Image benchmark:

```bash
python run.py \
  --model VideoChat3-4B \
  --data MMBench_DEV_EN_V11 \
  --mode all \
  --work-dir outputs
```

For proactive-response benchmarks, use `VideoChat3OVOTiming` to evaluate OVO-Timing and `VideoChat3ProactiveVQA` to evaluate ProactiveVideoQA (see `vlmeval/config.py` for details).

The model checkpoint and supported benchmark metadata are prepared automatically
when the corresponding provider permits it. Some benchmarks require accepting
their license or downloading media manually; follow the upstream
[quick-start guide](docs/en/Quickstart.md) when prompted.

## Evaluation

### Helper script

The portable helper accepts one or more registered datasets:

```bash
bash scripts/eval_videochat3/eval_4b.sh \
  Video-MME_2fps_limit_1024_448px_80kctx \
  LongVideoBench_2fps_limit_2048_448px_64kctx
```

It requires `LMUData`. Set `MODE=infer`, `MODE=eval`, or `MODE=all` to control the
stage:

```bash
MODE=infer bash scripts/eval_videochat3/eval_4b.sh Video-MME_2fps_limit_1024_448px_80kctx
MODE=eval  bash scripts/eval_videochat3/eval_4b.sh Video-MME_2fps_limit_1024_448px_80kctx
```

### Useful runner options

| Option | Meaning |
| --- | --- |
| `--mode all` | Run inference followed by evaluation. |
| `--mode infer` | Generate predictions only. |
| `--mode eval` | Evaluate existing predictions only. |
| `--reuse` | Reuse the latest compatible prediction/intermediate files. |
| `--work-dir PATH` | Write outputs under a custom directory. |
| `--retry N` | Set API-model and judge retry count; the default is `5`. |
| `--api-nproc N` | Set API request concurrency. |
| `--check-extracted-frames false` | Trust an existing frame cache without checking it. |

For long responses, avoid Excel cell-length limits:

```bash
export PRED_FORMAT=tsv
```

For benchmarks requiring an API judge, configure provider credentials through
environment variables or `.env`, then pass the appropriate `--judge` value. Keep
the default retry limit finite to avoid stalled jobs and unexpected API cost.

### Multiple GPUs

A single process can use all visible GPUs through automatic device placement:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --model VideoChat3-4B \
  --data Video-MME_2fps_limit_1024_448px_80kctx \
  --mode all
```

For data-parallel evaluation, launch multiple processes:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 run.py \
  --model VideoChat3-4B \
  --data Video-MME_2fps_limit_1024_448px_80kctx \
  --mode all
```

Each process receives a share of the visible GPUs. Choose the process count so
that one model instance fits in its assigned memory.

### Pre-extract video frames

Frame extraction can be done before a large evaluation:

```bash
bash scripts/pre_extract_video_frames/extract_video_frames.sh \
  --dataset Video-MME_2fps_limit_1024_448px_80kctx
```

The evaluation runner checks and fills missing cached frames by default. Use
`--check-extracted-frames false` only when the cache is already complete.

### Direct inference

```python
from vlmeval.config import supported_VLM

model = supported_VLM["VideoChat3-4B"]()
answer = model.generate([
    "assets/apple.jpg",
    "What is shown in this image?",
])
print(answer)
```

## Benchmarks

The repository retains VLMEvalKit's broad image/video registry and adds the
VideoChat3 adapter required to run it. Common entries include:

- Image: `MMBench_DEV_EN_V11`, `MMMU_DEV_VAL`, `MMStar`, `OCRBench`
- Video: `Video-MME_*`, `LongVideoBench_*`, `LVBench_*`, `VideoMMMU_*`
- Long-video and motion: `TOMATO_*`, `MMVU_*`, `MotionBench_*`, `TVBench_*`

Sampling rate, frame limit, and resolution variants are encoded in the registered
dataset name. See
[`vlmeval/dataset/video_dataset_config.py`](vlmeval/dataset/video_dataset_config.py)
for video datasets and [`vlmeval/config.py`](vlmeval/config.py) for the full model
registry.

## Results

Results are written below:

```text
<work-dir>/<model>/<run-id>/
```

Prediction files contain per-sample model outputs; score and rating files contain
benchmark metrics. Build a compact result table with:

```bash
python print_result.py outputs/VideoChat3-4B
```

The command recursively finds supported result files and writes
`outputs/VideoChat3-4B/summary.csv`.

## Repository layout

```text
run.py                                  Main evaluation entry point
vlmeval/config.py                       Model registry
vlmeval/vlm/videochat3/                 VideoChat3 adapter and prompts
vlmeval/dataset/video_dataset_config.py Video benchmark registry
scripts/eval_videochat3/                Portable evaluation recipe
scripts/pre_extract_video_frames/       Frame-cache preparation
```

For custom models and datasets, refer to the
[VLMEvalKit development guide](docs/en/Development.md).

## Troubleshooting

- **CUDA out of memory:** use fewer frames, a lower-resolution dataset variant, or
  expose more GPUs.
- **FlashAttention import error:** install a build compatible with the active
  PyTorch and CUDA versions.
- **Missing benchmark media:** verify `LMUData` and follow the dataset provider's
  download instructions.
- **Truncated long answers:** set `PRED_FORMAT=tsv` before inference.
- **Interrupted run:** repeat the command with `--reuse`.

## Acknowledgements

This project builds on
[OpenCompass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We thank its
authors, benchmark maintainers, and the open-source model community.

## License

The code in this repository is released under the [Apache License 2.0](LICENSE).
Model weights and datasets may use separate licenses; review their respective terms
before use.

<div align="right"><a href="#readme-top">Back to top ↑</a></div>
