# VideoChat3 Training

This directory contains the VideoChat3 training implementation, including the model
components, multimodal data pipeline, staged training configurations, checkpoint
initialization utility, and distributed launch scripts.

The training stack is developed on top of
[XTuner V1](https://github.com/InternLM/xtuner). VideoChat3 extends XTuner with its
video model architecture, tokenization and processing logic, datasets, and staged
training recipes.

## Installation

### Requirements

- Linux
- Python 3.10 or later
- NVIDIA GPUs with a working CUDA environment
- PyTorch 2.6 or later

Create and activate an isolated Python environment, then install the training
project from this directory:

```bash
cd xtuner-videochat3
python -m pip install --upgrade pip
pip install -e ".[video]"
```

XTuner recommends installing
[FlashAttention](https://github.com/Dao-AILab/flash-attention) for faster training.
Install a version compatible with the PyTorch, CUDA, and GPU architecture in the
training environment. The supplied launch scripts use `XTUNER_USE_FA3=0` by
default; Hopper users with FlashAttention-3 installed may override it:

```bash
export XTUNER_USE_FA3=1
```

Verify the editable installation and the training entry point:

```bash
python -c "import torch; from xtuner.version import __version__; print(torch.__version__, __version__)"
python xtuner/v1/train/cli/sft.py --help
```

## Build an Initial VideoChat3 Checkpoint

Stage0-1 and stage1-1 require a VideoChat3-format initial checkpoint. If only an
unaligned LLM and a pretrained ViT are available,
`tools_model_inits/init_model_weights.py` can assemble them into the checkpoint
layout consumed by the training code.

The two initial stages use different vision encoders:

- **stage0-1:** initialize the vision tower from MoonViT-SO-400M;
- **stage1-1:** initialize the vision tower from
  [MCG-NJU/I3D-ViT](https://huggingface.co/MCG-NJU/I3D-ViT), rather than
  MoonViT.

The utility:

1. loads the ViT weights into `model.vision_tower.*`;
2. maps the LLM weights into `model.language_model.*`;
3. builds the full model from a VideoChat3 `config.json`;
4. leaves newly introduced components, including the multimodal projector,
   randomly initialized for subsequent vision-language alignment;
5. runs a text-only forward check and saves the result with
   `save_pretrained()`.

### Prepare the inputs

The LLM and ViT directories must contain either a single `model.safetensors` or
sharded safetensors accompanied by `model.safetensors.index.json`. Prepare:

- an unaligned LLM checkpoint, such as a compatible Qwen3-4B checkpoint;
- a pretrained vision checkpoint: MoonViT-SO-400M for stage0-1, or
  [MCG-NJU/I3D-ViT](https://huggingface.co/MCG-NJU/I3D-ViT) for stage1-1;
- a VideoChat3 template directory containing at least `config.json`.

The template configuration must match the selected LLM and ViT, including hidden
sizes, layer counts, vocabulary size, vision architecture, and special-token IDs.

The utility currently uses paths defined in `main()` rather than command-line
arguments. Edit the following values in
`tools_model_inits/init_model_weights.py`:

```python
# stage0-1:
vit_path = "models/MoonViT-SO-400M"
qwen3_path = "models/Qwen3-4B-Instruct-2507"
current_dir = "path/to/VideoChat3-4B-template"
output_path = "VideoChat3-4B-stage0-init"

# stage1-1 uses I3D-ViT instead of MoonViT:
# vit_path = "models/I3D-ViT"
# output_path = "VideoChat3-4B-stage1-init"
```

Run the utility from this directory:

```bash
python tools_model_inits/init_model_weights.py
```

The utility builds the model and loads both source checkpoints in host memory.
Ensure that the machine has sufficient RAM. Missing keys belonging to newly
initialized multimodal modules are expected; unexpected keys are treated as an
architecture or checkpoint mismatch.

Set `model_path` in the corresponding initial-stage configuration to the
generated directory:

```python
# training_configs/stage0/VideoChat3_4B_train_stage0-1.py
model_path = "VideoChat3-4B-stage0-init"

# training_configs/stage1/VideoChat3_4B_train_stage1-1.py
model_path = "VideoChat3-4B-stage1-init"
```

This `model_path` is passed to the model loader through `TrainerConfig` and to
the tokenizer and media processor through `tokenizer_path` and
`VideoChat3TokenizeFnConfig.processor_path`.

Because stage0-1 uses MoonViT while stage1-1 uses I3D-ViT, their initialized
checkpoints should normally be generated and stored separately.

## Prepare Training Configurations and Data

Training recipes are under `training_configs/`. Every recipe has a matching
launcher:

| Stage | Configuration | Launcher |
| --- | --- | --- |
| stage0-1 | `training_configs/stage0/VideoChat3_4B_train_stage0-1.py` | `training_scripts/stage0/VideoChat3_4B_train_stage0-1.sh` |
| stage0-2 | `training_configs/stage0/VideoChat3_4B_train_stage0-2.py` | `training_scripts/stage0/VideoChat3_4B_train_stage0-2.sh` |
| stage1-1 | `training_configs/stage1/VideoChat3_4B_train_stage1-1.py` | `training_scripts/stage1/VideoChat3_4B_train_stage1-1.sh` |
| stage1-2 | `training_configs/stage1/VideoChat3_4B_train_stage1-2.py` | `training_scripts/stage1/VideoChat3_4B_train_stage1-2.sh` |
| stage2 | `training_configs/stage2/VideoChat3_4B_train_stage2.py` | `training_scripts/stage2/VideoChat3_4B_train_stage2.sh` |
| stage3 | `training_configs/stage3/VideoChat3_4B_train_stage3.py` | `training_scripts/stage3/VideoChat3_4B_train_stage3.sh` |

Before launching a stage, review these fields in its configuration:

- `model_path`: the initial model or the exported Hugging Face checkpoint from
  the preceding stage;
- `meta_data_path`: the dataset collection JSON;
- `work_dir`: the checkpoint and training-state output directory;
- each dataset's `anno_path`, `media_root`, sampling ratio, and media-processing
  limits.

Paths containing `xxxx` are placeholders and must be replaced with real
checkpoint directories. Model weights, annotations, and media data referenced by
the recipes are not downloaded or rewritten by the launch scripts.

## Launch Training

All launchers call `training_scripts/run_sft.sh`. The defaults are eight nodes
and eight processes/GPUs per node:

```text
NNODES=8
NPROC_PER_NODE=8
```

Stage3 is intended to use the eight-node default. Stage0, stage1, and stage2 can
be attempted with `NNODES=4` or fewer nodes when memory and global-batch
requirements permit.

### Slurm

Allocate one launcher task per node, then use `srun` to execute the stage script
on every selected node. For example, launch stage3 on eight nodes:

```bash
cd xtuner-videochat3
salloc -p <partition> -N 8 -n 8 \
  --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:8

export NNODES=8
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 \
  bash training_scripts/stage3/VideoChat3_4B_train_stage3.sh
```

For a four-node stage2 run:

```bash
export NNODES=4
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  bash training_scripts/stage2/VideoChat3_4B_train_stage2.sh
```

The launcher obtains `MASTER_ADDR` from `SLURM_JOB_NODELIST`; it does not assume
that an IP address is encoded in the node name. `NNODES` must equal the number of
nodes on which `srun` starts the script.

### Without Slurm

For a single-node run:

```bash
cd xtuner-videochat3
NNODES=1 NPROC_PER_NODE=8 \
  bash training_scripts/stage1/VideoChat3_4B_train_stage1-1.sh
```

For a multi-node run, execute the same command on every node. All nodes must use
the same `NNODES`, `MASTER_ADDR`, `MASTER_PORT`, and `RDZV_ID`. `MASTER_ADDR`
must be a hostname or IP address of the rendezvous node that is reachable from
the other nodes:

```bash
NNODES=4 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=40000 RDZV_ID=videochat3-stage2 \
  bash training_scripts/stage2/VideoChat3_4B_train_stage2.sh
```

No manual `NODE_RANK` is required: the c10d rendezvous assigns ranks. Ensure that
the rendezvous port is reachable and that code, data paths, and software
environments are consistent across nodes.

### Launcher environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `NNODES` | `8` | Number of participating nodes |
| `NPROC_PER_NODE` | `8` | Number of processes/GPUs per node |
| `MASTER_ADDR` | First Slurm node; `127.0.0.1` for non-Slurm single-node runs | Rendezvous host |
| `MASTER_PORT` | `40000` | Rendezvous port |
| `RDZV_ID` | Slurm job ID or task name | Unique rendezvous ID for the job |
| `LOG_DIR` | `work_dir/logs/<task-name>` | Per-node log directory |
| `XTUNER_USE_FA3` | `0` | Enable FlashAttention-3 when set to `1` |

Each node writes a separate log file to prevent multiple `tee` processes from
overwriting one another on a shared filesystem.

## Direct Training Entry Point

The scripts ultimately invoke the XTuner V1 SFT entry point. A single-node debug
run can be started directly with:

```bash
torchrun --standalone --nproc-per-node=8 \
  xtuner/v1/train/cli/sft.py \
  --config training_configs/stage1/VideoChat3_4B_train_stage1-1.py
```

Use the launch scripts for production multi-node jobs so that rendezvous
settings, `PYTHONPATH`, XTuner environment variables, and logging remain
consistent. To inspect all SFT command-line options:

```bash
python xtuner/v1/train/cli/sft.py --help
```

## Acknowledgements

This training implementation is built on
[XTuner V1](https://github.com/InternLM/xtuner). We thank the XTuner contributors
for the distributed training engine, FSDP infrastructure, configuration system,
and multimodal training foundations on which this project is developed.

Please cite the VideoChat3 paper from the
[project README](../README.md) when using this training code.

## License

This training code is released under the [Apache License 2.0](LICENSE). Users
must also comply with the licenses of the selected LLM, vision encoder, datasets,
and upstream dependencies.
