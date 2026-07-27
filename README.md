<p align="center">
  <img src="docs/public/parrot.png" width="112" alt="VideoChat3 logo">
</p>

<h1 align="center">VideoChat3</h1>

<p align="center">
  <strong>A Fully Open Video MLLM for Efficient, Generalist Video Understanding</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.14935">
    <img src="https://img.shields.io/badge/arXiv-2607.14935-b31b1b.svg" align="center" alt="arXiv"> <strong>Paper</strong>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://mcg-nju.github.io/VideoChat3/">
    <img src="docs/public/globe.svg" width="20" align="center" alt="Homepage"> <strong>Homepage</strong>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/collections/MCG-NJU/videochat3">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="24" align="center" alt="Hugging Face"> <strong>Models &amp; Data</strong>
  </a>
</p>

<p align="center">
  <img src="docs/public/paper/overview.png" width="100%" alt="VideoChat3 overview across temporal perception, long-video reasoning, temporal grounding, and online proactive response">
</p>

<p align="center">
  <em>One model for fine-grained motion, long-video reasoning, temporal grounding, and online proactive response.</em>
</p>

## ✨ Overview

**VideoChat3** is a 4B generalist video MLLM built to understand video across time—from subtle motion and hour-long stories to precise temporal evidence and live streams.

It combines **I3D-ViT** for 16× spatiotemporal compression with **Adaptive Frame Resolution** for evidence-aware streaming, trained on **Academic2M**, **LV116K**, and **OL617K**.

## :fire: Updates
- [x] **2026/07/27**: 🔥🔥🔥Release the full-stage training data and training checkpoints for each stage of VideoChat3.
- [x] **2026/07/27**: 🔥🔥🔥Release the training code for VideoChat3.
- [x] **2026/07/24**: 🔥🔥🔥Release the evaluation code for VideoChat3.
- [x] **2026/07/17**: 🔥🔥🔥Release the VideoChat3 model weights and training data.


## 🚀 Highlights

- 🎬 **Generalist video understanding:** one model for motion, long video, temporal grounding, and online proactive response.
- ⚡ **Token-efficient architecture:** I3D-ViT compresses redundant visual tokens while preserving spatiotemporal evidence.
- 🔍 **Adaptive streaming perception:** frame resolution is increased only when closer visual inspection is needed.
- 🔓 **Open resources:** model weights and the complete training datasets are publicly available.

## 🛠️ Training

The training implementation is available in
[`xtuner-videochat3`](xtuner-videochat3). It is built on [XTuner V1](https://github.com/internLM/xtuner) and includes
the VideoChat3 model and data pipeline, staged training configurations, initial
checkpoint construction, and Slurm/non-Slurm distributed launchers.

Install the training project:

```bash
cd xtuner-videochat3
pip install -e ".[video]"
```

After preparing the model and dataset paths in the selected configuration, start
a launcher from the same directory. For example, a single-node stage1-1 run is:

```bash
NNODES=1 NPROC_PER_NODE=8 \
  bash training_scripts/stage1/VideoChat3_4B_train_stage1-1.sh
```

See the [complete training guide](xtuner-videochat3/README.md) for initial
checkpoint construction, all stage configurations, Slurm launch commands, and
manual multi-node setup.

## 🏆 Evaluation

The evaluation implementation is available in [`Eval`](Eval). It is built on
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and supports all
benchmarks for both offline and online video understanding evaluated in the
paper.

Please refer to the [evaluation guide](Eval/README.md) for detailed instructions
on environment setup, model downloading, and evaluation.

## 🌟 Quick Start

Please follow the instructions in [VideoChat3-4B](https://huggingface.co/MCG-NJU/VideoChat3-4B) to set up the environment and download the model weights. We provide two codebases here—one for standard model inference and the other for online proactive-response inference—to help you get started quickly.

## Citation

```
@misc{videochat3,
      title={VideoChat3: Fully Open Video MLLM for Efficient and Generalist Video Understanding}, 
      author={Xinhao Li and Yuhan Zhu and Xiangyu Zeng and Yuhao Dong and Haoning Wu and Zhiqiu Zhang and Yuandong Yang and Changlian Ma and Qingyu Zhang and Yansong Shi and Xinyu Chen and Haoran Chen and Zizheng Huang and Jun Zhang and Kun Ouyang and Lin Sui and Ziang Yan and Yicheng Xu and Chenting Wang and Yinan He and Hongjie Zhang and Yi Wang and Yu Qiao and Yali Wang and Ziwei Liu and Kai Chen and Limin Wang},
      year={2026},
      eprint={2607.14935},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.14935}, 
}
```
