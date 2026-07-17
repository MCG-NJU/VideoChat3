<p align="center">
  <img src="docs/public/parrot.png" width="112" alt="VideoChat3 logo">
</p>

<h1 align="center">VideoChat3</h1>

<p align="center">
  <strong>A Fully Open Architecture and Data Recipe for Efficient Video Instruction Training</strong>
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
## ✨ Overview

VideoChat3 is an efficient, generalist video-centric multimodal large language model with 4B parameters. It is built for fine-grained motion understanding, long-form video reasoning, temporal grounding, and online streaming perception.

VideoChat3 combines an **Inflated 3D Vision Transformer (I3D-ViT)** for efficient spatiotemporal representation with **Adaptive Frame Resolution** for streaming video perception. Its scalable data pipeline curates three complementary instruction-tuning datasets - **VideoChat3-Academic2M**, **VideoChat3-LV116K**, and **VideoChat3-OL617K** - covering general, long-form, and streaming video scenarios.

The project aims to provide a reproducible foundation for efficient, real-world video understanding.

## 🚀 Highlights

- 🎬 **Generalist video understanding:** one model for motion, long video, temporal grounding, and live streaming.
- ⚡ **Token-efficient architecture:** I3D-ViT compresses redundant visual tokens while preserving spatiotemporal evidence.
- 🔍 **Adaptive streaming perception:** frame resolution is increased only when closer visual inspection is needed.
- 🔓 **Open resources:** model weights and the complete training datasets are publicly available.

## 📋 TODO

- [x] 🤗 Release model weights and data
- [ ] 🛠️ Release training code

## 🌐 Project Page

The project homepage is maintained in [`docs/`](./docs) and deployed at <https://mcg-nju.github.io/VideoChat3/>.

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


