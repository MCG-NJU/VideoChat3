<p align="center">
  <img src="docs/public/parrot.png" width="112" alt="VideoChat3 logo">
</p>

<h1 align="center">VideoChat3</h1>

<p align="center">
  <strong>A Fully Open Architecture and Data Recipe for Efficient Video Instruction Training</strong>
</p>

<p align="center">
  <a href="https://mcg-nju.github.io/VideoChat3/">Project Page</a> ·
  <a href="https://huggingface.co/collections/MCG-NJU/videochat3">Models &amp; Data</a>
</p>

## Overview

VideoChat3 is an efficient, generalist video-centric multimodal large language model with 4B parameters. It is built for fine-grained motion understanding, long-form video reasoning, temporal grounding, and online streaming perception.

VideoChat3 combines an **Inflated 3D Vision Transformer (I3D-ViT)** for efficient spatiotemporal representation with **Adaptive Frame Resolution** for streaming video perception. Its scalable data pipeline curates three complementary instruction-tuning datasets - **VideoChat3-Academic2M**, **VideoChat3-LV116K**, and **VideoChat3-OL617K** - covering general, long-form, and streaming video scenarios.

The project aims to provide a reproducible foundation for efficient, real-world video understanding.

## Highlights

- **Generalist video understanding:** one model for motion, long video, temporal grounding, and live streaming.
- **Token-efficient architecture:** I3D-ViT compresses redundant visual tokens while preserving spatiotemporal evidence.
- **Adaptive streaming perception:** frame resolution is increased only when closer visual inspection is needed.
- **Open resources:** model weights and the complete training datasets are publicly available.

## TODO

- [x] Release model weights and data
- [ ] Release training code

## Project Page

The project homepage is maintained in [`docs/`](./docs) and deployed at <https://mcg-nju.github.io/VideoChat3/>.
