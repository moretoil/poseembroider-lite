# PoseEmbroider-Lite

**English** | [简体中文](README.zh-CN.md)

**A torch-free, ~300 MB deployment of [PoseEmbroider](https://github.com/naver/poseembroider) (ECCV 2024) for image → SMPL-X human mesh recovery.**

Takes a photo, predicts SMPL-X pose & shape, outputs a textured-less mesh (`OBJ`) you can open in Blender — on CPU, in ~0.2 s per image, without installing PyTorch/CUDA.

<p align="center">
  <img src="docs/images/input.jpg" width="22%" alt="input">
  <img src="docs/images/front.png" width="22%" alt="front view">
  <img src="docs/images/side.png" width="22%" alt="side view">
  <img src="docs/images/top.png" width="22%" alt="top view">
</p>
<p align="center"><i>Input → Front / Side / Top views of the predicted SMPL-X mesh</i></p>

## Highlights

| | Original PoseEmbroider | **This repo** |
|---|---|---|
| Runtime | PyTorch + CUDA (~5–7 GB) | **NumPy + ONNX Runtime (~90 MB)** |
| Weights | ~1.2 GB of `.pth` | **179 MB fp16 ONNX + 28 MB SMPL-X** |
| Total footprint | ~8.5 GB | **~300 MB** |
| Speed (CPU, per image) | — | **~170–340 ms** |
| Accuracy vs original pipeline | — | **0.12° mean joint rotation error** |

- **Two-stage slimming**
  1. Skip the text/pose encoder checkpoints (unused for image-only inference) → −585 MB, outputs bit-identical.
  2. Export the whole image→SMPL-X-params pipeline to ONNX, fp16 quantization, and a hand-written NumPy SMPL-X LBS → no PyTorch at all.
- **Web UI included** — drag & drop an image, get tri-view renders (or an OBJ if rendering libs are absent).
- **Validated numerically** against the original torch pipeline (see [Accuracy](#accuracy)).

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```
Tested with Python 3.8–3.12. On Python 3.8 only `onnxruntime<=1.19` is available — use the **fp32** model (see note below).

### 2. Get the models

Models are hosted on Hugging Face: **[foxdream/poseembroider-lite](https://huggingface.co/foxdream/poseembroider-lite/tree/main)**

```bash
# recommended (fp16, 179 MB)
huggingface-cli download foxdream/poseembroider-lite image_to_params_fp16.onnx --local-dir models

# or fp32 (357 MB, needed for Python 3.8 / onnxruntime < 1.31)
huggingface-cli download foxdream/poseembroider-lite image_to_params.onnx --local-dir models

# SMPL-X body model (already reduced/fp32, 28 MB)
huggingface-cli download foxdream/poseembroider-lite smplx_neutral_lite_f32.npz --local-dir models
```

> **Note:** the **fp16** model requires `onnxruntime >= 1.31`. With older runtimes (e.g. Python 3.8) it silently produces `NaN` — use the **fp32** model there.

### 3. Run

```bash
# CLI: image -> mesh + parameters
python runtime/infer.py --image photo.jpg --out_dir out

# Web UI (http://127.0.0.1:6007)
python webui.py
```

`runtime/infer.py` writes:
- `out/smplx_mesh.obj` — SMPL-X mesh (10 475 verts), import it in Blender (`Up Axis = Y`)
- `out/smplx_params.npz` — `rotvec` (22×3), `betas` (10), `verts` (10475×3)

Optional tri-view rendering (`--render`) needs `pyrender trimesh PyOpenGL`.

## Accuracy

Measured against the original PoseEmbroider torch pipeline on the same image:

| Component | Joint rotation error | Betas error | Vertex error |
|---|---|---|---|
| fp32 ONNX | **0.000°** (bit-identical) | 0.0000 | 0 mm |
| fp16 ONNX | 0.122° (max 0.60°) | 0.0030 | — |
| NumPy SMPL-X LBS | — | — | **3e-7** (float32 limit) |

Numbers reproduce with `validate/compare_final.py`.

## Speed

RTX 2080 Ti server, CPU-only onnxruntime (single image):

| Variant | ONNX | NumPy LBS | End-to-end |
|---|---|---|---|
| fp32 | 211 ms | 30 ms | ~241 ms |
| fp16 | 311 ms | 30 ms | ~341 ms |

On a typical desktop CPU: ~157 ms (fp32) + 5 ms LBS. Set `OMP_NUM_THREADS=4` when serving the web UI — unrestricted BLAS threads made the LBS 60× slower under Flask.

## Repository layout

```
.
├── runtime/
│   ├── infer.py            # standalone inference (ONNX + NumPy LBS + OBJ export)
│   └── numpy_lbs_fast.py   # SMPL-X linear blend skinning, pure NumPy
├── webui.py                # Flask UI: upload -> mesh / tri-view renders
├── models/                 # (downloaded, gitignored) ONNX + SMPL-X lite package
├── export/                 # one-off scripts used to build the ONNX models
│   ├── lite_encoders.py        # stage 1: skip unused text/pose encoders
│   ├── export_params_onnx.py   # stage 2: torch -> ONNX export
│   ├── export_fp16.py          # fp16 conversion (onnxruntime transformer tool)
│   ├── pack_smplx.py           # SMPL-X npz slimming: 104 MB -> 28 MB
│   └── dump_features.py        # dump torch reference outputs for comparison
├── validate/               # numerical alignment & benchmarking
│   ├── compare_final.py        # lite vs original torch pipeline
│   ├── validate_lbs.py         # NumPy LBS vs smplx (torch) LBS
│   └── final_bench.py          # latency benchmark
└── docs/images/            # README figures
```

## How it works

```
image (192x256, /255)
  -> ONNX: SMPLer-X ViT-B -> image projection -> L2 norm
           -> + modality encoding -> intermodality token
           -> embroider transformer core
           -> pose head (22x3x3 rotmats) + shape head (10 betas)
  -> rotmat -> rotvec (NumPy)  [global orient fixed to Y-up: Rx(-90°)]
  -> NumPy SMPL-X LBS (55 joints, flat hand mean) -> 10 475 vertices
```

## Re-exporting the ONNX models

Only needed if you want to rebuild `image_to_params*.onnx` from the original checkpoints. This requires the original PoseEmbroider setup (PyTorch, checkpoints, SMPL-X, SMPLer-X backbone — see the upstream repo). Then run, in order:

```bash
python export/export_params_onnx.py   # torch -> fp32 ONNX (needs original checkpoints)
python export/export_fp16.py          # fp32 -> fp16
python export/pack_smplx.py           # slim the SMPL-X npz
```

## Limitations

This is a **semantic pose representation** model repurposed for mesh recovery, not a specialized HMR network. It is noticeably weaker than dedicated methods on:

- **fine leg/hip articulation** (e.g. crossed legs, precise lumbar bending),
- rare poses (training data is 50 k farthest-point-sampled BEDLAM poses),
- heavy self-occlusion.

Reasons (see the paper's limitations): the image features are **average-pooled into a single 512-d token**, trained only with a **ranking (contrastive) objective** — no reconstruction loss — and the regression head is trained on **frozen** features. For accurate full-body capture prefer SMPLer-X / HMR2.0 / OSX / Multi-HMR.

## Credits & license

- Original method and checkpoints: [PoseEmbroider](https://github.com/naver/poseembroider) (Delmas et al., ECCV 2024), [arXiv:2409.06535](https://arxiv.org/abs/2409.06535).
- Image encoder: [SMPLer-X](https://github.com/caizhongang/SMPLer-X) (ViT-B).
- Body model: [SMPL-X](https://smpl-x.is.tue.mpg.de/).

This work is a **derivative of PoseEmbroider** and is distributed under the same **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** license — **non-commercial use only**. The SMPL-X body model is subject to its own license from the Max Planck Institute; you must obtain it yourself.

## Citation

```bibtex
@inproceedings{delmas2024poseembroider,
  title={PoseEmbroider: Towards a 3D, Visual, Semantic-aware Human Pose Representation},
  author={Delmas, Ginger and Weinzaepfel, Philippe and Moreno-Noguer, Francesc and Rogez, Gr{\'e}gory},
  booktitle={ECCV},
  year={2024}
}
```
