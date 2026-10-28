# AGENTS.md — project guide for AI coding agents

This repository is **poseembroider-lite**: a torch-free ONNX deployment of PoseEmbroider (ECCV 2024) for single-image SMPL-X mesh recovery. Read this before editing anything.

## What the code does (data flow)

```
input image
  -> cv2/PIL resize to (192,256), RGB, /255  [runtime/infer.py: preprocess]
  -> onnxruntime session: image_to_params*.onnx
       [SMPLer-X ViT-B] -> [image projection 768->512] -> [L2 norm]
       -> [+ modality encoding] -> [intermodality token]
       -> [embroider transformer core]
       -> [pose head: 22x3x3 rotmats] + [shape head: 10 betas]
  -> global orientation fix: rotmat[:,0] @= Rx(-90)   (see Conventions)
  -> rotmat -> rotvec (NumPy)
  -> numpy_lbs_fast.SMPLXNumpy.forward(betas, rotvec)
       55-joint SMPL-X linear blend skinning -> verts (10475,3) + joints (55,3)
  -> export OBJ / render tri-views
```

## File map

| Path | Role |
|---|---|
| `runtime/infer.py` | **Entry point.** `LitePoseEstimator` loads the ONNX session + SMPL-X lite npz, does preprocessing, inference, the Rx(-90) fix, LBS, OBJ export, optional rendering. |
| `runtime/numpy_lbs_fast.py` | NumPy translation of `smplx/lbs.py` (only the 22-joint subset is predicted; hands/jaw/eyes stay at the flat mean pose). Alignment with torch is ~3e-7. |
| `webui.py` | Flask UI (port 6007). Upload → predict → save `webui_out/smplx_mesh.obj` + show tri-views if `pyrender`/`trimesh` are importable, otherwise show the OBJ path. |
| `export/*.py` | One-off build scripts (need the **original** torch setup + checkpoints). Not used at inference time. |
| `validate/*.py` | Numerical alignment & benchmark scripts. |
| `models/` | Downloaded artefacts (gitignored): `image_to_params_fp16.onnx`, `image_to_params.onnx`, `smplx_neutral_lite_f32.npz`. Hosted at HuggingFace `foxdream/poseembroider-lite`. |

## Environment constraints (IMPORTANT)

- **fp16 ONNX requires `onnxruntime >= 1.31`.** On older runtimes (Python 3.8 caps at ORT 1.19) the fp16 model silently outputs `NaN`. Use `image_to_params.onnx` (fp32) there. `runtime/infer.py` defaults to the fp32 file for this reason.
- Python 3.8–3.12 supported. Core deps: `numpy`, `onnxruntime`, `pillow`, `flask`. Rendering is optional (`pyrender`, `trimesh`, `PyOpenGL` + system EGL/OSMesa).
- Serving the web UI: set `OMP_NUM_THREADS=4` before launching. Without it, BLAS thread contention inside Flask made the NumPy LBS ~60× slower (5 ms → 300 ms).
- On headless Linux, `PYOPENGL_PLATFORM=osmesa` (with `libosmesa6` installed) is the fallback when EGL is unavailable; the renderer must be created **after** setting it.

## Conventions that will bite you

1. **Global orientation is BEDLAM camera frame (Y-down).** `runtime/infer.py` left-multiplies the root rotation by `Rx(-90°)`:
   ```python
   R_fix = np.array([[1,0,0],[0,0,1],[0,-1,0]], np.float32)  # Rx(-90)
   rotmat[:, 0] = R_fix @ rotmat[:, 0]
   ```
   Removing this makes the mesh lie flat (front view looks like a top view). Do not "fix" it by changing the camera in the renderer.
2. **`flat_hand_mean=True` → pose mean is all zeros**, not `hands_mean`. (An early NumPy port wrongly added the hand means → 2.3 m offset.)
3. **`smplx.lbs.batch_rigid_transform` pads the homogeneous joint coordinate with 0** (`F.pad` default, unlike `transform_mat`'s explicit `value=1`). The transform correction is therefore `t - R@J`.
4. **ONNX export:** call `torch.backends.mha.set_fastpath_enabled(False)` first, otherwise `aten::_transformer_encoder_layer_fwd` fails to export.
5. **`HPSEstimator.forward` needs `item['poses']`** even for image-only inference — it is only used to read `.device`. A dummy tensor is fine.
6. **int8 quantization is rejected** — 19–40° joint error on real images. Keep fp16 as the only quantized variant. Do not introduce int8 without validating on real photographs (random-noise inputs are misleading).

## Commands

```bash
# inference (writes out/smplx_mesh.obj + smplx_params.npz)
python runtime/infer.py --image photo.jpg --out_dir out

# tri-view rendering (needs pyrender+trimesh)
python runtime/infer.py --image photo.jpg --out_dir out --render

# web UI
python webui.py            # http://127.0.0.1:6007

# numerical validation (requires the original torch pipeline + checkpoints)
python validate/compare_final.py     # lite vs torch: expect ~0 deg / ~1e-3 betas / ~3e-7 verts
python validate/validate_lbs.py      # NumPy LBS vs smplx LBS: expect ~3e-7
python validate/final_bench.py       # latency table
```

## Editing rules

- **Do not commit model files** (`models/*.onnx` are in `.gitignore`; they live on HuggingFace).
- Keep the fp32 path as the default (compatibility); fp16 is opt-in via `--onnx`.
- `runtime/infer.py` and `webui.py` share preprocessing — if you change resize/normalization, change **both** (the web UI imports `infer` so it inherits, but verify).
- Do not regenerate `smplx_neutral_lite_f32.npz` casually: `pack_smplx.py` keeps only the first 10 betas of `shapedirs`; the runtime assumes `num_betas=10`.
- Accuracy changes must be checked with `validate/compare_final.py`, not by eyeballing renders.
- Preserve the docstrings/comments: they encode non-obvious numerical conventions.

## Known limitations (do not "fix" these — they are model properties)

Average-pooled 512-d image token + ranking-only contrastive training + frozen features ⇒ weak fine articulation (crossed legs, lumbar bending), rare poses, heavy self-occlusion. Dedicated HMR models (SMPLer-X, HMR2.0, OSX, Multi-HMR) are strictly better for precise capture. This repo is about **efficient deployment of an existing representation model**, not about model accuracy research.
