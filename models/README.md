# models/

Model artefacts are **not** stored in this repository. Download them from Hugging Face:

**https://huggingface.co/foxdream/poseembroider-lite/tree/main**

| File | Size | Use |
|---|---|---|
| `image_to_params_fp16.onnx` | ~179 MB | recommended (onnxruntime ≥ 1.31) |
| `image_to_params.onnx` | ~357 MB | fp32 fallback (Python 3.8 / old onnxruntime) |
| `smplx_neutral_lite_f32.npz` | ~28 MB | slimmed SMPL-X body model (required) |

```bash
huggingface-cli download foxdream/poseembroider-lite image_to_params_fp16.onnx --local-dir models
huggingface-cli download foxdream/poseembroider-lite smplx_neutral_lite_f32.npz --local-dir models
```

Place files exactly in this directory (`models/`), or pass explicit paths with
`--onnx` / `--smplx` to `runtime/infer.py`.
