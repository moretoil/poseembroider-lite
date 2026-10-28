"""【第二步：导出 ONNX】把 图像->SMPL-X参数 整条链导出为 ONNX（fp32）。

链路：SMPLer-X ViT-B -> image projection(768->512) -> L2 归一化
      -> + modality encoding -> intermodality token -> embroider transformer core
      -> pose head (22x3x3 旋转矩阵) + shape head (10 个 betas)

前置条件（原版 torch 环境）：
  * pip install poseembroider（本仓库源码），并准备好 4 个 checkpoint
  * tools/torch_models/smplerx/smpler_x_b32.pth.tar
  * tools/smpl_models/smplx/SMPLX_NEUTRAL.npz
输出：models/image_to_params.onnx（约 357MB）

【关键】必须先关闭 Transformer 融合算子，否则导出报
  UnsupportedOperatorError: aten::_transformer_encoder_layer_fwd
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

# NOTE: PyTorch fuses TransformerEncoderLayer into an aten op that ONNX
# cannot export; disable the fast path so the plain attention is traced.
torch.backends.mha.set_fastpath_enabled(False)

sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/export')
import lite_encoders
lite_encoders.apply()

import poseembroider.config as config
from poseembroider.downstream.pose_estimation.evaluate_pose_estimation import load_model

OUT = '/root/autodl-fs/poseembroider-lite/models'
os.makedirs(OUT, exist_ok=True)

DEVICE = torch.device('cpu')
model_path = json.load(open(config.PRETRAINED_MODEL_DICT))['hpsestimator_model_poseembroider_bedlamscript']
hps, _ = load_model(model_path, DEVICE)
hps.eval()
rep_model = hps.representation_wrapper.representation_model


class ImageToParams(nn.Module):
    def __init__(self, hps, rep_model):
        super().__init__()
        self.hps = hps
        self.rep = rep_model

    def forward(self, images):
        mf = self.rep.get_modality_features(images=images)
        img = mf['image_tokens_emb'] + self.rep.modality_encoding_image
        inter = self.rep.intermodality_token.expand(images.shape[0], -1, -1)
        x = torch.cat((inter, img), dim=1)
        feat = self.rep.forward_in_embroider_core(x)
        rotmat = self.hps.pose_prediction_head(feat)
        betas = self.hps.bodyshape_prediction_head(feat)
        return rotmat, betas


wrapper = ImageToParams(hps, rep_model).eval()
dummy = torch.rand(1, 3, 256, 192)

with torch.no_grad():
    t0 = time.perf_counter()
    ref = wrapper(dummy)
    print("reference forward:", f"{time.perf_counter()-t0:.3f}s")
print("ref rotmat:", tuple(ref[0].shape), "ref betas:", tuple(ref[1].shape))

onnx_path = os.path.join(OUT, 'image_to_params.onnx')
torch.onnx.export(
    wrapper, dummy, onnx_path,
    input_names=['images'], output_names=['pose_rotmat', 'betas'],
    dynamic_axes={'images': {0: 'batch'}, 'pose_rotmat': {0: 'batch'}, 'betas': {0: 'batch'}},
    opset_version=17, do_constant_folding=True,
)
print("exported:", onnx_path, f"{os.path.getsize(onnx_path)/1e6:.1f} MB")

# ---------------- onnxruntime verification
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
outs = sess.run(None, {'images': dummy.numpy()})
d0 = np.abs(outs[0] - ref[0].detach().numpy()).max()
d1 = np.abs(outs[1] - ref[1].detach().numpy()).max()
print(f"ORT vs torch: rotmat max_abs_diff={d0:.3e} | betas max_abs_diff={d1:.3e}")

# ---------------- int8 dynamic quantization
from onnxruntime.quantization import quantize_dynamic, QuantType
int8_path = os.path.join(OUT, 'image_to_params_int8.onnx')
quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
print("quantized:", int8_path, f"{os.path.getsize(int8_path)/1e6:.1f} MB")

sess8 = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
outs8 = sess8.run(None, {'images': dummy.numpy()})
d0q = np.abs(outs8[0] - ref[0].detach().numpy())
d1q = np.abs(outs8[1] - ref[1].detach().numpy())
print(f"int8 vs fp32: rotmat max_abs_diff={d0q.max():.3e} mean={d0q.mean():.3e} | betas max_abs_diff={d1q.max():.3e}")

# rotation error in degrees (geodesic)
import roma
r_ref = torch.from_numpy(outs[0]); r_q = torch.from_numpy(outs8[0])
# re-orthonormalize not needed; compute geodesic distance
err_deg = roma.rotmat_geodesic_distance(r_q, r_ref).mean().item() * 180 / np.pi
print(f"int8 mean joint rotation error: {err_deg:.3f} deg")

# ---------------- timing on CPU
imgs = np.random.rand(1, 3, 256, 192).astype(np.float32)
for name, s in [('fp32', sess), ('int8', sess8)]:
    for _ in range(2): s.run(None, {'images': imgs})
    t0 = time.perf_counter()
    for _ in range(10): s.run(None, {'images': imgs})
    print(f"{name} CPU latency: {(time.perf_counter()-t0)/10*1000:.1f} ms/img")
