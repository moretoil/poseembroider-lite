"""【第三步：fp16 转换】fp32 ONNX -> fp16（体积减半，精度损失约 0.12 度）。

使用 onnxruntime.transformers.float16（对 Transformer 模型的 Cast 处理更稳，
onnxconverter-common 的版本会留下类型不匹配的节点导致加载失败）。

输出：models/image_to_params_fp16.onnx（约 179MB）

注意：onnxruntime < 1.31 加载 fp16 模型会输出 NaN（Python 3.8 只能装 ORT 1.19），
      这种环境请使用 fp32 模型。
"""


import os, sys, time
import numpy as np, cv2, torch
sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/export')
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

FP32 = '/root/autodl-fs/poseembroider-lite/models/image_to_params.onnx'
OUT = '/root/autodl-fs/poseembroider-lite/models/image_to_params_fp16.onnx'
m = onnx.load(FP32)
m16 = convert_float_to_float16(m, keep_io_types=True)
onnx.save(m16, OUT)
print('fp16 saved:', f'{os.path.getsize(OUT)/1e6:.1f} MB')

import lite_encoders
lite_encoders.apply()
torch.backends.mha.set_fastpath_enabled(False)
import roma, onnxruntime as ort
import poseembroider.config as config
from poseembroider.augmentations import DataProcessingModule
from poseembroider.downstream.pose_estimation.evaluate_pose_estimation import load_model
from export_params_onnx import ImageToParams

IMG = '/root/autodl-tmp/poseembroider/testpose2_crop.jpg'
model_path = __import__('json').load(open(config.PRETRAINED_MODEL_DICT))['hpsestimator_model_poseembroider_bedlamscript']
hps, _ = load_model(model_path, 'cpu'); hps.eval()
wrapper = ImageToParams(hps, hps.representation_wrapper.representation_model).eval()
img = cv2.imread(IMG); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_AREA)
x = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
dp = DataProcessingModule(phase='eval')
x_norm = dp({'images': x, 'poses': torch.zeros((1,1,3))})['images']
with torch.no_grad():
    rot_t, betas_t = wrapper(x_norm)

sess = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
lat = []
for _ in range(3):
    t0 = time.perf_counter()
    r, b = sess.run(None, {'images': x_norm.numpy().astype(np.float32)})
    lat.append(time.perf_counter()-t0)
err = roma.rotmat_geodesic_distance(torch.from_numpy(r), rot_t).numpy()*180/np.pi
print(f'fp16: rot err mean {err.mean():.3f} deg | max {err.max():.3f} | betas max {np.abs(b-betas_t.numpy()).max():.4f}')
print(f'fp16 latency: {np.mean(lat)*1000:.0f} ms')
