"""【实验存档：int8 静态量化】QDQ 格式 + COCO 校准图。

结论：不适用于本模型。
  - int8 动态量化：真实图像上关节旋转误差 19.3 度（随机噪声输入只有 0.8 度，具有欺骗性）
  - int8 静态量化(QDQ)：40.1 度，且 CPU 上因 dequant 开销反而更慢（1.5s/张）
原因：ViT + 对比学习表征对数值扰动过于敏感。最终选择 fp16。

本脚本仅作记录，不要用于生产模型。
"""

import sys, os, glob
import numpy as np
import cv2
import torch
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

FP32 = '/root/autodl-fs/poseembroider-lite/models/image_to_params.onnx'
OUT = '/root/autodl-fs/poseembroider-lite/models/image_to_params_int8_static.onnx'
CALIB_DIR = '/root/autodl-fs/poseembroider-lite/validate/calib'


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x)


class Reader(CalibrationDataReader):
    def __init__(self, files):
        self.data = [preprocess(f) for f in files]
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        x = self.data[self.idx]
        self.idx += 1
        return {'images': x}

    def rewind(self):
        self.idx = 0


files = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
print(f"{len(files)} calibration images")

quantize_static(
    FP32, OUT, Reader(files),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)
print("quantized:", OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB")

# ---------------- evaluate on the real test image vs torch reference
sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/export')
sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/runtime')
import lite_encoders
lite_encoders.apply()
torch.backends.mha.set_fastpath_enabled(False)
import roma
import onnxruntime as ort
import poseembroider.config as config
from poseembroider.augmentations import DataProcessingModule
from poseembroider.downstream.pose_estimation.evaluate_pose_estimation import load_model
from export_params_onnx import ImageToParams

IMG = '/root/autodl-tmp/poseembroider/testpose2_crop.jpg'
model_path = __import__('json').load(open(config.PRETRAINED_MODEL_DICT))['hpsestimator_model_poseembroider_bedlamscript']
hps, _ = load_model(model_path, 'cpu')
hps.eval()
wrapper = ImageToParams(hps, hps.representation_wrapper.representation_model).eval()

img = cv2.imread(IMG)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_AREA)
x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
dp = DataProcessingModule(phase='eval')
item = dp({'images': x, 'poses': torch.zeros((1, 1, 3))})
x_norm = item['images']
with torch.no_grad():
    rot_t, betas_t = wrapper(x_norm)

sess = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
import time
r_list = []
for _ in range(3):
    t0 = time.perf_counter()
    r, b = sess.run(None, {'images': x_norm.numpy().astype(np.float32)})
    r_list.append(time.perf_counter() - t0)
err = roma.rotmat_geodesic_distance(torch.from_numpy(r), rot_t).numpy() * 180 / np.pi
print(f"static-int8: rot err mean {err.mean():.3f} deg | max {err.max():.3f} | betas max {np.abs(b-betas_t.numpy()).max():.4f}")
print(f"latency: {np.mean(r_list)*1000:.0f} ms")
