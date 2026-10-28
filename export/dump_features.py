"""【基准生成】用原版 torch 管线跑一张图，把 pose/betas 存成 npz。

用途：作为数值对齐的 ground truth（baseline.npz），供 validate/compare_final.py 对比。
用法：
  python dump_features.py --out baseline.npz          # 原版完整加载
  python dump_features.py --lite --out lite.npz       # 应用 lite_encoders 裁剪
注意：预处理必须与 runtime/infer.py 一致（cv2 INTER_AREA, 192x256, /255）。
"""

import os, sys, json, time, argparse
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
import numpy as np
import cv2
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--lite', action='store_true')
parser.add_argument('--image', default='/root/autodl-tmp/poseembroider/testpose2_crop.jpg')
parser.add_argument('--out', required=True)
args = parser.parse_args()

t0 = time.perf_counter()
if args.lite:
    sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/export')
    import lite_encoders
    lite_encoders.apply()

import poseembroider.config as config
from poseembroider.augmentations import DataProcessingModule
from poseembroider.downstream.pose_estimation.evaluate_pose_estimation import load_model

DEVICE = torch.device('cuda:0')
model_path = json.load(open(config.PRETRAINED_MODEL_DICT))['hpsestimator_model_poseembroider_bedlamscript']
model, _ = load_model(model_path, DEVICE)
dp = DataProcessingModule(phase="eval")

img = cv2.imread(args.image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_AREA)
img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

item = {"images": img_t, "poses": torch.zeros((1, 1, 3))}
item = dp(item)
item = {k: v.to(DEVICE) for k, v in item.items()}
with torch.no_grad() and torch.inference_mode():
    pred_rotmat, pred_betas = model(item=item, representation_model_input_types=["images"])
torch.cuda.synchronize()
t_build_and_infer = time.perf_counter() - t0

np.savez(args.out, pose=pred_rotmat.cpu().numpy(), betas=pred_betas.cpu().numpy())
print(f"mode={'lite' if args.lite else 'baseline'} build+infer={t_build_and_infer:.1f}s")
print("pose shape:", tuple(pred_rotmat.shape), "betas:", pred_betas[0].tolist())
