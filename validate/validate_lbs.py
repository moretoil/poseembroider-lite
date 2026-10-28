"""【LBS 对齐验证】NumPy 版 SMPL-X 蒙皮 vs smplx(torch) 官方实现。

用同一组 rotvec/betas 分别前向，比较 vertices 与 joints。
期望：max_abs_diff ~3e-7（float32 精度极限）。
常见错误提醒：flat_hand_mean=True 时 pose_mean 必须全零。
"""

import sys, json, numpy as np, torch
sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/export')
sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/runtime')
import roma
from numpy_lbs import SMPLXNumpy

SMPLX_DIR = '/root/autodl-tmp/poseembroider/tools/smpl_models/smplx'
NPZ = SMPLX_DIR + '/SMPLX_NEUTRAL.npz'

d = np.load('/root/autodl-fs/poseembroider-lite/validate/baseline.npz')
rotmat = torch.from_numpy(d['pose'])      # (1,22,3,3)
betas = torch.from_numpy(d['betas'])      # (1,10)
rotvec = roma.rotmat_to_rotvec(rotmat)    # (1,22,3)

# --- torch reference (smplx library)
import smplx
bm = smplx.SMPLX(SMPLX_DIR, gender='neutral', num_betas=10, use_pca=False, flat_hand_mean=True)
with torch.no_grad():
    out = bm(global_orient=rotvec[:, :1], body_pose=rotvec[:, 1:].reshape(1, -1), betas=betas)
ref_v = out.vertices.numpy()
ref_j = out.joints.numpy()[:, :55]

# --- numpy implementation
m = SMPLXNumpy(NPZ, num_betas=10)
v, j = m.forward(betas.numpy(), rotvec.numpy())

print("verts:", v.shape, "ref:", ref_v.shape)
print(f"verts  max_abs_diff: {np.abs(v - ref_v).max():.3e}")
print(f"joints max_abs_diff: {np.abs(j - ref_j).max():.3e}")
print(f"verts mean_abs_diff: {np.abs(v - ref_v).mean():.3e}")
