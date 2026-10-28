"""【验收脚本】lite 输出 vs 原版 torch 管线。

比较项：
  * 关节旋转误差（测地距离，单位：度）  —— fp16 期望 ~0.12 度
  * betas 最大绝对差                    —— 期望 < 0.01
  * 顶点误差（毫米）                    —— NumPy LBS 期望 < 1e-4 mm 量级
输入：validate/baseline.npz（torch 基准）+ validate/lite_out/smplx_params.npz
"""


import sys, numpy as np, torch
sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/runtime')
import roma
from numpy_lbs import SMPLXNumpy

base = np.load('/root/autodl-fs/poseembroider-lite/validate/baseline.npz')   # original torch pipeline
lite = np.load('/root/autodl-fs/poseembroider-lite/validate/lite_out/smplx_params.npz')  # onnx int8 + numpy LBS

R_ref = torch.from_numpy(base['pose'])          # (1,22,3,3) rotmats from torch
betas_ref = base['betas']                        # (1,10)
betas_lite = lite['betas']
rotvec_lite = lite['rotvec']
R_lite = roma.rotvec_to_rotmat(torch.from_numpy(rotvec_lite))

err = roma.rotmat_geodesic_distance(R_lite, R_ref).numpy() * 180 / np.pi
print(f"joint rotation error: mean {err.mean():.3f} deg | max {err.max():.3f} deg")
print(f"betas max abs diff: {np.abs(betas_ref - betas_lite).max():.4f}")

# reference vertices via torch smplx
import smplx
bm = smplx.SMPLX('/root/autodl-tmp/poseembroider/tools/smpl_models/smplx', gender='neutral',
                 num_betas=10, use_pca=False, flat_hand_mean=True)
rv = roma.rotmat_to_rotvec(R_ref)
with torch.no_grad():
    v_ref = bm(global_orient=rv[:, :1], body_pose=rv[:, 1:].reshape(1, -1), betas=torch.from_numpy(betas_ref)).vertices.numpy()
v_lite = lite['verts']
print(f"vertices: max {np.abs(v_ref - v_lite).max()*1000:.1f} mm | mean {np.abs(v_ref - v_lite).mean()*1000:.1f} mm")
