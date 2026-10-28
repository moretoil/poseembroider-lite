"""【坐标系实验】确定 global_orient 的修正旋转。

背景：模型输出的全局朝向是 BEDLAM 相机坐标系（Y 向下），直接渲染会"躺平"。
做法：对 root 旋转分别左乘 identity / Rx(±90) / Ry(180) / Rz(180)，
      看哪种让站立输入恢复 y 方向跨度最大（身高 ~1.58m）。
结论：Rx(-90) 正确（会打印各候选的包围盒范围）。
"""

import os, sys
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/runtime')

import numpy as np
import trimesh
import pyrender
from PIL import Image
from infer import LitePoseEstimator, rotmat_to_rotvec
from numpy_lbs_fast import SMPLXNumpy, rodrigues

IMG = '/tmp/_lite_upload_1793208055552.png'
OUT = '/root/autodl-fs/poseembroider-lite/validate/orient'
os.makedirs(OUT, exist_ok=True)

est = LitePoseEstimator()
res = est.predict(IMG)
rotvec = res['rotvec'].copy()   # (1,22,3)
betas = res['betas']


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


FIXES = {
    'identity': np.eye(3, dtype=np.float32),
    'Rx180': Rx(np.pi),
    'Rx90': Rx(np.pi / 2),
    'Rx-90': Rx(-np.pi / 2),
    'Rz180': Rz(np.pi),
    'Ry180': Ry(np.pi),
}

renderer = pyrender.OffscreenRenderer(384, 384)
cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)


def look_at(pos, target, up):
    f = target - pos; f = f / np.linalg.norm(f)
    r = np.cross(f, up); r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, pos
    return m


for name, Rfix in FIXES.items():
    rv = rotvec.copy()
    R0 = rodrigues(rv[:, 0])          # (1,3,3)
    R0new = Rfix[None] @ R0
    rv[:, 0] = rotmat_to_rotvec(R0new)
    verts, _ = est.body.forward(betas, rv)
    v = verts[0]
    ext = v.max(0) - v.min(0)
    print(f"{name:9s} extents x={ext[0]:.2f} y={ext[1]:.2f} z={ext[2]:.2f}  "
          f"y_is_tallest={ext[1] > ext[0] and ext[1] > ext[2]}")
    # render front view (+z) for quick visual check
    mesh = trimesh.Trimesh(v, est.body.faces, process=False)
    center = (v.max(0) + v.min(0)) / 2
    scale = float(np.max(ext))
    pos = center + scale * 1.9 * np.array([0.0, 0.0, 1.0])
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.35] * 3)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    scene.add(cam, pose=look_at(pos, center, np.array([0., 1., 0.])))
    scene.add(pyrender.DirectionalLight(intensity=3.5), pose=look_at(pos, center, np.array([0., 1., 0.])))
    color, _ = renderer.render(scene)
    Image.fromarray(color).save(os.path.join(OUT, f'{name}.png'))

renderer.delete()
print('saved renders to', OUT)
