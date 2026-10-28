"""poseembroider-lite 推理入口（不依赖 PyTorch）。

流程：
    图片 -> 预处理(192x256, RGB, /255) -> ONNX(图像编码器+表征核心+姿势/体型头)
         -> 旋转矩阵转旋转向量 -> NumPy 版 SMPL-X LBS -> 顶点 -> 导出 OBJ / 渲染

用法：
    python runtime/infer.py --image photo.jpg --out_dir out
    python runtime/infer.py --image photo.jpg --out_dir out --render   # 需要 pyrender

依赖：numpy、onnxruntime、pillow；渲染另需 pyrender + trimesh + PyOpenGL。
"""
import os, sys, argparse, time

# 无显示器的 Linux 上用 OSMesa 软件渲染；必须在 import pyrender/OpenGL 之前设置
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

import numpy as np
import onnxruntime as ort
from PIL import Image

# 以便直接 `python runtime/infer.py` 运行时也能 import 同目录模块
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from numpy_lbs_fast import SMPLXNumpy  # NumPy 版 SMPL-X 蒙皮

# 默认模型：本地环境（Python 3.8 只能装 onnxruntime 1.19）用 fp32 最稳；
# 若 onnxruntime >= 1.31，可改用 fp16 模型（体积减半、精度损失仅 ~0.12°）。
DEFAULT_ONNX = os.path.join(HERE, '../models/image_to_params.onnx')
DEFAULT_SMPLX = os.path.join(HERE, '../models/smplx_neutral_lite_f32.npz')


def rotmat_to_rotvec(R):
    """旋转矩阵 (...,3,3) -> 轴角向量 (...,3)，纯 NumPy 实现（对应 torch 的 roma.rotmat_to_rotvec）。

    分三种情况处理：
      - theta ≈ 0   : 旋转向量为 0（单位矩阵）
      - theta ≈ π   : 用特征向量求轴（数值上 sin(theta)≈0，不能用反对称部分）
      - 一般情况    : axis = (R - Rᵀ) / (2 sin(theta))，乘上 theta
    """
    R = R.astype(np.float64)  # 用 float64 提高小角度时的数值稳定性
    tr = np.trace(R, axis1=-2, axis2=-1)
    theta = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    out = np.zeros(R.shape[:-2] + (3,), np.float64)
    small = theta < 1e-6
    big = theta > np.pi - 1e-5
    mid = ~(small | big)

    if mid.any():
        Rm, tm = R[mid], theta[mid]
        # 旋转矩阵的反对称部分给出 "2 sin(theta) * axis"
        axis = np.stack([Rm[..., 2, 1] - Rm[..., 1, 2],
                         Rm[..., 0, 2] - Rm[..., 2, 0],
                         Rm[..., 1, 0] - Rm[..., 0, 1]], axis=-1)
        axis /= (2.0 * np.sin(tm))[..., None]
        out[mid] = axis * tm[..., None]

    if big.any():
        # theta≈π 时 sin(theta)≈0，改为求 R 的 +1 特征值对应的特征向量作为旋转轴
        for i in np.where(big.reshape(-1))[0]:
            Rm = R.reshape(-1, 3, 3)[i]
            w, V = np.linalg.eig(Rm)
            axis = np.real(V[:, np.argmin(np.abs(w - 1.0))])
            axis /= np.linalg.norm(axis)
            out.reshape(-1, 3)[i] = axis * theta.reshape(-1)[i]

    return out.astype(np.float32)


class LitePoseEstimator:
    """轻量推理封装：ONNX Runtime 会话 + SMPL-X 蒙皮模型。"""

    def __init__(self, onnx_path=DEFAULT_ONNX, smplx_npz=DEFAULT_SMPLX, num_betas=10, threads=0):
        so = ort.SessionOptions()
        if threads > 0:                       # 限制线程数，避免与 Flask/BLAS 线程抢占
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
        self.body = SMPLXNumpy(smplx_npz, num_betas=num_betas)
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, image_path):
        """图片 -> 模型输入张量 (1,3,256,192)。

        必须与原版管线一致：cv2 读图、BGR->RGB、INTER_AREA 缩放到 (192,256)、/255。
        插值方式不同会明显改变输出（腿/骨盆区域尤其敏感），不要随意替换。
        没有 cv2 时退回 PIL（结果会有细小差异）。
        """
        try:
            import cv2
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_AREA)
        except ImportError:
            img = np.asarray(Image.open(image_path).convert('RGB').resize((192, 256), Image.BILINEAR))
        x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(x)

    def predict(self, image_path):
        """单张图片推理，返回 dict(rotmat, rotvec, betas, verts, joints, 耗时)。"""
        x = self.preprocess(image_path)

        t0 = time.perf_counter()
        rotmat, betas = self.sess.run(None, {self.input_name: x})
        t_infer = time.perf_counter() - t0

        # —— 全局朝向修正（关键）——
        # 模型输出的 global_orient 位于 BEDLAM 的相机坐标系（Y 轴向下），
        # 若不修正，人体网格会"躺平"（俯视图反而像正面图）。
        # 左乘 Rx(-90°) 转到标准世界坐标系：Y 向上、面朝 +Z。
        R_fix = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, -1.0, 0.0]], np.float32)  # Rx(-90°)
        rotmat = np.ascontiguousarray(rotmat)
        rotmat[:, 0] = R_fix @ rotmat[:, 0]
        rotvec = rotmat_to_rotvec(rotmat)

        t0 = time.perf_counter()
        verts, joints = self.body.forward(betas.astype(np.float32), rotvec)
        t_lbs = time.perf_counter() - t0

        return dict(rotmat=rotmat, rotvec=rotvec, betas=betas, verts=verts, joints=joints,
                    t_infer=t_infer, t_lbs=t_lbs)


def export_obj(path, verts, faces):
    """把网格写成 Wavefront OBJ（Blender 导入：Up Axis = Y）。"""
    with open(path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fc in faces:
            f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")  # OBJ 索引从 1 开始


def render_views(verts, faces, out_dir, prefix='render_view', angles=(-45, 0, 45), size=512):
    """（可选）渲染三个环绕视角的 PNG，需要 pyrender + trimesh。"""
    import trimesh, pyrender
    mesh = trimesh.Trimesh(verts, faces, process=False)
    center = (verts.max(0) + verts.min(0)) / 2
    scale = float(np.max(verts.max(0) - verts.min(0)))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    renderer = pyrender.OffscreenRenderer(size, size)
    for a in angles:
        rad = np.radians(a)
        pos = center + scale * 1.7 * np.array([np.sin(rad), 0.12, np.cos(rad)])
        # 构造 look-at 相机位姿：相机 -Z 朝向目标
        f = center - pos; f /= np.linalg.norm(f)
        r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = r, u, -f, pos
        # 每个视角新建 Scene（避免 pyrender/networkx 版本的 remove_node 兼容问题）
        scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.35] * 3)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        scene.add(cam, pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=3.5), pose=pose)
        color, _ = renderer.render(scene)
        Image.fromarray(color).save(os.path.join(out_dir, f'{prefix}{a:+d}.png'))
    renderer.delete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='poseembroider-lite 单图推理')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--out_dir', default='./out', help='输出目录')
    parser.add_argument('--onnx', default=DEFAULT_ONNX, help='ONNX 模型路径（可换成 fp16 版）')
    parser.add_argument('--smplx', default=DEFAULT_SMPLX, help='SMPL-X 精简 npz 路径')
    parser.add_argument('--render', action='store_true', help='渲染三视角（需要 pyrender）')
    parser.add_argument('--repeat', type=int, default=1, help='重复推理用于测速')
    args = parser.parse_args()

    est = LitePoseEstimator(args.onnx, args.smplx)
    res = est.predict(args.image)

    # 重复推理测稳态耗时（第一次可能包含线程池/内存预热）
    lat = []
    for _ in range(args.repeat - 1):
        r = est.predict(args.image)
        lat.append(r['t_infer'] + r['t_lbs'])

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, 'smplx_params.npz'),
             rotvec=res['rotvec'], betas=res['betas'], verts=res['verts'])
    export_obj(os.path.join(args.out_dir, 'smplx_mesh.obj'), res['verts'][0], est.body.faces)

    print(f"infer {res['t_infer']*1000:.1f} ms | numpy LBS {res['t_lbs']*1000:.1f} ms")
    if lat:
        print(f"steady-state latency: {np.mean(lat)*1000:.1f} ms (n={len(lat)})")
    print("betas:", np.round(res['betas'][0], 3).tolist())
    if args.render:
        render_views(res['verts'][0], est.body.faces, args.out_dir)
        print("rendered 3 views")
    print("DONE", args.out_dir)
