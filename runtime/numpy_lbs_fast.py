"""NumPy 版 SMPL-X 线性混合蒙皮（LBS），把 smplx/lbs.py 翻译成不依赖 torch 的实现。

输入是模型预测的 22 个关节旋转（1 个全局 + 21 个身体关节，轴角表示）；
其余 33 个关节（下颌 jaw 1 + 双眼 2 + 左右手 30）保持 "平手" 均值姿态，
与 smplx.SMPLX(flat_hand_mean=True) 的行为一致。

数值上与 torch 版对齐：顶点最大误差约 3e-7（float32 精度极限）。
"""
import numpy as np

NUM_VERTS = 10475          # SMPL-X 顶点数
NUM_JOINTS = 55            # SMPL-X 全部关节数
POSE_DIM = (NUM_JOINTS - 1) * 9  # 486 = (55-1) 个关节的旋转矩阵(3x3)展平


def rodrigues(r):
    """轴角 (...,3) -> 旋转矩阵 (...,3,3)（罗德里格斯公式）。"""
    theta = np.linalg.norm(r, axis=-1, keepdims=True)
    r_norm = r / np.maximum(theta, 1e-8)   # 零角度时用 1e-8 避免除零
    # 旋转轴对应的反对称矩阵 K
    K = np.zeros(r.shape[:-1] + (3, 3), dtype=np.float32)
    K[..., 0, 1] = -r_norm[..., 2]
    K[..., 0, 2] = r_norm[..., 1]
    K[..., 1, 0] = r_norm[..., 2]
    K[..., 1, 2] = -r_norm[..., 0]
    K[..., 2, 0] = -r_norm[..., 1]
    K[..., 2, 1] = r_norm[..., 0]
    eye = np.eye(3, dtype=np.float32)
    sin, cos = np.sin(theta)[..., None], np.cos(theta)[..., None]
    # R = I + sin(θ)K + (1-cos(θ))K²
    return eye + sin * K + (1.0 - cos) * (K @ K)


class SMPLXNumpy:
    """从精简 npz（只含前 10 个 betas 的 shapedirs）加载 SMPL-X 并做前向蒙皮。"""

    def __init__(self, npz_path, num_betas=10):
        d = np.load(npz_path)
        self.v_template = d['v_template'].astype(np.float32)                    # (10475,3) 中性姿态模板
        sd = d['shapedirs'].astype(np.float32)[:, :, :num_betas]
        self.shapedirs_flat = sd.reshape(NUM_VERTS * 3, num_betas)              # (V*3,10) 体型基
        self.posedirs = d['posedirs'].astype(np.float32).reshape(NUM_VERTS * 3, POSE_DIM)  # (V*3,486) 姿态修正基
        self.J_regressor = d['J_regressor'].astype(np.float32)                  # (55,V) 顶点->关节回归矩阵
        self.weights = d['weights'].astype(np.float32)                          # (V,55) 蒙皮权重
        parents = d['kintree_table'][0].astype(np.int64)
        parents[parents == 4294967295] = -1                                     # uint32 的 -1 编码
        self.parents = parents                                                  # 骨骼树父关节索引
        self.faces = d['faces'].astype(np.int64) if 'faces' in d else d['f'].astype(np.int64)
        # 关键：flat_hand_mean=True 时 pose_mean 全零（不是 hands_mean！）
        self.pose_mean = np.zeros(165, dtype=np.float32)

    def forward(self, betas, pose_rotvec):
        """betas (B,10) + 关节轴角 (B,22,3) -> 顶点 (B,V,3)、关节 (B,55,3)。"""
        B = pose_rotvec.shape[0]

        # 1) 拼出 165 维全姿态：[global(3), body(21*3), jaw(3), eyes(6), hands(2*15*3)]
        #    未预测的关节置零 + pose_mean(全零) => 保持平手姿态
        full = np.zeros((B, 165), dtype=np.float32)
        full[:, :3] = pose_rotvec[:, 0]
        full[:, 3:66] = pose_rotvec[:, 1:].reshape(B, -1)
        full += self.pose_mean[None]

        # 2) 轴角 -> 旋转矩阵 (B,55,3,3)
        R = rodrigues(full.reshape(-1, 3)).reshape(B, NUM_JOINTS, 3, 3)

        # 3) 体型形变：v_shaped = v_template + shapedirs @ betas
        v0 = self.v_template.reshape(1, NUM_VERTS * 3)
        v_shaped = (v0 + (self.shapedirs_flat @ betas.T).T).reshape(B, NUM_VERTS, 3)

        # 4) 姿态形变（pose blend shapes）：v_posed = v_shaped + posedirs @ (R[1:] - I)
        pose_feat = (R[:, 1:] - np.eye(3, dtype=np.float32)).reshape(B, POSE_DIM)
        v_posed = v_shaped + (pose_feat @ self.posedirs.T).reshape(B, NUM_VERTS, 3)

        # 5) 回归关节点，并求相对父关节的偏移（骨骼树）
        J = np.matmul(self.J_regressor, v_shaped)   # (B,55,3)
        rel = J.copy()
        rel[:, 1:] -= J[:, self.parents[1:]]

        # 6) 每个关节的局部变换矩阵 [R | t]，再沿骨骼树连乘得到全局变换
        Tm = np.zeros((B, NUM_JOINTS, 4, 4), dtype=np.float32)
        Tm[..., :3, :3] = R
        Tm[..., :3, 3] = rel
        Tm[..., 3, 3] = 1.0
        chain = [Tm[:, 0]]
        for i in range(1, NUM_JOINTS):
            chain.append(chain[self.parents[i]] @ Tm[:, i])
        T = np.stack(chain, axis=1)

        # posed joints = 链式变换的平移列
        joints = T[..., :3, 3].copy()
        # 关键细节：smplx 的 batch_rigid_transform 里齐次坐标用 F.pad 补的是 0（不是 1），
        # 所以相对变换的平移修正是 t - R@J（而不是 t - (R@J + t)）
        J_h = np.concatenate([J, np.zeros((B, NUM_JOINTS, 1), np.float32)], axis=-1)
        T[..., :3, 3] -= (T @ J_h[..., None])[..., 0][..., :3]

        # 7) 蒙皮：把每个关节的变换按权重混合到每个顶点上
        #    Tvw[v] = Σ_j weights[v,j] * T[j]   （4x4）
        T_flat = T.reshape(B, NUM_JOINTS, 16)
        Tvw = (self.weights @ T_flat[0]).reshape(1, NUM_VERTS, 4, 4)
        if B > 1:
            Tvw = np.stack([(self.weights @ T_flat[b]).reshape(NUM_VERTS, 4, 4) for b in range(B)])

        # 8) 对每个顶点做 4x4 变换（手写分量展开比批量 matmul 快很多）
        x, y, z = v_posed[..., 0], v_posed[..., 1], v_posed[..., 2]
        out = np.empty((B, NUM_VERTS, 3), dtype=np.float32)
        for i in range(3):
            out[..., i] = (Tvw[..., i, 0] * x + Tvw[..., i, 1] * y
                           + Tvw[..., i, 2] * z + Tvw[..., i, 3])
        return out, joints
