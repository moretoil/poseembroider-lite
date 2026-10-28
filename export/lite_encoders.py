"""【第一步：裁剪无用编码器】构建模型时跳过 text/pose 预训练编码器。

背景：原版 PoseEmbroider 初始化时会加载 posetext 文本编码器(313MB) + distilbert(268MB)
      + posevae 姿态编码器，但"仅图像输入"的推理从不调用它们。
做法：monkey-patch TextEncoder/PoseEncoder 的 __init__，只保留 projection 头
      （权重仍在主模型 checkpoint 里，必须保留结构才能加载 state_dict）。
用法：在构建模型之前执行 -- import lite_encoders; lite_encoders.apply()
验证：与完整加载的输出逐位一致（max_abs_diff = 0），见 validate/compare_final.py。
"""
import torch
import torch.nn as nn
import poseembroider.model_modules as M


def _lite_text_init(self, latentD=512, text_encoder_name='posetext', projection_type="layerplus"):
    nn.Module.__init__(self)
    self.latentD = latentD
    self.pretrained_text_encoder = None          # was: posetext checkpoint (313MB) + distilbert (268MB)
    self.pretrained_output_dim = 512             # posetext latentD
    self.projection_type = projection_type
    self.text_projection = M.get_projection(projection_type, self.pretrained_output_dim, self.latentD)


def _lite_pose_init(self, latentD=512, num_body_joints=22, pose_encoder_name='posevae', projection_type='layerplus'):
    nn.Module.__init__(self)
    self.latentD = latentD
    self.num_body_joints = num_body_joints
    self.pretrained_pose_encoder = None          # was: posevae checkpoint (4MB)
    self.pretrained_output_dim = 512             # posevae encoder hidden dim (encoder[7] output)
    self.projection_type = projection_type
    self.pose_projection = M.get_projection(projection_type, self.pretrained_output_dim, self.latentD)


def _noop_grad(self, value):
    pass


def apply():
    M.TextEncoder.__init__ = _lite_text_init
    M.PoseEncoder.__init__ = _lite_pose_init
    M.TextEncoder.set_pretrained_weights_grad = _noop_grad
    M.PoseEncoder.set_pretrained_weights_grad = _noop_grad
    print("[lite] patched TextEncoder / PoseEncoder (pretrained encoders skipped)")
