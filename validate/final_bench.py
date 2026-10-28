"""【性能基准】fp32/fp16 ONNX 的精度与 CPU 延迟对比。

输出：文件体积、关节旋转误差、betas 误差、单张延迟、端到端延迟。
注意：测延迟前先 warmup 若干次，否则会包含线程池初始化开销。
"""


import sys, time, numpy as np, cv2, torch
sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/runtime')
import roma, onnxruntime as ort
from numpy_lbs_fast import SMPLXNumpy

MODELS = '/root/autodl-fs/poseembroider-lite/models'
img = cv2.imread('/root/autodl-tmp/poseembroider/testpose2_crop.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_AREA)
x = (img.astype(np.float32).transpose(2,0,1)[None] / 255.0)

# torch reference for accuracy
sys.path.insert(0, '/root/autodl-fs/poseembroider-lite/export')
import lite_encoders; lite_encoders.apply()
torch.backends.mha.set_fastpath_enabled(False)
import poseembroider.config as config
from poseembroider.augmentations import DataProcessingModule
from poseembroider.downstream.pose_estimation.evaluate_pose_estimation import load_model
from export_params_onnx import ImageToParams
mp = __import__('json').load(open(config.PRETRAINED_MODEL_DICT))['hpsestimator_model_poseembroider_bedlamscript']
hps, _ = load_model(mp, 'cpu'); hps.eval()
w = ImageToParams(hps, hps.representation_wrapper.representation_model).eval()
xn = DataProcessingModule(phase='eval')({'images': torch.from_numpy(x*255).to(torch.uint8), 'poses': torch.zeros((1,1,3))})['images']
with torch.no_grad():
    rot_t, betas_t = w(xn)

body = SMPLXNumpy(MODELS + '/smplx_neutral_lite_f32.npz')
for tag in ['fp32', 'fp16']:
    path = f'{MODELS}/image_to_params.onnx' if tag=='fp32' else f'{MODELS}/image_to_params_fp16.onnx'
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    import os
    size = os.path.getsize(path)/1e6
    r, b = sess.run(None, {'images': x.astype(np.float32)})
    err = roma.rotmat_geodesic_distance(torch.from_numpy(r), rot_t).numpy()*180/np.pi
    for _ in range(3): sess.run(None, {'images': x.astype(np.float32)})
    t0 = time.perf_counter()
    for _ in range(10): sess.run(None, {'images': x.astype(np.float32)})
    t_onnx = (time.perf_counter()-t0)/10
    rotvec = None
    print(f'{tag}: {size:.0f} MB | rot err {err.mean():.3f} deg | betas diff {np.abs(b-betas_t.numpy()).max():.4f} | onnx {t_onnx*1000:.0f} ms')
    if tag == 'fp16':
        import infer as rt
        rv = rt.rotmat_to_rotvec(r)
        body.forward(b, rv)
        t0 = time.perf_counter()
        for _ in range(10): body.forward(b, rv)
        t_lbs = (time.perf_counter()-t0)/10
        print(f'  + numpy LBS {t_lbs*1000:.1f} ms | END-TO-END {(t_onnx+t_lbs)*1000:.0f} ms/img')
