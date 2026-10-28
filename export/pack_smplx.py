"""【SMPL-X 瘦身】把官方 SMPLX_NEUTRAL.npz 从 104MB 压到 27.7MB。

只保留推理必需项：
  v_template、J_regressor、weights、kintree_table、faces、posedirs
  shapedirs 只保留前 10 个 betas（模型只预测 10 个体型系数）
并保存为 npz_compressed；另生成一份 float16 版本（13MB，未默认使用）。

输出：models/smplx_neutral_lite_f32.npz / smplx_neutral_lite_f16.npz
注意：运行时假定 num_betas=10，改动此文件需同步 runtime/numpy_lbs_fast.py。
"""


import numpy as np, os
SRC = '/root/autodl-tmp/poseembroider/tools/smpl_models/smplx/SMPLX_NEUTRAL.npz'
OUT32 = '/root/autodl-fs/poseembroider-lite/models/smplx_neutral_lite_f32.npz'
OUT16 = '/root/autodl-fs/poseembroider-lite/models/smplx_neutral_lite_f16.npz'
d = np.load(SRC)
common = dict(
    v_template=d['v_template'].astype(np.float32),
    shapedirs=d['shapedirs'][:, :, :10].astype(np.float32),
    J_regressor=d['J_regressor'].astype(np.float32),
    weights=d['weights'].astype(np.float32),
    kintree_table=d['kintree_table'],
    faces=d['f'].astype(np.int64),
)
posedirs32 = d['posedirs'].astype(np.float32)
np.savez_compressed(OUT32, posedirs=posedirs32, **common)
np.savez_compressed(OUT16, posedirs=d['posedirs'].astype(np.float16), **{k: (v.astype(np.float16) if v.dtype==np.float32 else v) for k,v in common.items()})
for p in [OUT32, OUT16]:
    print(p, f'{os.path.getsize(p)/1e6:.1f} MB')
