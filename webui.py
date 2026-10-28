"""poseembroider-lite 本地 Web UI（不强制依赖渲染库）。

功能：浏览器上传图片 -> 服务端推理 -> 保存 OBJ/NPZ 并展示。
  - 如果装了 pyrender + trimesh：额外显示正/侧/顶三视角渲染
  - 如果没装：页面直接给出 OBJ 路径，可用 Blender 打开

运行：python webui.py    （监听 127.0.0.1:6007）
依赖：numpy、onnxruntime、pillow、flask（渲染为可选）
"""
import os, sys, io, time, base64

# 项目根 + runtime 目录都加入模块搜索路径，方便 import infer
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'runtime'))

import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string

from infer import LitePoseEstimator, export_obj

# ---------------------------------------------------------------- 可选渲染
try:
    import trimesh
    import pyrender
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # 无头环境优先 EGL
    RENDER_OK = True
except Exception as e:
    RENDER_OK = False
    print("[webui] rendering disabled:", e)

app = Flask(__name__)
est = LitePoseEstimator()          # 全局只加载一次（ONNX 会话 + SMPL-X 数据）
OUT_DIR = os.path.join(HERE, 'webui_out')
os.makedirs(OUT_DIR, exist_ok=True)

if RENDER_OK:
    # 离屏渲染器全局复用，避免每次请求重新初始化（较慢）
    _RENDERER = pyrender.OffscreenRenderer(512, 512)
    _CAM = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)


def _look_at(pos, target, up):
    """构造相机位姿矩阵（相机 -Z 指向目标，列向量为 right/up/-forward）。"""
    f = target - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, pos
    return m


# 三个正交视角：(显示名, 相机相对中心的方向, up 向量)
# 注意：人体已在上游做了 Rx(-90°) 修正，这里是标准 Y-up 世界坐标
VIEWS = [
    ('Front', (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ('Side', (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ('Top', (0.0, 1.0, 0.001), (0.0, 0.0, 1.0)),  # 加微小偏移避免视线与 up 平行
]


def render_b64(verts, faces, direction, up):
    """渲染一个视角并返回可直接嵌入 HTML 的 base64 PNG。"""
    mesh = trimesh.Trimesh(verts, faces, process=False)
    center = (verts.max(0) + verts.min(0)) / 2
    scale = float(np.max(verts.max(0) - verts.min(0)))
    pos = center + scale * 1.9 * np.array(direction, dtype=float)
    pose = _look_at(pos, center, np.array(up, dtype=float))
    # 每个视角新建 Scene（pyrender + networkx 3.x 的 remove_node 有兼容问题）
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.35] * 3)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    scene.add(_CAM, pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.5), pose=pose)
    color, _ = _RENDERER.render(scene)
    buf = io.BytesIO()
    Image.fromarray(color).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def img_b64_pil(im, max_w=420):
    """把原图缩到合适宽度并转成 base64 JPEG，控制页面体积。"""
    im = im.convert('RGB')
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- 页面模板
PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PoseEmbroider-Lite</title>
<style>
  :root { --bg:#0f1115; --card:#181b22; --line:#2a2f3a; --tx:#e7e9ee; --dim:#9aa3b2; --acc:#4da3ff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--tx); font-family: -apple-system,"Segoe UI",Roboto,"Noto Sans SC",sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 40px 24px 80px; }
  h1 { font-size: 26px; margin: 0 0 6px; }
  .badge { font-size: 12px; background:#123a66; color:#8fc3ff; border:1px solid #1e4f8a; padding:3px 8px; border-radius:999px; vertical-align:middle; margin-left:10px; }
  .sub { color: var(--dim); margin: 0 0 28px; font-size: 14px; }
  form { display:flex; gap:14px; align-items:stretch; flex-wrap:wrap; margin-bottom:30px; }
  .drop { flex:1 1 460px; border:1.5px dashed var(--line); border-radius:14px; padding:26px; text-align:center; color:var(--dim);
          cursor:pointer; transition:.15s; background:var(--card); }
  .drop:hover, .drop.over { border-color: var(--acc); color: var(--tx); }
  .drop input { display:none; }
  button { background:var(--acc); color:#04121f; font-weight:700; border:none; border-radius:12px; padding:0 34px; font-size:15px; cursor:pointer; }
  .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin-bottom:18px; }
  @media (max-width: 900px) { .grid { grid-template-columns: repeat(2, 1fr); } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px; }
  .card h3 { font-size:12px; font-weight:600; color:var(--dim); margin:0 0 8px; text-transform:uppercase; letter-spacing:.8px; }
  .card img { width:100%; border-radius:8px; display:block; background:#fff; }
  .meta { display:flex; gap:22px; flex-wrap:wrap; background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 20px; margin-bottom:18px; font-size:14px; color:var(--dim); }
  .meta b { color:var(--tx); font-weight:600; }
  .betas { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 20px; margin-bottom:18px;}
  .betas h3 { margin:0 0 12px; font-size:12px; color:var(--dim); text-transform:uppercase; letter-spacing:.8px; }
  .bars { display:flex; gap:10px; align-items:flex-end; height:90px; }
  .bar { width:34px; height:100%; display:flex; align-items:flex-end; background:#0b0d11; border-radius:6px; overflow:hidden; }
  .bar span { width:100%; background:linear-gradient(180deg,#4da3ff,#2563b8); border-radius:6px 6px 0 0; }
  .bar-labels { display:flex; gap:10px; margin-top:6px; }
  .bar-labels div { width:34px; text-align:center; font-size:10px; color:var(--dim); }
  .files { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 20px; font-size:14px; }
  .files code { background:#0b0d11; padding:2px 8px; border-radius:6px; color:#8fc3ff; word-break:break-all; }
  .err { background:#3a1212; border:1px solid #6b2020; color:#ffb4b4; padding:14px 18px; border-radius:12px; margin-bottom:20px; font-size:14px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>PoseEmbroider-Lite <span class="badge">no torch · ONNX · CPU</span></h1>
  <p class="sub">上传人物图片 → SMPL-X 姿势/体型 → 导出 OBJ 给 Blender</p>

  {% if error %}<div class="err">{{ error }}</div>{% endif %}

  <form method="post" enctype="multipart/form-data" id="f">
    <label class="drop" id="drop">
      <input type="file" name="image" accept="image/*" id="file" required>
      <span id="label">拖拽图片到这里，或点击选择</span>
    </label>
    <button type="submit">Estimate</button>
  </form>

  {% if result %}
  <div class="grid">
    <div class="card"><h3>Input</h3><img src="{{ result.orig }}"></div>
    {% if result.views %}
      {% for v in result.views %}
      <div class="card"><h3>{{ v.angle }} View</h3><img src="{{ v.img }}"></div>
      {% endfor %}
    {% else %}
      <div class="card" style="grid-column: span 3;">
        <h3>Mesh exported</h3>
        <p style="color:var(--dim); font-size:14px; line-height:1.7;">
          渲染库未安装（pyrender/trimesh），已直接导出 OBJ。<br>
          用 Blender 打开：<code>{{ result.obj_path }}</code>
        </p>
      </div>
    {% endif %}
  </div>
  <div class="meta">
    <span>总耗时 <b>{{ result.total }} ms</b></span>
    <span>ONNX 推理 <b>{{ result.t_onnx }} ms</b></span>
    <span>SMPL-X LBS <b>{{ result.t_lbs }} ms</b></span>
    {% if result.views %}<span>三视角渲染 <b>{{ result.t_render }} ms</b></span>{% endif %}
  </div>
  <div class="files">
    <div style="margin-bottom:8px; color:var(--dim); font-size:12px; text-transform:uppercase; letter-spacing:.8px;">Outputs</div>
    <div style="line-height:2;">
      OBJ: <code>{{ result.obj_path }}</code><br>
      参数: <code>{{ result.npz_path }}</code>
    </div>
  </div>
  <div class="betas" style="margin-top:18px;">
    <h3>SMPL-X body shape (betas)</h3>
    <div class="bars">
      {% for b in result.betas %}<div class="bar"><span style="height:{{ b.h }}%"></span></div>{% endfor %}
    </div>
    <div class="bar-labels">
      {% for b in result.betas %}<div>{{ b.v }}</div>{% endfor %}
    </div>
  </div>
  {% endif %}
</div>
<script>
  // ---- 拖拽上传 ----
  const form = document.getElementById('f');
  const input = document.getElementById('file');
  const label = document.getElementById('label');
  const drop = document.getElementById('drop');

  input.addEventListener('change', () => { if (input.files.length) label.textContent = input.files[0].name; });

  // 悬停到放置区时的视觉反馈 + 阻止默认行为
  ['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));

  // 关键：drop 必须 preventDefault，否则 Chrome 会直接打开图片；
  // 同时要把 DataTransfer 里的文件塞进 <input type=file>，否则表单是空的。
  drop.addEventListener('drop', ev => {
    ev.preventDefault();
    drop.classList.remove('over');
    if (ev.dataTransfer.files.length) {
      const dt = new DataTransfer();
      dt.items.add(ev.dataTransfer.files[0]);
      input.files = dt.files;
      label.textContent = input.files[0].name;
      form.submit();   // 拖入后自动开始推理
    }
  });

  // 拦住页面其他区域的默认拖放（防止浏览器跳转到图片）
  window.addEventListener('dragover', ev => ev.preventDefault());
  window.addEventListener('drop', ev => ev.preventDefault());
</script>
</body>
</html>"""


@app.route('/', methods=['GET', 'POST'])
def index():
    """GET：返回空页面；POST：处理上传图片并返回结果页。"""
    if request.method == 'POST':
        try:
            f = request.files.get('image')
            if f is None or not f.filename:
                return render_template_string(PAGE, error='请选择一张图片')
            t0 = time.perf_counter()

            # 保存上传图（每次覆盖，避免堆积）
            raw = f.read()
            tmp = os.path.join(OUT_DIR, '_upload.png')
            Image.open(io.BytesIO(raw)).convert('RGB').save(tmp)

            # 推理
            res = est.predict(tmp)
            verts = res['verts'][0]

            # 导出 OBJ 与参数（供 Blender / 下游使用）
            obj_path = os.path.join(OUT_DIR, 'smplx_mesh.obj')
            npz_path = os.path.join(OUT_DIR, 'smplx_params.npz')
            export_obj(obj_path, verts, est.body.faces)
            np.savez(npz_path, rotvec=res['rotvec'], betas=res['betas'], verts=res['verts'])

            # 可选的三视角渲染
            views = []
            t_render = 0.0
            if RENDER_OK:
                t_r0 = time.perf_counter()
                for label_, direction, up in VIEWS:
                    views.append({'angle': label_, 'img': render_b64(verts, est.body.faces, direction, up)})
                t_render = time.perf_counter() - t_r0

            # betas 条形图数据（按最大值/最小值归一化到高度百分比）
            betas = res['betas'][0]
            lo, hi = float(betas.min()), float(betas.max())
            rng = max(hi - lo, 1e-6)
            bars = [{'v': round(float(b), 2), 'h': int(8 + 92 * (float(b) - lo) / rng)} for b in betas]

            result = {
                'orig': img_b64_pil(Image.open(tmp)),
                'views': views,
                'total': int((time.perf_counter() - t0) * 1000),
                't_onnx': int(res['t_infer'] * 1000),
                't_lbs': int(res['t_lbs'] * 1000),
                't_render': int(t_render * 1000),
                'betas': bars,
                'obj_path': obj_path,
                'npz_path': npz_path,
            }
            return render_template_string(PAGE, result=result)
        except Exception as e:
            return render_template_string(PAGE, error=f'处理失败: {e}')
    return render_template_string(PAGE)


if __name__ == '__main__':
    print(f"rendering {'ENABLED' if RENDER_OK else 'DISABLED (install pyrender+trimesh to enable)'}")
    app.run(host='127.0.0.1', port=6007, threaded=True)
