# Mushroom Readme — MuG Diffusion RTX 50 系复刻指南

记录从 GitHub 源码到在本机（RTX 5060 Laptop GPU，Blackwell / sm_120）完整运行成功的**全部改动与步骤**，供复刻参考。

## 0. 背景：为什么不能直接用官方整合包

官方整合包内嵌 **Python 3.8 + torch 1.13.1+cu116**，最高只支持到 sm_86/sm_37 等旧架构。
RTX 50 系（sm_120）需要 **PyTorch ≥ 2.7 + CUDA 12.8**，而 torch 2.7 要求 **Python ≥ 3.9**。
整合包的 Python 3.8 是 PyInstaller 内嵌的、无法升级 → 只能源码部署。

目录约定（下文均以此为准）：

```
C:\AAAyang\MugDiffusion\MugDiffusion\          ← 官方整合包根目录（保留不动，提供 model.ckpt 与 ffmpeg.exe）
├── MugDiffusion-src\                          ← 源码部署目录（本教程主体）
│   ├── .venv\                                 ← Python 3.10 虚拟环境
│   ├── models\ckpt\model.ckpt                 ← 硬链接自整合包
│   ├── start_mugdiffusion.bat                 ← 启动脚本（自建）
│   └── scripts\MinaCalc-1.0.tar.gz            ← minacalc 源码包（官方自带）
├── runtime\py310\                             ← 独立安装的 Python 3.10.11
└── runtime\mingw64\                           ← winlibs MinGW（仅编译 minacalc 用，构建完可删）
```

## 1. 获取源码

仓库：https://github.com/Keytoyze/Mug-Diffusion
国内直连 GitHub 失败时，在链接前加代理前缀下载 zip：

```
https://ghfast.top/https://github.com/Keytoyze/Mug-Diffusion/archive/refs/heads/main.zip
```

解压为 `MugDiffusion-src`（与整合包根目录平级或其子目录均可，注意下文路径）。

## 2. Python 3.10 独立安装（不动系统 PATH）

下载 [python-3.10.11-amd64.exe](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)，静默安装到指定目录：

```bat
python-3.10.11-amd64.exe /quiet InstallAllUsers=0 TargetDir=C:\AAAyang\MugDiffusion\MugDiffusion\runtime\py310 Include_pip=1
```

创建 venv：

```bat
cd /d C:\AAAyang\MugDiffusion\MugDiffusion\MugDiffusion-src
C:\AAAyang\MugDiffusion\MugDiffusion\runtime\py310\python.exe -m venv .venv
```

> 为什么是 3.10：torch 2.7 要求 ≥3.9；而 minacalc 用 MinGW 编译依赖的 `cygwinccompiler` 在 Python 3.12+ 已被移除，3.10 刚好两头兼容。

## 3. 安装依赖（关键：版本锁定）

**全部在 venv 内执行**（下文 `pip` = `.venv\Scripts\pip.exe`）。

```bat
:: ① 先升级 pip —— 旧 pip 23.0.1 有元数据大小写 bug（typing-extensions），
::    会把某些包误回退成源码构建然后失败
.venv\Scripts\python.exe -m pip install --upgrade pip

:: ② PyTorch cu128 —— 首个原生支持 sm_120/Blackwell 的稳定版
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

:: ③ 其余依赖（对应官方 requirements.txt，但直接按实测锁定版本安装，一步到位）
pip install numpy==1.26.4 pillow==10.4.0 pyyaml omegaconf tqdm einops scipy librosa audioread soundfile matplotlib packaging eyed3 requests opt-einsum
pip install pytorch_lightning==1.9.5
pip install gradio==3.50.2 fastapi==0.100.1 starlette==0.27.0 pydantic==2.4.2
pip install reamber==0.2.1
```

### 为什么锁定这些版本（每一行都是踩过的坑）

| 包 | 锁定版本 | 原因 |
|---|---|---|
| torch / tv / ts | 2.7.1 / 0.22.1 / 2.7.1 (cu128) | 首个支持 sm_120 的稳定版 |
| gradio | 3.50.2 | webui.py 是 gradio 3.x API，4.x 不兼容 |
| fastapi + starlette | 0.100.1 + 0.27.0 | gradio 3.50.2 的配套要求 |
| **pydantic** | **2.4.2** | pydantic ≥2.5 与 fastapi 0.100.1 不兼容：启动即崩 `FieldInfo has no attribute in_` |
| pytorch_lightning | 1.9.5 | 代码用了 `pytorch_lightning.utilities.distributed`，PL 2.x 已移除 |
| numpy | 1.26.4（<2） | numpy 2.x 破坏旧 API |
| pillow | 10.4.0（<11） | pillow ≥11 与 gradio 3.x 冲突 |
| reamber | **0.2.1** | webui 从 `reamber.algorithms.playField.parts` 导入 `PFDrawBpm`；1.0.0 缺该类且强制 pillow≥11 |

> minacalc（requirements.txt 最后一行 `scripts/MinaCalc-1.0.tar.gz`）**不在上面**，单独在第 5 步编译安装。

## 4. 模型文件（复用整合包）

官方整合包内的模型是 v1.0.0（1.84 GB）。用硬链接避免双份占盘：

```bat
cd /d C:\AAAyang\MugDiffusion\MugDiffusion\MugDiffusion-src
if not exist models\ckpt mkdir models\ckpt
mklink /H models\ckpt\model.ckpt ..\models\ckpt\model.ckpt
copy ..\models\ckpt\model.yaml models\ckpt\model.yaml
```

没有整合包时，按官方 README 的发布渠道下载 v1.0.0 的 `model.ckpt` + `model.yaml`（GitHub 资源同样可用 ghfast.top 前缀加速）。

## 5. 编译 minacalc（Windows 无 MSVC 环境 → MinGW 路线）

minacalc 是纯 CPython C API 的 C++ 扩展（2 个 cpp），官方 setup.py 按 MSVC 写（`/std:c++17`）。机器上没有 Visual Studio Build Tools 时，用 winlibs MinGW 编译：

### 5.1 准备 MinGW

下载 winlibs gcc 16.2 portable（带 binutils/gendef 的完整版），解压到 `runtime\mingw64\`。

### 5.2 生成 python310 的导入库（MinGW 链接 Python 扩展必需）

```bat
cd /d C:\AAAyang\MugDiffusion\MugDiffusion\runtime\mingw64\bin
gendef C:\AAAyang\MugDiffusion\MugDiffusion\runtime\py310\python310.dll
dlltool -d python310.def -l libpython310.a -D python310.dll
copy /y libpython310.a ..\lib\
```

### 5.3 解压并改造 setup.py（共 2 处改动）

解压 `scripts\MinaCalc-1.0.tar.gz` 到任意临时目录，把 setup.py 整体替换为：

```python
from distutils.core import setup, Extension

import sys

setup(
    name = 'MinaCalc',
    version = '1.0',
    description = 'Python interface for MinaCalc',
    ext_modules = [
        Extension(
            'minacalc',
            extra_compile_args=['-std=c++17'],   # 改动①：原为 '/std:c++17'（MSVC 风格）
            extra_link_args=['-static'],          # 改动②：全静态链接，否则运行时缺 libwinpthread-1.dll
            undef_macros=['NDEBUG'],
            sources=['MinaCalcModule.cpp', 'MinaCalc/MinaCalc.cpp']
        )
    ])
```

### 5.4 构建 + 安装（必须用 venv 的 Python 3.10 执行）

```bat
cd /d <解压出的 MinaCalc-1.0 目录>
set PATH=C:\AAAyang\MugDiffusion\MugDiffusion\runtime\mingw64\bin;%PATH%
C:\AAAyang\MugDiffusion\MugDiffusion\MugDiffusion-src\.venv\Scripts\python.exe setup.py build_ext --compiler=mingw32
xcopy /y build\lib*\minacalc*.pyd C:\AAAyang\MugDiffusion\MugDiffusion\MugDiffusion-src\.venv\Lib\site-packages\
```

验证：`.venv\Scripts\python.exe -c "import minacalc; print(minacalc.__file__)"`，且 `tasklist` 确认不依赖 libwinpthread-1.dll（成品为静态链接，只依赖 python310.dll / KERNEL32 / UCRT）。

> 有 MSVC Build Tools 的机器可跳过 MinGW：直接 `pip install scripts/MinaCalc-1.0.tar.gz` 原版即可（`/std:c++17` 就是给 MSVC 的）。

## 6. 源码补丁（共 6 处）

以下改动已直接打在本仓库副本上，复刻时照抄即可。

### ① torch.load 的 weights_only（torch 2.6+ 默认值变更，3 处）

torch 2.6 起 `torch.load` 默认 `weights_only=True`，加载旧 ckpt 会报 UnpicklingError。给三处 `torch.load(...)` 追加 `weights_only=False`：

- [webui.py](webui.py) `load_model_from_config()` 内：`pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)`
- [mug/firststage/autoencoder.py](mug/firststage/autoencoder.py#L47)：`sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]`
- [mug/diffusion/diffusion.py](mug/diffusion/diffusion.py#L192)：`sd = torch.load(path, map_location="cpu", weights_only=False)`

### ② Slider 传 float 崩溃（startMapping 内，seed 处理之后）

gradio 3.50.2 的 Slider 传 **float**（作者定义 `value=4.0`），下游 `range(float)` 直接崩，前端表现为 "Get Generation" 按钮触发的一排报错。在 `startMapping()` 中 seed 设置之后加：

```python
    count = int(count)
    step = int(step)
```

### ③ 空谱面防护（custom_gridify 入口）

采样结果为零物件时，`save_osu_file → gridify → timing()` 会对空列表 `time_list[0]` 抛 IndexError，用户只能看到一串无详情的报错。在 [webui.py](webui.py) `custom_gridify()` 开头加：

```python
            def custom_gridify(hit_objects):
                if not hit_objects:
                    raise gr.Error("生成的谱面没有音符 (The generated beatmap has no notes). "
                                   "请更换音频或调整参数后重试，例如增大 Sampling step。"
                                   "(Please try a different audio or adjust parameters, "
                                   "e.g. increase Sampling step, then retry.)")
                new_hit_objects, bpm, offset = gridify(hit_objects, verbose=False)
```

### ④ 预览渲染失败不炸任务（PlayField 块包 try/except）

谱面文件已保存后，预览图渲染失败不应导致整个任务报错：

```python
                # reamber generate example
                try:
                    m = OsuMap.read_file(file_path)
                    pf = (
                            PlayField(m=m, duration_per_px=5, padding=40) +
                            PFDrawBpm() +
                            PFDrawBeatLines() +
                            PFDrawColumnLines() +
                            PFDrawNotes() +
                            PFDrawOffsets()
                    )
                    originalHeight = pf.export().height
                    processedHeight = getHeight(originalHeight, float(3.3))
                    pic = pf.export_fold(max_height=processedHeight)
                    previews.append(pic)
                except Exception:
                    # The .osz is already saved; a failed preview should not fail the whole job.
                    pass
```

### ⑤ .osz 封面用音频内嵌封面（替代默认占位图）

在 ffmpeg 转 mp3 的代码块之后、样本循环之前，插入封面提取（从**原始音频**提取，失败回退默认图）：

```python
            # use the embedded cover art of the original audio as the background, if any
            bg_path = os.path.join(save_dir, "bg.jpg")
            try:
                cover_proc = subprocess.run(
                    ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                     '-i', audioPath, '-map', '0:v:0', '-frames:v', '1', bg_path],
                    capture_output=True, text=True)
                cover_ok = cover_proc.returncode == 0 and os.path.isfile(bg_path)
            except Exception:
                cover_ok = False
            if not cover_ok:
                shutil.copyfile("asset/bg.jpg", bg_path)
```

同时**删除**样本循环内的旧复制语句 `shutil.copyfile("asset/bg.jpg", os.path.join(save_dir, "bg.jpg"))`，否则会用默认图覆盖提取结果。

### ⑥ 启动脚本 `start_mugdiffusion.bat`

放在 `MugDiffusion-src` 根目录，双击启动（把整合包根目录加进 PATH 以复用其 `ffmpeg.exe`）：

```bat
@echo off
rem MuG Diffusion launcher (GPU: PyTorch 2.7.1 + CUDA 12.8, supports RTX 5060 sm_120)
cd /d "%~dp0"
set "PATH=%~dp0..;%PATH%"
".venv\Scripts\python.exe" webui.py
pause
```

没有整合包时，去掉第 4 行并自行安装 ffmpeg 进 PATH。

## 7. 启动与验证

双击 `start_mugdiffusion.bat`，控制台正常应输出：

- `Loading model from models/ckpt/model.ckpt`（约十几秒）
- **不出现** `WARNING: CUDA GPU is not available. Fallback to CPU mode`（出现即说明 GPU 不可用）
- `Running on local URL: http://127.0.0.1:7860`

浏览器打开 http://127.0.0.1:7860 ，传音频、填标题/歌手、点 **Get Generation**，
产物在 `MugDiffusion-src\outputs\beatmaps\<歌手 - 标题>\` 与同名 `.osz`。

## 8. 踩坑速查表

| 症状 | 原因 | 解法 |
|---|---|---|
| `sm_120 is not compatible with the current PyTorch installation` | torch < 2.7 不含 Blackwell 架构 | 第 3 步 ② 的 cu128 版本 |
| Blocks 创建即崩 `FieldInfo has no attribute in_` | pydantic ≥2.5 搭 fastapi 0.100.1 | `pydantic==2.4.2` |
| `AttributeError: pytorch_lightning.utilities.distributed` | 装了 PL 2.x | `pytorch_lightning==1.9.5` |
| 加载 ckpt 报 UnpicklingError / weights_only | torch 2.6+ 默认 `weights_only=True` | 补丁① |
| 点 Get Generation 一排报错无详情 | Slider 传 float → `range(float)` 崩 | 补丁②（日志里是 `TypeError: 'float' object cannot be interpreted as an integer`） |
| `ImportError: DLL load failed` (minacalc) | MinGW 未静态链接 | setup.py 加 `extra_link_args=['-static']` 重编 |
| pip 误回退源码构建/元数据错误 | 旧 pip 23.0.1 大小写 bug | 先 `pip install --upgrade pip` |
| 预览渲染报错但谱面已生成 | reamber/pillow 版本漂移 | 锁 `reamber==0.2.1` + `pillow<11`（补丁④兜底） |
| 提示"生成的谱面没有音符" | 音频缺有效节奏信号 / Sampling step 过低 | 换正常音乐、增大 Sampling step（默认 100） |
| gradio_client 报 `Incorrect padding` | **gradio_client 0.6.1 自身的 base64 解析缺陷**，仅 Python 客户端受影响 | 浏览器 UI 不受影响，忽略；看服务端产物 |
| 沙箱/容器内报 No CUDA GPUs are available 但 nvidia-smi 正常 | 沙箱阻止 CUDA 设备访问（cuInit 返回 100） | 在沙箱外正常运行即可，非环境问题 |
