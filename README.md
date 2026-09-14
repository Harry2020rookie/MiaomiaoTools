# 妙妙工具

本分支包含 macOS 手动截图适配、自动层数识别、实托邦识别与节点预测优化。
应用包不随源码提交；按下方说明构建后，双击 `outputs/妙妙工具-实托邦.app` 使用。
安装、构建和使用说明见 [MACOS.md](MACOS.md)。以下保留上游 Windows 版说明。

妙妙工具是一款面向《明日方舟》集成战略“黑流树海”主题的 Windows 桌面辅助工具。它可以从完整游戏截图中恢复节点与连线，按地图规则推断未揭示节点的候选类型，并提供完全独立的手动编辑页面。

程序只读取用户主动导入的图片或桌面截图，不读取游戏内存、不注入游戏进程，也不会自动操作游戏。

## 主要功能

- 支持第 1-5 层地图以及第五层特殊 `5x9` 基底。
- 上传截图识别与 Windows 实时截图识别。
- 根据地图区域和相对坐标适配不同截图分辨率、长宽比。
- 自动识别节点、连线、理想源、流窜居民等地图信息。
- 根据 BFS 距离、节点类别、生成数量和固定规则计算候选节点。
- 自动识别页与完全独立的手动编辑页。
- 节点标注、拖拽、删除、移动、撤销与恢复。
- 候选筛选、高亮预览、右键快速编辑。
- 明暗主题和本地地图背景。

## 运行环境

- Windows 10/11 64 位
- Python 3.11 或更高版本

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

也可以在依赖安装完成后双击 `run.bat`。

## 使用方法

1. 选择当前层数。
2. 点击“上传并识别截图”，或在 Windows 桌面版中使用实时截图。
3. 检查恢复出的地图、节点类型和候选结果。
4. 点击或右键节点查看候选；需要时使用校正工具修改节点。
5. 无法自动识别的设备可以切换到手动编辑页，选择基底后完成标注。

识别时应尽量使用包含完整地图的原始截图。程序会先定位地图区域，再用地图内相对坐标完成节点识别；裁剪过度、遮挡严重或游戏更新导致图标变化时，仍可能需要人工校正。

## 测试

测试依赖和运行命令：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`data/screen` 和 `data/scenario*.png` 是回归测试样本，不会被 Windows 构建脚本打包进 EXE。运行程序所需的图标、规则和模板位于 `data/icons`、`data/rules`、`data/templates` 和 `data/template`。

## 构建 Windows EXE

```powershell
python -m pip install -r requirements-dev.txt
.\build_windows.bat
```

输出文件为：

```text
dist/妙妙工具.exe
```

## 项目结构

```text
.
├─ app.py                  # 桌面程序入口
├─ blackstream/            # 地图状态、画布、识别编排和主窗口
├─ vision/                 # OpenCV 图像识别
├─ models/                 # 网格、模板和规则模型
├─ rules/                  # 候选过滤和规则加载
├─ data/
│  ├─ icons/               # 节点图标
│  ├─ rules/               # 节点生成规则
│  ├─ templates/           # 地图拓扑 JSON
│  ├─ template/            # 手动编辑页基底预览
│  └─ screen/              # 回归测试截图
└─ tests/                  # 自动化回归测试
```

## Android 状态

当前仓库提供 Windows 桌面版源码，尚未包含可安装 Android APK。识别核心已经支持从图片字节流读取相册图片，并处理 EXIF 方向信息，为后续 Android 相册导入版本提供基础；Android 界面和本地 OpenCV 运行环境仍需单独实现。

## 免责声明

本项目为非官方玩家工具，与鹰角网络、Hypergryph 或《明日方舟》官方无隶属或合作关系。游戏名称、截图、图标及其他游戏素材的权利归其各自权利人所有。本工具仅用于学习、研究和个人辅助用途。

发布者应在公开仓库前阅读 [NOTICE.md](NOTICE.md)，并根据实际授权情况选择合适的开源许可证。
