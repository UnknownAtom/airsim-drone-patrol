# AirSim + YOLO 无人机巡航项目：AI 协作说明

本文件不是普通用户教程，而是提供给后续 AI 编程助手的项目上下文。任何 AI 在修改代码前，都应先阅读本文件，并以当前目录中的实际代码为准；历史对话、旧代码片段和旧截图不能覆盖现有实现。

## 1. 项目定位

这是一个基于 AirSim 和 YOLO 的无人机仿真验证程序，目标是：

- 在 AirSim 多旋翼场景中自动起飞；
- 按 JSON 航点执行巡航；
- 持续读取前视 Scene 相机画面；
- 使用 YOLO 在线检测行人、车辆等目标；
- 在 PyQt6 窗口中显示检测框、类别、置信度、帧号和源分辨率。

当前优先级是“稳定取图、实时检测、窗口不卡死、航点飞行安全”。CSV/Excel 自动导出暂时不启用。

## 2. 当前运行状态

当前主程序是 `simu.py`（组装入口）。飞行/取图/检测逻辑分别在 `flight.py`、`capture.py`、`detector.py` 中，界面在 `ui_qt.py`（PyQt6 前端，左侧 PIL 视频区 + 右侧 Qt Widgets 面板，自包含）。整体包含：

- AirSim 多航点巡航；
- 起飞、爬升、航点移动、悬停和降落；
- 独立相机采集线程；
- 独立 YOLO 检测线程；
- PyQt6 GUI 主线程显示，视频画面内部使用 PIL 绘制；
- 只保留最新帧，避免检测队列堆积造成“画面停在第一帧”；
- AirSim 相机连接失败后的重试和重连；
- 碰撞状态基线与起飞初期 grace period；
- 可选保存原始相机帧用于排查取图问题。
- 检测结果会校验对应帧是否进入显示历史，拒绝未来帧和过期帧的错误叠加；
- 飞行线程发布 `DISCONNECTED/CONNECTING/READY/TAKING_OFF/CRUISING/STOPPING/LANDING/ERROR/STOPPED` 状态；
- 界面显示采集 FPS、推理 FPS、推理耗时、检测延迟和丢帧数量；
- 启动时校验 AirSim 端口、超时、模型阈值、图像尺寸、巡航速度等参数。

当前不应默认恢复：

- CSV/Excel 自动导出；
- 手动键盘飞控；
- 把相机采集放回飞控线程；
- 将原始相机帧提前缩放后再送给 YOLO。

## 3. 文件结构

| 文件 | 作用 |
|---|---|
| `simu.py` | 组装入口：参数解析、线程协调、GUI 主循环（原单文件拆分后的主程序） |
| `flight.py` | 飞行模块：航点加载/钳制、碰撞监控、飞行线程 |
| `capture.py` | 取图模块：相机连接/重连、帧采集线程、最新帧入队 |
| `detector.py` | 检测模块：YOLO 模型加载/推理、检测线程、检测快照 |
| `ui_qt.py` | PyQt6 前端：左侧 PIL 视频区（检测框/HUD）+ 右侧 Qt Widgets 面板（信息卡片/进度条/航点路线/按钮/运行指标） |
| `settings.json` | AirSim 用户配置，建议明确设置 `SimMode: Multirotor` 和相机分辨率 |
| `waypoints.json` | 当前巡航航点配置 |
| `visdrone-yolov26l.pt` | 默认主模型（Ultralytics 格式，VisDrone 10 类 + others） |
| `yolov5s-visdrone.pt` | 旧版 YOLOv5 VisDrone 权重（备用，Torch Hub 加载） |
| `settings_airsimnh_hd.json` | 1280×720 相机配置示例 |
| `requirements.txt` | Python 依赖列表 |


> 界面字体使用系统字体（微软雅黑/Bahnschrift），不依赖项目内字体文件。

前端窗口以某个本地 SCD 风格界面为视觉参考（路径从略），但不修改该参考目录。实际修改目标始终是当前 AirSim 项目目录。界面采用 Material Design 3 蓝色系：近白 surface 背景、primaryContainer 淡蓝视频区、surfaceContainer 层次卡片、`#0061A4` 主色胶囊按钮、error 红语义色。

右侧面板为 MD3 卡片分组布局（自上而下）：状态卡（状态大标题 + 副标题）、飞行数据卡（相机/目标、航点/圈数、高度/速度、运行指标、丢帧/延迟共 5 行）、巡航进度卡（4px 线性进度条 + 圈数/进度文本 + 航点路线图）、目标卡（固定高度滚动列表，显示全部检测目标）、操作卡（胶囊按钮：多航点巡航/实时视觉检测/停止任务）。所有界面文字使用同一套中文兼容字体绘制，中文、数字和英文不再按字符切换字体。窗口较窄或较矮时自动切换为底部紧凑状态栏。相机原始画面保持不变。

## 4. 运行环境

用户使用 Windows、PowerShell、VS Code。当前曾使用过：

- Python 3.14；
- NVIDIA GeForce RTX 4060 Laptop GPU；
- CUDA 12.6 对应的 PyTorch；
- AirSim；
- OpenCV、NumPy、Pillow、Ultralytics、PyQt6。

安装依赖：

```powershell
cd <项目目录>
python -m pip install -r requirements.txt
```

AirSim 未启动时脚本依然可以正常启动：飞行线程会持续等待连接并显示“等待 AirSim 信号”，按 Q 可退出；启动场景后自动接管继续任务。默认启动命令：

```powershell
cd <项目目录>
python simu.py --device 0
```

调试模式：

```powershell
python simu.py --device 0 --debug
```

CPU 或低性能测试：

```powershell
python simu.py --device cpu --imgsz 320 --poll-interval 0.2
```

指定 AirSim RPC 端点：

```powershell
python simu.py --device 0 --airsim-ip 127.0.0.1 --airsim-port 41451 --airsim-timeout 5
```

飞行线程和相机线程共用上述 RPC 端点。每次连接失败都会新建客户端，避免复用已经超时的 msgpack 会话。连接诊断会显示在终端和右侧状态区，包括端口不可达、RPC 超时和场景尚未就绪等情况。

按 `Q` 可请求安全停止。程序结束时会取消移动任务、降落、解除解锁和 API 控制，并打印运行摘要。

## 5. 模型加载规则

默认模型是 `visdrone-yolov26l.pt`，这是 Ultralytics 格式权重，直接用 `ultralytics.YOLO()` 加载。旧模型 `yolov5s-visdrone.pt` 是旧版 YOLOv5 权重，不能直接用新版 `ultralytics.YOLO()` 当作普通 Ultralytics 模型加载。

`simu.py` 当前按模型文件名适配：

- 文件名包含 `yolov5`：使用 YOLOv5 Torch Hub 加载；
- 其他 `.pt` 文件，例如 `visdrone-yolov26l.pt`：使用 Ultralytics YOLO 加载。

因此不要随意把旧权重改成：

```python
YOLO("yolov5s-visdrone.pt")
```

切换回旧模型：

```powershell
python simu.py --model yolov5s-visdrone.pt --device 0
```

设备参数约定：

- `--device 0`：GPU 0；
- `--device cuda:0`：GPU 0；
- `--device cpu`：CPU；
- 不写 `--device`：自动优先使用 CUDA。

`--imgsz` 是 YOLO 推理尺寸，不是 AirSim 相机采集尺寸。默认推理尺寸为 640；不要为了提高速度而把 AirSim 原始帧改成 256×144 后再推理。

VisDrone 权重常见类别为：

```text
pedestrian, people, bicycle, car, van,
truck, tricycle, awning-tricycle, bus, motor
```

## 6. AirSim 相机配置与图像质量

项目提供的 `settings_airsimnh_hd.json` 将默认相机配置为 1280×720、90° FOV，并显式指定 `ApiServerPort: 41451`。AirSim 实际读取的是用户文档目录下的：

```text
%USERPROFILE%\Documents\AirSim\settings.json
```

需要时，将示例配置内容合并或复制到该位置，然后重启 AirSim。修改设置后，必须观察视频区 HUD 中的源分辨率，确认不再是 `256x144`。

图像处理的关键约束：

1. `simGetImages()` 获取 Scene 图像；
2. 原始数组直接送入 YOLO，保持当前已验证的通路；
3. 仅在 GUI 显示阶段 resize 到窗口大小；
4. 检测框按照源图像到显示窗口的比例映射。

如果源分辨率仍是 256×144，优先检查 AirSim 的 `settings.json`、相机名称和场景是否重启，不要只放大 GUI 窗口。

## 7. AirSim 连接规则

程序默认连接 `127.0.0.1:41451`。`--airsim-ip`、`--airsim-port` 和 `--airsim-timeout` 会同时作用于飞行客户端和相机客户端。

不要同时启动 AirSimNH 和 CityEnviron 并让它们都使用默认的 `41451` 端口。端口处于 LISTEN 状态不代表 AirSim RPC 已经可用；如果 `ping()` 超时，先关闭另一个仿真器并重启目标场景，再检查端口。

建议在 `%USERPROFILE%\Documents\AirSim\settings.json` 中明确指定多旋翼模式，避免启动时弹出车辆选择对话框：

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "CameraDefaults": {
    "CaptureSettings": [
      {"ImageType": 0, "Width": 1280, "Height": 720, "FOV_Degrees": 90}
    ]
  }
}
```

如果界面持续显示“等待 AirSim 信号”，先用同一 Python 环境执行：

```powershell
python -c "import airsim; c=airsim.MultirotorClient(ip='127.0.0.1', port=41451, timeout_value=5); print(c.ping())"
```

如果这个探测也出现 `TimeoutError`，问题在 AirSim RPC 服务或端口冲突，不在 YOLO 和 GUI。

## 8. 线程架构

当前程序必须保持以下职责分离：

```text
AirSim 飞行线程
    └─ 起飞、移动、航点判断、碰撞检查、降落

AirSim 相机线程
    └─ simGetImages → 最新原始帧 → 检测队列 + 显示队列

YOLO 检测线程
    └─ 取最新帧 → 推理 → 最新检测结果

PyQt6 GUI 主线程
    └─ 取最新显示帧 → PIL 绘制 → QLabel / QApplication.processEvents
```

队列大小刻意设为 1，并且新帧会覆盖旧帧。这是为了降低延迟：实时画面宁可丢弃旧帧，也不能让检测线程处理几秒前的画面。

禁止把 `simGetImages()`、YOLO 推理或 AirSim RPC 长调用重新放进 GUI 主线程，否则窗口可能再次卡死。也不要在到达航点后直接对一个尚未结束的移动任务调用阻塞式 `hoverAsync().join()`；当前实现会先 `cancelLastTask()`。

## 9. 航点和坐标规则

AirSim 使用 NED 坐标系：

- `x`、`y`：水平位置，单位为米；
- `z < 0`：向上；
- `z = -15`：约 15 米高度；
- `z > 0`：低于起飞参考面，容易造成误解或撞地。

当前 `waypoints.json` 是约 30 m × 24 m 的往复式扫描路线，共 10 个航点：

- `x` 范围：8～32；
- `y` 范围：0～24；
- `z`：-15；
- 速度：1.5 m/s；
- 容差：1.5 m。

航点格式：

```json
[
  {"x": 8, "y": 0, "z": -15, "speed": 1.5, "tolerance": 1.5},
  {"x": 32, "y": 0, "z": -15, "speed": 1.5, "tolerance": 1.5}
]
```

修改路线时必须同时检查：

- 场景中的建筑、墙体、电线和树木位置；
- 起飞点到第一个航点的连线；
- 高度是否足够；
- `--max-speed` 和 `--waypoint-timeout` 是否匹配路线长度；
- 是否需要保留默认的 X/Y 分轴飞行。

程序会将航点高度限制为不低于 `--cruise-z` 的安全高度，并将速度限制为 `--max-speed`。默认巡航高度为 `-15`，最大速度为 `2.0 m/s`。

## 10. 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `visdrone-yolov26l.pt` | 模型路径 |
| `--conf` | `0.35` | 置信度阈值 |
| `--iou` | `0.45` | NMS IoU 阈值 |
| `--imgsz` | `640` | YOLO 推理尺寸 |
| `--camera` | `0` | AirSim 相机名称 |
| `--loops` | `1` | 巡航圈数，`0` 表示持续巡航 |
| `--poll-interval` | `0.10` | 飞行遥测轮询间隔 |
| `--capture-fps` | `25` | 相机目标采集帧率；实际值受 `simGetImages` RPC 耗时限制 |
| `--cruise-z` | `-15` | 巡航高度，NED 坐标 |
| `--max-speed` | `2.0` | 速度上限 |
| `--no-axis-split` | 关闭 | 关闭 X 后 Y 的分轴移动 |
| `--no-display` | 关闭 | 不打开可视化窗口 |
| `--save-every N` | `0` | 每 N 帧保存一张 PNG，0 表示关闭 |
| `--debug` | 关闭 | 输出帧号和航点调试信息 |

保存取图验证帧：

```powershell
python simu.py --device 0 --save-every 30
```

保存的文件位于 `captures/`。该功能只是原始帧排查，不是 CSV/Excel 导出。

## 11. 排查顺序

遇到“只有第一帧”“窗口卡死”时，按下面顺序检查：

1. 视频区 HUD 的帧号是否持续递增；
2. 视频区 HUD 的源分辨率是否为期望的 1280×720，而不是 256×144；
3. `--debug` 输出中 `capture frame_id` 和 `detect frame_id` 是否递增；
4. 是否出现 `[CAMERA]`、`[DETECTOR]` 或 `[FLIGHT]` 错误；
5. 用 `--save-every 30` 检查 `captures/` 中的图片；
6. 确认 AirSim 场景没有关闭，且 `%USERPROFILE%\Documents\AirSim\settings.json` 已生效；
7. 确认模型加载方式和 `--device` 参数正确。

遇到“识别不到行人”时，先区分：

- 如果 HUD 显示的源分辨率仍为 256×144：先修相机配置；
- 如果源图像清晰但框很少：再调整 `--conf`、`--imgsz`、模型或相机视角；
- 如果只有框但类别乱码：检查终端编码和模型类别名称，不要先改检测逻辑。

遇到“撞墙”时，先减小速度、提高高度、缩短航点间距并检查路线，不要通过取消碰撞监控来掩盖路线问题。默认发生新碰撞后会停止；只有用户明确要求时才使用 `--continue-after-collision`。

## 12. AI 修改代码的硬性规则

任何代码修改都应遵守：

1. 先读取当前 `simu.py` 和相关配置，再提出修改；不要按旧对话中的版本猜测代码结构。
2. 保留飞行、取图、检测、GUI 四个职责的线程隔离。
3. 保留“只处理最新帧”的队列策略，不要改成无限队列。
4. 保留 AirSim 原始分辨率送入 YOLO 的流程；GUI resize 只能发生在显示阶段。
5. 不要把旧版 `yolov5s-visdrone.pt` 直接改成新版 Ultralytics 加载方式。
6. 不要擅自恢复 CSV/Excel 自动导出，也不要为了导出而增加后台写盘任务。
7. 修改航点前检查 NED 的负 Z 高度、场景障碍物和起飞路径。
8. 不要在 GUI 线程调用可能阻塞的 AirSim RPC 或 `join()`。
9. 修改后至少执行语法检查；涉及取图、模型或线程时，还要说明实际验证范围。
10. 保留现有模型文件和 AirSim 配置示例，不要删除或覆盖用户文件。

推荐的最小验证：

```powershell
python -m py_compile simu.py flight.py capture.py detector.py ui_qt.py
```

如果改动涉及完整依赖，再运行：

```powershell
python -m pip install -r requirements.txt
python simu.py --device 0 --debug
```

## 13. 暂停使用的功能与后续方向

CSV/Excel 自动导出已经按用户要求暂时砍掉。`requirements.txt` 中仍保留 `pandas` 和 `openpyxl`，它们是历史依赖，不能据此判断导出功能已经启用。

后续如果用户明确恢复数据导出，建议先定义字段和写盘频率，再单独增加不会阻塞相机、检测和 GUI 的导出线程。可考虑的字段包括帧号、时间戳、巡航圈数、航点编号、类别、置信度、边界框坐标和源图像分辨率。

其他可扩展方向：

- 在 GUI 中增加碰撞状态显示；
- 增加相机姿态和 FOV 的可视化配置；
- 为 CityEnviron 单独规划安全航线；
- 增加视频录制或按事件保存图像；

扩展前先确认不会破坏当前已经验证的实时取图和四线程结构。
