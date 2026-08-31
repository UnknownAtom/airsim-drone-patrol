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

当前主程序是 `simu.py`（组装入口）。飞行/取图/检测逻辑分别在 `flight.py`、`capture.py`、`detector.py` 中，界面在 `ui_qt.py`（自包含的 PyQt6 Widgets 前端，左侧 PIL 视频区 + 右侧任务控制面板）。整体包含：

- AirSim 多航点巡航；
- 起飞、爬升、航点移动、悬停和降落；
- 独立相机采集线程；
- 独立 YOLO 检测线程；
- PyQt6 GUI 主线程显示，视频画面内部使用 PIL 绘制；
- 只保留最新帧，避免检测队列堆积造成“画面停在第一帧”；
- AirSim 相机连接失败后的重试和重连；
- 飞行中 RPC 断线后的重新连接、重新确认场景和按当前位置恢复当前航点；
- 启动阶段 `reset()` 或 API 控制失败也会进入重连流程；重连后会检查无人机是否已落地，必要时重新起飞；
- 起飞和降落使用状态轮询，停止/Q 键不再依赖无限等待的异步任务 `join()`；
- 碰撞状态基线与起飞初期 grace period；
- 可选保存原始相机帧用于排查取图问题。
- 检测结果会校验对应帧是否进入显示历史，拒绝未来帧和过期帧的错误叠加；
- 飞行线程发布 `DISCONNECTED/CONNECTING/READY/TAKING_OFF/CRUISING/STOPPING/LANDING/ERROR/STOPPED` 状态；
- 界面显示任务状态、采集/推理/GUI FPS、取图与 YOLO 耗时、相机/检测丢帧和当前检测目标；完整统计仍保留在终端摘要；
- 阶段三增加取图 RPC、图像解析、YOLO、GUI 渲染的滚动平均/最大耗时和最近窗口 FPS；
- 固定使用原始 Scene 采集模式；压缩 Scene 实测会降低当前本机采集性能，相关分支已移除；
- CUDA/RTX 自动使用 FP16，模型启动时预热，并记录模型加载、预热和真实首帧推理耗时；
- GUI 默认限制为 18 FPS，采集线程仍可独立按 25 FPS 调度；无新画面时跳过 PIL 重绘；
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
| `frame_stream.py` | 中立帧消息与单槽最新帧通道；采集向检测和 GUI 各发布一份最新帧 |
| `airsim_connection.py` | AirSim 客户端创建、独立会话关闭和连接生命周期辅助 |
| `performance.py` | 线程安全滚动耗时统计和最近窗口 FPS 统计 |
| `benchmark_capture.py` | 独立相机基准工具，不加载 YOLO、不起飞，测试原始 Scene |
| `detector.py` | 检测模块：YOLO 模型加载/推理、检测线程、检测快照 |
| `ui_qt.py` | PyQt6 前端：左侧 PIL 视频区（检测框/HUD）+ 右侧任务控制面板（状态/指标/进度/目标/按钮） |
| `settings.json` | AirSim 用户配置，建议明确设置 `SimMode: Multirotor` 和相机分辨率 |
| `waypoints.json` | 当前巡航航点配置 |
| `visdrone-yolov26l.pt` | 默认主模型（Ultralytics 格式，VisDrone 10 类 + others） |
| `yolov5s-visdrone.pt` | 旧版 YOLOv5 VisDrone 权重（备用，Torch Hub 加载） |
| `settings_airsimnh_hd.json` | 1280×720 相机配置示例 |
| `requirements.txt` | Python 依赖列表 |


> 界面字体使用系统字体（微软雅黑/Bahnschrift），不依赖项目内字体文件。`--theme` 参数目前保留兼容性，界面统一使用浅色主题。

前端不再依赖外部参考界面。当前 GUI 采用浅色大圆角扁平风格：纯色背景、白色卡片、浅灰指标块、蓝色状态、红色停止操作；不使用渐变。左侧为实时前视画面，右侧为任务状态、核心性能指标、巡航进度、检测目标和停止操作。

右侧面板自上而下包含：状态卡（状态和相机提示）、性能指标 2×3 网格（采集 FPS、取图耗时、推理 FPS、YOLO 耗时、GUI FPS、相机/检测丢帧）、巡航进度卡（进度条 + 圈数/航点文本 + 航点路线图）、检测目标卡（固定高度滚动列表）和操作卡（停止任务）。AirSim 就绪后自动开始巡航，停止任务会安全降落并结束本次任务。当前不再使用独立的“紧凑模式”或底部状态栏；右侧面板始终是主控制区域。窗口初始尺寸默认 `1600×900`，由 `--display-width` 和 `--display-height` 控制。

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

GUI 初始窗口默认为 `1600×900`，可按屏幕尺寸调整：

```powershell
python simu.py --device 0 --display-width 1280 --display-height 720 --display-fps 18
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

CUDA 设备会自动使用 FP16；`--half` 可显式表达这一意图，CPU 会自动回退 FP32。模型加载后会用代表性的 1280×720 空帧预热一次，预热不计入实时检测 FPS；结束摘要会同时打印模型加载时间、预热时间和真实首帧推理时间。可用 `--imgsz 416` 降低延迟，`--imgsz 640` 保持默认检测效果，`--imgsz 768` 适合更重视小目标的测试。

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
2. 默认原始数组直接送入 YOLO，保持当前已验证的通路；
3. 原始数组直接送入 YOLO，GUI resize 只发生在显示阶段；
4. AirSim 源图像保持至少 1280×720；
5. 检测框按照源图像到显示窗口的比例映射。

如果源分辨率仍是 256×144，优先检查 AirSim 的 `settings.json`、相机名称和场景是否重启，不要只放大 GUI 窗口。

## 7. AirSim 连接规则

程序默认连接 `127.0.0.1:41451`。`--airsim-ip`、`--airsim-port` 和 `--airsim-timeout` 会同时作用于飞行客户端和相机客户端。

不要同时启动 AirSimNH 和 CityEnviron 并让它们都使用默认的 `41451` 端口。端口处于 LISTEN 状态不代表 AirSim RPC 已经可用；如果 `ping()` 超时，先关闭另一个仿真器并重启目标场景，再检查端口。飞行和相机线程各自创建客户端，但统一使用 `airsim_connection.py` 的端点配置和释放逻辑。

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
    └─ simGetImages → FramePacket → 独立的检测/显示最新帧通道

YOLO 检测线程
    └─ 取最新帧 → 推理 → 最新检测结果

PyQt6 GUI 主线程
    └─ 取最新显示帧 → 限频 PIL 绘制 → QLabel / 限频 QApplication.processEvents
```

队列大小刻意设为 1，并且新帧会覆盖旧帧。这是为了降低延迟：实时画面宁可丢弃旧帧，也不能让检测线程处理几秒前的画面。
`capture.py` 不导入检测或 GUI 模块；通道类型定义在 `frame_stream.py`，由 `simu.py` 在组装时连接生产者与消费者。

阶段三的性能口径：

- `simGetImages` 平均/最大耗时：只统计 AirSim RPC 阶段；
- 图像解析平均/最大耗时：统计原始字节流到 NumPy 数组的解析；
- 取图 FPS：最近约 3 秒的成功采集帧率，同时保留总平均 FPS；
- YOLO 平均耗时：最近 120 次真实推理样本；最大耗时：本次运行累计最大值；
- 检测结果延迟：从帧提交检测队列到推理完成；
- GUI 渲染平均/最大耗时：PIL 画布、检测框、HUD 和 QPixmap 更新；
- 相机/检测丢帧：最新帧覆盖队列的累计丢弃数量。

默认 GUI `--display-fps 18`，不会降低相机采集或 YOLO 队列提交频率；如果只做采集基准，可使用 `--no-display`，并对比结束摘要中的分项统计。

若只测试 AirSim 原始取图链路，使用独立基准工具，避免模型加载和飞行路径影响结果：

```powershell
python benchmark_capture.py --frames 120
```

禁止把 `simGetImages()`、YOLO 推理或 AirSim RPC 长调用重新放进 GUI 主线程，否则窗口可能再次卡死。飞行线程中的起飞、爬升和降落使用状态轮询与超时，不依赖阻塞式 `join()`；到达航点后会先 `cancelLastTask()` 再发送悬停。

飞行中的连续 RPC 失败会触发安全恢复：关闭失效会话、重新 `ping()`、确认场景状态、重新取得 API 控制和解锁状态，然后按当前无人机位置重新计算当前航点。恢复流程不会调用 `reset()`，也不会把无人机传送回起点。若重连失败，任务进入 `ERROR` 并执行有界的降落/释放流程。

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
| `--display-fps` | `18` | GUI 最大渲染帧率，不限制采集线程 |
| `--display-width` / `--display-height` | `1600` / `900` | GUI 初始窗口尺寸 |
| `--theme` | `light` | 保留兼容参数；当前界面统一使用浅色主题 |
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
python -m py_compile simu.py flight.py capture.py detector.py ui_qt.py performance.py benchmark_capture.py
```

如果改动涉及完整依赖，再运行：

```powershell
python -m pip install -r requirements.txt
python simu.py --device 0 --debug
```

## 13. 暂停使用的功能与后续方向

CSV/Excel 自动导出已经按用户要求暂时砍掉，当前依赖文件不再包含 `pandas` 和 `openpyxl`。

后续如果用户明确恢复数据导出，建议先定义字段和写盘频率，再单独增加不会阻塞相机、检测和 GUI 的导出线程。可考虑的字段包括帧号、时间戳、巡航圈数、航点编号、类别、置信度、边界框坐标和源图像分辨率。

其他可扩展方向：

- 在 GUI 中增加碰撞状态显示；
- 增加相机姿态和 FOV 的可视化配置；
- 为 CityEnviron 单独规划安全航线；
- 增加视频录制或按事件保存图像；

扩展前先确认不会破坏当前已经验证的实时取图和四线程结构。
