# AirSim 无人机巡航与目标检测

一个基于 **AirSim + YOLO** 的无人机仿真验证程序：自动起飞、按航点自主巡航、实时检测地面目标（行人、车辆等），以及一个简陋的GUI

## 功能

- **多航点巡航**：JSON 配置航点，支持多圈巡航、X/Y 分轴移动、速度与高度钳制
- **目标检测**：默认 VisDrone 预训练权重（行人/车辆等 11 类），兼容 YOLOv5 与 Ultralytics 两种模型格式
- **稳定性保护**：飞行状态机、RPC 连续失败计数、停止/降落状态显示和启动参数校验
- **断线恢复**：飞行中 AirSim RPC 断开后按当前位置重连并恢复当前航点；若仿真器重启导致无人机落地，会先重新起飞，不执行中途自动 reset
- **实时诊断**：显示采集/检测/GUI FPS、取图/图像解析/推理平均与最大耗时、检测延迟和丢帧数量
- **性能模式**：固定使用原始 Scene 取图；GUI 默认限频，CUDA 自动使用 FP16 并进行模型预热
- **调试参数支持**：`--save-every` / `--save-ui-every` 保存原始帧与界面帧，`--debug` 输出帧率诊断等等

## 架构

```
simu.py
 ├── flight.py   飞行模块：航点/碰撞监控/飞行线程（含断线重连）
 ├── airsim_connection.py  AirSim 客户端创建与释放辅助
 ├── capture.py  取图模块：相机连接重连/帧采集线程
 ├── detector.py 检测模块：模型加载/推理/检测线程
 ├── performance.py  性能滚动统计与 FPS 窗口
 ├── benchmark_capture.py  不启动飞行的原始 Scene 取图基准工具
 └── ui_qt.py   界面模块：PyQt6 前端（左视频 PIL + 右面板 Widgets）
```

配置文件：`waypoints.json`（巡航航点）、`settings_airsimnh_hd.json`（1280×720 相机配置示例）；详细说明见 `README_AI.md`。

## 环境

- AirSim 模拟器（AirSimNH 场景）
- Python 3.10+
- NVIDIA GPU（可选；无 GPU 时使用 `--device cpu`）

## 开始

```powershell
# 1. 安装依赖
cd <项目目录>
python -m pip install -r requirements.txt

# 2. 启动 AirSim 场景（先打开 AirSimNH，确认 RPC 服务可用）

# 3. 运行巡航程序
python simu.py --device 0
```

按 `Q` 键安全停止：程序会取消移动任务、降落、解除解锁并打印运行摘要。

连接异常时，程序会在后台持续重试，并在界面中显示具体 RPC 错误。AirSimNH 和 CityEnviron 不要同时使用默认的 `41451` 端口；运行目标场景时只保留一个仿真器，或为不同场景配置不同端口。

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `visdrone-yolov26l.pt` | 模型路径（Ultralytics 或 其他格式） |
| `--conf` / `--iou` | `0.35` / `0.45` | 置信度 / NMS IoU 阈值 |
| `--imgsz` | `640` | YOLO 推理尺寸 |
| `--half` | 自动 | CUDA/RTX 默认 FP16；CPU 自动回退 FP32 |
| `--capture-fps` | `25` | 相机目标采集帧率，实际值受 AirSim RPC 延迟限制 |
| `--display-fps` | `18` | GUI 最大渲染帧率，不限制相机采集 |
| `--waypoints-file` | `waypoints.json` | 航点配置文件 |
| `--loops` | `1` | 巡航圈数，`0` 表示持续巡航 |
| `--cruise-z` | `-15` | 巡航高度（NED 坐标，负值向上） |
| `--max-speed` | `2.0` | 速度上限（m/s） |
| `--save-every N` | `0` | 每 N 帧保存一张原始帧到 `captures/` |
| `--debug` | 关闭 | 输出帧号与航点诊断信息 |
| `--airsim-ip` / `--airsim-port` | `127.0.0.1` / `41451` | AirSim RPC 地址与端口 |
| `--airsim-timeout` | `5` | 单次 AirSim RPC 超时时间（秒） |
| `--rpc-retry-limit` | `5` | 遥测连续失败上限，超过后进入断线重连流程 |

原始 Scene 取图基准可使用以下命令，观察 `simGetImages`、图像解析和实际采集 FPS：

```powershell
python simu.py --device 0 --capture-fps 25
python benchmark_capture.py --frames 120
```

项目固定使用原始 Scene。实测压缩模式在当前本机 AirSim 环境中将采集速度从约 13.7 FPS 降至 6.84 FPS，因此已移除压缩取图分支。

## 项目结构

```
simu.py              组装入口
flight.py            航点/碰撞/飞行线程（含断线重连）
airsim_connection.py AirSim 客户端创建/释放辅助
capture.py           相机/取图线程
detector.py          YOLO 加载/推理/检测线程
performance.py       滚动耗时统计与 FPS 窗口
benchmark_capture.py 取图基准工具（不加载 YOLO/不起飞）
ui_qt.py             PyQt6 界面
waypoints.json       巡航航点
settings_airsimnh_hd.json  1280×720 相机配置示例
README_AI.md         面向 AI 助手的详细项目说明
```

## 坐标系说明

AirSim 使用 NED 坐标系：`z < 0` 表示向上，默认巡航高度 `z = -15`（约 15 米）。修改航点前请检查场景障碍物与起飞路径。

## 致谢

- [AirSim](https://github.com/microsoft/AirSim) —— 微软开源无人机/汽车仿真平台
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) —— 目标检测框架
- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) —— 航拍目标检测数据集
