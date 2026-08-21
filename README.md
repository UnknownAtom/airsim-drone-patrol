# AirSim 无人机巡航与目标检测

一个基于 **AirSim + YOLO** 的无人机仿真验证程序：自动起飞、按航点自主巡航、实时检测地面目标（行人、车辆等），以及一个简陋的GUI

## 功能

- **多航点巡航**：JSON 配置航点，支持多圈巡航、X/Y 分轴移动、速度与高度钳制
- **目标检测**：默认 VisDrone 预训练权重（行人/车辆等 11 类），兼容 YOLOv5 与 Ultralytics 两种模型格式
- **调试参数支持**：`--save-every` / `--save-ui-every` 保存原始帧与界面帧，`--debug` 输出帧率诊断等等

## 架构

```
simu.py
 ├── flight.py   飞行模块：航点/碰撞监控/飞行线程
 ├── capture.py  取图模块：相机连接重连/帧采集线程
 ├── detector.py 检测模块：模型加载/推理/检测线程
 └── ui_qt.py   界面模块：PyQt6 前端（左视频 PIL + 右面板 Widgets）
```

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

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `visdrone-yolov26l.pt` | 模型路径（Ultralytics 或 其他格式） |
| `--conf` / `--iou` | `0.35` / `0.45` | 置信度 / NMS IoU 阈值 |
| `--imgsz` | `640` | YOLO 推理尺寸 |
| `--waypoints-file` | `waypoints.json` | 航点配置文件 |
| `--loops` | `1` | 巡航圈数，`0` 表示持续巡航 |
| `--cruise-z` | `-15` | 巡航高度（NED 坐标，负值向上） |
| `--max-speed` | `2.0` | 速度上限（m/s） |
| `--save-every N` | `0` | 每 N 帧保存一张原始帧到 `captures/` |
| `--debug` | 关闭 | 输出帧号与航点诊断信息 |

## 项目结构

```
simu.py              组装入口
flight.py            航点/碰撞/飞行线程
capture.py           相机/取图线程
detector.py          YOLO 加载/推理/检测线程
ui_qt.py              PyQt6 界面（ui.py 为原纯 PIL 版）
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
