# UI 改造任务：视频区渲染从 PIL 迁移到 QPainter

> 本文件是给 AI 助手的一次性任务说明，信息自包含，不需要查看历史对话。
> 完成并验证后本文件可删除。

## 1. 背景与目标

当前 `ui_qt.py` 的左侧视频区用 **PIL（Pillow）** 绘制：每帧先创建一个 PIL 画布，
用 `ImageDraw` 画背景/检测框/HUD/文字，再转 numpy → QImage → QPixmap 显示在 QLabel 上。
这套方案的问题：

1. **双渲染体系**：视频区是 PIL 绘制，右侧面板是 Qt Widgets，视觉参数（字体、圆角）维护在两套代码里；
2. **渲染链长**：PIL 画布 → numpy → QImage → QPixmap，每帧多次全幅转换；
3. **无谓依赖**：PIL 在本项目只被 `ui_qt.py` 使用，迁移后可从 `requirements.txt` 移除。

**目标**：视频区改为一个 **`VideoWidget(QWidget)` 自绘控件**，全部用 `QPainter`
在 `paintEvent()` 里绘制（图像 + 检测框 + HUD + chips），视觉外观 **1:1 保持不变**
（浅色大圆角扁平风格），删除 `ui_qt.py` 中所有 PIL 代码。

## 2. 硬性约束（不可违反）

- **`simu.py`、`flight.py`、`capture.py`、`detector.py`、`frame_stream.py`、`performance.py` 一律不改**；
- `DetectionDisplay` 对外接口保持完全兼容，`simu.py` 才能零改动：
  - `show(packet, snapshot, args, ui) -> bool`
  - `process_events(force=False) -> bool`
  - `render_fps`（property）、`render_performance`（property）、`render_count`、`last_render`
- 保留可选依赖逻辑：`qfluentwidgets` / `pyqtgraph` 的 `try/except ImportError` 降级路径不得破坏；
- 保留 `ui_theme.py`（COLORS/TYPE/_hex/_make_font）与 `ui_components.py`（MetricTile/PerformanceChart/RouteWidget）的结构，不要把它们合并回 `ui_qt.py`；
- 工作区有**未提交改动**（`ui_qt.py`、`ui_components.py`、`ui_theme.py`、`requirements.txt`、两个 README），必须基于当前工作区内容修改，禁止 `git reset`/`git checkout`/整文件重写；
- 不引入任何新依赖（只允许从 `requirements.txt` 中删除 `pillow`）；
- 不要给视频控件加 `QGraphicsDropShadowEffect`（每帧重绘时阴影开销大）；
- Qt 的 QSS **不支持 CSS transition**，不要写 `transition:` 属性（写了会被忽略）。

## 3. 当前代码事实（`ui_qt.py`，共 938 行；行号可能因并发修改漂移，按符号名定位）

### 3.1 需要删除的 PIL 代码（全部在 `ui_qt.py`）

| 符号 | 位置 | 说明 |
|---|---|---|
| `from PIL import Image, ImageDraw, ImageFont` | 第 19 行 | 删除 |
| `UIFonts` 类 | 约 82-137 行 | PIL 字体加载器（msyh/bahnschrift），整体删除 |
| `_rgba()` | 约 140 行 | PIL 颜色辅助，删除 |
| `_rounded()` | 约 144 行 | PIL 圆角矩形，删除 |
| `_fit_image()` | 约 159 行 | 等比适配计算，逻辑迁入 `VideoWidget` |
| `_clip()` | 约 180 行 | 数值钳制，逻辑迁入 `VideoWidget` |
| `_rounded_mask()` + `@lru_cache` | 约 184 行 | 圆角遮罩缓存，删除 |
| `_paste_rounded()` | 约 195 行 | 圆角贴图，删除 |
| `_PilRenderer._text()` / `_center_text()` | 约 223/234 行 | 删除（QPainter.drawText 替代） |
| `_PilRenderer._draw_corner_bbox()` | 约 253 行 | 四角锚点检测框，逻辑迁入 `VideoWidget` |
| `_PilRenderer._chip()` | 约 272 行 | 底部胶囊，逻辑迁入 `VideoWidget` |
| `_PilRenderer._draw_preview()` | 约 297 行 | 主绘制函数，整体迁入 `VideoWidget.paintEvent` |
| `_PilRenderer._render_video()` | 约 831 行（在 DetectionDisplay 内） | 删除，渲染改由 `VideoWidget` 完成 |

### 3.2 保留不动

- `_PilRenderer._status_values()`（约 427-465 行）：只读 `ui` 字典和 `self.last_snapshot`，无 PIL 依赖，**保留**（`_PilRenderer` 基类可保留，仅删除其绘制方法；或把 `_status_values` 提升到 `DetectionDisplay`，二选一，保持行为不变即可）；
- `DISPLAY_NAME_MAP` / `_display_name()`；
- `DetectionDisplay` 的状态卡/指标卡/性能曲线/进度卡/目标卡/按钮等面板构建代码；
- `show()` 的帧率限制、检测新鲜度（8 帧历史时间窗）逻辑。

### 3.3 相关文件现状

- `ui_theme.py`：`COLORS`（浅色板，键：bg/surface/surface_sub/preview/primary/primary_soft/primary_dark/text/muted/muted_light/border/success/warning/warning_soft/white）、`TYPE`（字号表，`small`=(12,400)、`body`=(14,400) 等）、`_hex()`、`_make_font(size, weight, latin)`（**已经用 `setPixelSize`**，与像素对齐）。视频区 QPainter 文字直接用 `_make_font()`，字体体系即统一。
- `ui_components.py`：`RouteWidget`（QPainter 自绘，可参考其写法）、`MetricTile`、`PerformanceChart`。
- `requirements.txt`：包含 `pillow`，迁移完成后删除。

## 4. 目标设计：`VideoWidget(QWidget)`

建议放在 `ui_qt.py` 内（或 `ui_components.py`，二选一，倾向 `ui_qt.py` 以减少跨文件改动）。

```python
class VideoWidget(QWidget):
    """QPainter 自绘视频区：等比图像 + 四角锚点检测框 + HUD + 底部 chips。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._frame: np.ndarray | None = None   # RGB ndarray (H, W, 3)
        self._boxes: list[tuple[float, float, float, float, int, float, str]] = []
        self._frame_id = 0
        self._detection_fresh = False

    def set_frame(self, frame, boxes, frame_id, detection_fresh) -> None:
        # frame: RGB ndarray；boxes: 源图坐标 (xmin, ymin, xmax, ymax, class_id, conf, name)
        self._frame = frame
        self._boxes = list(boxes)
        self._frame_id = int(frame_id)
        self._detection_fresh = bool(detection_fresh)
        self.update()

    def paintEvent(self, event) -> None: ...
```

### 4.1 `paintEvent` 绘制顺序（严格按此顺序，视觉与现版 1:1）

设 `w, h = self.width(), self.height()`，painter 开启
`Antialiasing` 与 `TextAntialiasing` 两个 RenderHint。

1. **最外层**：`QRectF(0, 0, w, h)` 填充 `surface`（白 #FFFFFF），圆角 28，1px `border` 描边
   （对应现版 `video_label` 的 QSS：`background: surface; border-radius: 28px; border: 1px solid border`）。
2. **内层**：`QRectF(12, 12, w-24, h-24)` 填充 `preview`（#E4E8F0），圆角 20
   （对应 `_draw_preview` 的 `inner`）。
3. **空状态**（`self._frame is None`）：居中绘制文字 `"系统就绪 · 等待相机画面"`，
   字体 `_make_font(14)`，颜色 `muted`（#64748B），垂直中心向上偏移 10px。
4. **有帧**：
   a. 等比适配：`scale = min((w-24-2)/sw, (h-24-2)/sh)`，图像区域 `(px, py, pw, ph)` 居中
      （对应 `_fit_image`：内边距 12+1，圆角 18）。用 `QPainterPath.addRoundedRect(..., 18, 18)`
      `setClipPath` 后 `drawImage`（`QImage` 从 RGB ndarray 构造，`Format_RGB888`，
      必须 `.copy()` 或持有引用），再恢复 clip。
   b. **HUD 胶囊**：左上 `(16, 16)` 起，高 32，宽 = 文字宽 + 36，圆角 16，填充 `surface`；
      内左侧绿点 `success`（#16A34A）直径 6，圆心 `(29, 32)`；文字 `"前视监控  ·  {W}×{H}"`
      字体 `_make_font(12)` 颜色 `text`（#1E293B），`x=40, y=23` 基线（对应现版数值）。
   c. **检测框**（`self._detection_fresh` 且 `boxes` 非空）：每个 box 映射到显示坐标
      `left = px + xmin*scale`（同法 top/right/bottom），钳制到图像区（对应 `_clip`）。
      **四角锚点样式**：每条边只画两端各 14px 的线段（共 8 条短线），线宽 3，
      颜色 `primary`（#2563EB）当 `conf >= 0.45`，否则 `warning`（#DC2626）。
      标签：`"{中文名} {conf:.0%}"`，高 24，宽 = 文字宽 + 18，圆角 12，填充 `surface`；
      位置优先框上方（`top - 28`），越界则放框下方（`bottom + 4`），再钳制进图像区；
      文字 `_make_font(12)` 颜色同框色，`x = left + 9`。
   d. **底部 chips**：`y = h - 12 - 32`（即内层底 - 48），高 32，圆角 16，填充 `surface`：
      - 帧号 chip：`x = 28`，文字 `f"帧 {frame_id:06d}"`，颜色 `text`；
      - 目标数 chip：`x = 142`，文字 `f"目标 {count}"`，颜色 `primary`。
      chips 文字 `_make_font(12)`，垂直居中。

> 所有数值来自现有 PIL 实现（见 §3.1 与上文），迁移时以"观感一致"为最终验收标准，
> 允许 ±2px 的微调，但不得改变视觉风格（四角锚点框、无描边、大圆角、浅色）。

### 4.2 QImage 构造（关键）

```python
rgb = np.ascontiguousarray(frame)          # 确保内存连续
h, w = rgb.shape[:2]
qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()  # copy 保证安全
```
`FramePacket.frame` 是 RGB（AirSim Scene 原生 RGB），**不要做 BGR/RGB 转换**。

### 4.3 文字测量

用 `painter.fontMetrics().horizontalAdvance(text)`（先 `painter.setFont(_make_font(...))`）。
中文没问题（`_make_font` 默认微软雅黑）。

## 5. 实施步骤

1. **备份**：`git stash push -m "before-qpainter-video"` 或在别处复制 `ui_qt.py`（工作区有未提交改动，先留后路）。
2. 在 `ui_qt.py` 顶部：删除 `from PIL import ...`；`PyQt6.QtGui` import 增加
   `QColor, QFont, QPainter, QPainterPath, QPen, QRectF`（`QImage/QPixmap` 视需要保留）。
3. 新增 `VideoWidget` 类（§4 规格）。
4. 删除 §3.1 列出的全部 PIL 符号；`_fit_image`/`_clip`/`_draw_corner_bbox`/`_chip`/`_draw_preview`
   的逻辑按 §4.1 迁入 `VideoWidget`（可保留 `_fit_image`/`_clip` 为模块级函数供 VideoWidget 使用，不必强行内联）。
5. `DetectionDisplay`：
   - `_build_window()` 中 `self.video_label = QLabel(...)`（含 QSS）替换为
     `self.video_widget = VideoWidget()`（stretch=7 不变，`setStyleSheet` 不再需要）；
   - `show()` 中 `self._render_video(...)` 替换为：
     ```python
     self.video_widget.set_frame(
         self.last_packet.frame if self.last_packet is not None else None,
         boxes,                       # 新鲜且 show_detections 时取 snapshot.boxes，否则 []
         self.last_packet.frame_id if self.last_packet is not None else 0,
         self._detection_fresh,
     )
     ```
     `render_stats.add(...)` 计时范围保持覆盖 `set_frame` + `_update_panel`（量级不变即可）；
   - 删除 `_render_video()`；
   - `last_render`（`--save-ui-every` 用）：`save_render=True` 时改为
     `self.last_render = <VideoWidget.grab() 转 BGR ndarray>`：
     ```python
     pixmap = self.video_widget.grab()
     image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
     ptr = image.constBits(); ptr.setsize(image.height() * image.width() * 3)
     rgb = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.width(), 3).copy()
     self.last_render = rgb[:, :, ::-1]  # RGB -> BGR，与旧行为一致
     ```
     注意 `grab()` 仅在窗口已显示时调用（与旧版 `last_render` 语义一致：无窗口时为 None 也可接受，但不要抛异常）。
6. `requirements.txt` 删除 `pillow`（先 `grep -r "PIL\|pillow" *.py` 确认零引用）。
7. 清理 `_ui_refactor.png` 等临时截图（可选）。

## 6. 验证清单（必须全部通过）

```powershell
# 1. 编译
python -m py_compile ui_qt.py ui_theme.py ui_components.py simu.py

# 2. PIL 零残留
#    在 ui_qt.py 中不应再有 PIL/ImageDraw/ImageFont/UIFonts/_paste_rounded 等符号

# 3. 无头模式（不应创建 GUI，不应报错）
python simu.py --no-display --device 0
#    观察：模型加载、线程启动、无 Traceback；几秒后 Ctrl+C 或直接关闭

# 4. GUI 冒烟（会弹窗）
python simu.py --device 0 --imgsz 320 --save-ui-every 1 --display-fps 10
#    - 窗口正常弹出且响应（拖拽/关闭正常，Q 键可退出）；
#    - captures/ 下生成 ui_00001.png（验证 last_render 链路）；
#    - 视频区空状态显示“系统就绪 · 等待相机画面”；
#    - 若 AirSim 可用：确认视频画面、检测框（四角锚点）、HUD、底部 chips 显示正常。
```

注意：无 AirSim 时画面为空状态属正常；有 AirSim 场景时重点看：
HUD 胶囊文字、四角锚点检测框、标签、底部"帧/目标"chips 的观感与改版前一致
（可用改版前截图对比）。

## 7. 常见坑（先看再写）

- `QImage` 构造自 numpy 必须 `copy()` 或保证数组存活，否则画面花屏/崩溃；
- 帧是 **RGB**，不要 cvtColor；
- `drawImage(QRectF, QImage)` 会缩放，注意目标矩形用浮点坐标避免锯齿；
- 圆角裁剪用 `QPainterPath` + `setClipPath`，绘制后 `setClipping(False)` 恢复；
- chips/标签文字垂直居中：用 `drawText(QRectF, AlignVCenter | AlignLeft, text)` 比手算基线稳；
- `horizontalAdvance` 是 Qt6 方法（Qt5 是 `width()`），项目用 PyQt6，没问题；
- 不要用 QSS 给视频控件设背景后再叠加绘制（会双层），背景一律在 `paintEvent` 里画；
- `grab()` 在窗口未显示时会返回空图，`last_render` 置 None 即可，不要抛异常中断主循环。
