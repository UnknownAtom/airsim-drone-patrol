"""PyQt6 前端（自包含）：左侧 PIL 视频区 + 右侧 Qt Widgets 面板。

本模块不再依赖独立的 ``ui.py``（原纯 PIL 前端已移除）：PIL 渲染基础设施
（配色、字体、消息、预览绘制）直接内联在本模块中，仅保留 Qt 版实际用到
的部分。

- 左侧视频区：PIL 渲染检测框、HUD、chips，结果转为 ``QPixmap`` 显示在
  ``QLabel`` 上；
- 右侧面板：Qt Widgets（状态区、信息卡片、进度条、航点路线、目标区、
  操作按钮、紧凑状态条）；
- ``show(packet, snapshot, args, ui) -> bool`` 与 ``process_events()``
  签名/语义不变：返回 True 表示用户请求退出（Q 键或关闭窗口）。

视觉规范（SCD 风格）：浅灰背景 #F2F2F7、白卡片圆角 12px、深蓝主色
#324CB4、进度条淡蓝轨道 #EEF1FB、成功绿 / 警告红语义色。
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from performance import RateWindow, RollingStats, StatsSnapshot

WINDOW_TITLE = "AirSim 无人机目标检测系统"

# ---------------------------------------------------------------------------
# PIL 渲染基础设施（原 ui.py 内联，仅保留 Qt 版使用部分）
# ---------------------------------------------------------------------------

DISPLAY_NAME_MAP = {
    "pedestrian": "行人",
    "people": "人群",
    "bicycle": "自行车",
    "car": "汽车",
    "van": "面包车",
    "truck": "卡车",
    "tricycle": "三轮车",
    "awning-tricycle": "篷车",
    "bus": "公交车",
    "motor": "摩托车",
    "others": "其他",
}


def _display_name(name: str) -> str:
    return DISPLAY_NAME_MAP.get(str(name).strip().lower(), str(name))


# MD3（Material Design 3）蓝色系浅色色板。
# PIL 视频区（预览面/HUD/检测框）与 Qt 面板共用，键名保持兼容。
COLORS: dict[str, tuple[int, int, int]] = {
    "bg": (253, 252, 255),            # surface
    "surface": (255, 255, 255),       # surfaceContainerLowest
    "preview": (207, 229, 255),       # primaryContainer（视频区淡蓝）
    "primary": (0, 97, 164),          # primary（M3 Blue）
    "primary_soft": (207, 229, 255),  # primaryContainer
    "text": (25, 28, 32),             # onSurface
    "muted": (68, 71, 78),            # onSurfaceVariant
    "muted_light": (117, 119, 127),   # outline
    "border": (196, 199, 207),        # outlineVariant
    "soft_gray": (241, 244, 250),     # surfaceContainer
    "soft_blue": (214, 227, 247),     # secondaryContainer
    "surface_high": (229, 233, 240),  # surfaceContainerHighest（进度轨道）
    "success": (59, 125, 63),
    "warning": (186, 26, 26),         # error
    "warning_soft": (255, 218, 214),  # errorContainer
    "white": (255, 255, 255),
    "image_border": (255, 255, 255),
}

TYPE = {
    "title": (20, 700),
    "section": (20, 700),
    "status": (32, 700),
    "value": (16, 500),
    "card": (16, 500),
    "body": (14, 400),
    "label": (12, 500),
    "small": (12, 400),
    "button": (16, 700),
}


class UIFonts:
    """PIL 侧中文字体加载（微软雅黑优先，黑体回退）。"""

    CJK = {
        400: ("msyh.ttc", "msyh.ttf", "simhei.ttf"),
        500: ("msyh.ttc", "msyh.ttf", "simhei.ttf"),
        700: ("msyhbd.ttc", "msyhbd.ttf", "msyh.ttc"),
    }
    LATIN = {
        400: ("bahnschrift.ttf", "segoeui.ttf"),
        500: ("bahnschrift.ttf", "segoeui.ttf"),
        700: ("bahnschrift.ttf", "segoeuib.ttf"),
    }

    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.cache: dict[tuple[int, int, bool], ImageFont.FreeTypeFont] = {}

    def _load(self, size: int, weight: int, cjk: bool) -> ImageFont.FreeTypeFont:
        key = (size, weight, cjk)
        if key in self.cache:
            return self.cache[key]
        names = (self.CJK if cjk else self.LATIN).get(weight, self.LATIN[400])
        font = None
        for name in names:
            for directory in (self.base_dir / "fonts", Path("C:/Windows/Fonts")):
                path = directory / name
                if not path.exists():
                    continue
                try:
                    font = ImageFont.truetype(str(path), size)
                    break
                except OSError:
                    continue
            if font is not None:
                break
        if font is None:
            font = ImageFont.load_default(size=size)
        self.cache[key] = font
        return font

    def measure(self, text: str, size: int, weight: int = 400) -> float:
        return self._load(size, weight, True).getlength(str(text))

    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: str,
        size: int,
        weight: int = 400,
        fill: tuple[int, int, int, int] = (26, 26, 46, 255),
    ) -> float:
        font = self._load(size, weight, True)
        draw.text(xy, str(text), font=font, fill=fill, anchor="la")
        return font.getlength(str(text))


class UIMessages:
    """线程安全的消息队列（飞行/取图线程写入，GUI 读取）。"""

    def __init__(self, maxlen: int = 40, ttl: float = 8.0) -> None:
        self._items: deque[tuple[float, str, str]] = deque(maxlen=maxlen)
        self._ttl = ttl
        self._lock = Lock()

    def push(self, kind: str, text: str) -> None:
        with self._lock:
            self._items.append((time.monotonic() + self._ttl, kind, text))

    def snapshot(self, now: float | None = None) -> list[tuple[str, str]]:
        now = time.monotonic() if now is None else now
        with self._lock:
            return [(kind, text) for expires, kind, text in self._items if expires > now]


def _rgba(color: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    return color + (alpha,)


def _rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int],
    radius: int = 20,
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(
        tuple(int(value) for value in box),
        radius=int(radius),
        fill=_rgba(fill),
        outline=_rgba(outline) if outline else None,
        width=width,
    )


def _fit_image(
    source_w: int,
    source_h: int,
    box: tuple[int, int, int, int],
) -> tuple[float, float, float, float, float]:
    x1, y1, x2, y2 = box
    scale = min(
        (x2 - x1) / max(1, source_w),
        (y2 - y1) / max(1, source_h),
    )
    width = source_w * scale
    height = source_h * scale
    return (
        x1 + (x2 - x1 - width) / 2,
        y1 + (y2 - y1 - height) / 2,
        width,
        height,
        scale,
    )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _paste_rounded(
    canvas: Image.Image,
    source: Image.Image,
    xy: tuple[int, int],
    size: tuple[int, int],
    radius: int,
) -> None:
    resized = source.resize(size, Image.Resampling.BILINEAR).convert("RGBA")
    mask = Image.new("L", size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255
    )
    canvas.paste(resized, xy, mask)


class _PilRenderer:
    """PIL 预览渲染器：检测框、HUD、chips（原 ui.py DetectionDisplay 精简版）。"""

    def __init__(self) -> None:
        self.scheme = COLORS
        self.fonts = UIFonts()
        self.last_packet: Any = None
        self.last_snapshot: Any = None
        self.last_render: np.ndarray | None = None
        self.render_stats = RollingStats()
        self._render_rate = RateWindow(seconds=3.0)
        self.render_count = 0
        self._next_render_at = 0.0
        self._last_event_pump = 0.0
        self._detection_fresh = False
        # Keep a short history so a DetectionSnapshot can be verified against
        # a frame that was actually displayed. The live queue intentionally
        # drops old frames, so a future detection result must never be painted
        # on an unrelated current frame.
        self._frame_history: deque[Any] = deque(maxlen=8)
        self._detection_packet: Any = None

    def _text(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: str,
        style: str,
        color: str = "text",
    ) -> None:
        size, weight = TYPE[style]
        self.fonts.draw(draw, xy, text, size, weight, _rgba(self.scheme[color]))

    def _center_text(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        text: str,
        style: str,
        color: str = "text",
    ) -> None:
        size, weight = TYPE[style]
        text_width = self.fonts.measure(text, size, weight)
        self.fonts.draw(
            draw,
            ((box[0] + box[2] - text_width) / 2, box[1]),
            text,
            size,
            weight,
            _rgba(self.scheme[color]),
        )

    def _draw_target_icon(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        color: tuple[int, int, int],
        scale: float = 1.0,
    ) -> None:
        radius = int(13 * scale)
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=_rgba(color),
            width=2,
        )
        core = int(4 * scale)
        draw.ellipse(
            (cx - core, cy - core, cx + core, cy + core),
            fill=_rgba(color),
        )
        gap = int(17 * scale)
        length = int(7 * scale)
        draw.line((cx - gap, cy, cx - gap + length, cy), fill=_rgba(color), width=2)
        draw.line((cx + gap - length, cy, cx + gap, cy), fill=_rgba(color), width=2)
        draw.line((cx, cy - gap, cx, cy - gap + length), fill=_rgba(color), width=2)
        draw.line((cx, cy + gap - length, cx, cy + gap), fill=_rgba(color), width=2)

    def _chip(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        text: str,
        *,
        fill: str = "primary_soft",
        color: str = "primary",
        dot: str | None = None,
    ) -> tuple[int, int]:
        size, weight = TYPE["small"]
        dot_width = 12 if dot else 0
        width = int(self.fonts.measure(text, size, weight) + 24 + dot_width)
        height = 30
        _rounded(draw, (x, y, x + width, y + height), self.scheme[fill], 15)
        text_x = x + 12
        if dot:
            draw.ellipse((x + 10, y + 12, x + 16, y + 18), fill=_rgba(self.scheme[dot]))
            text_x += dot_width
        self.fonts.draw(draw, (text_x, y + 7), text, size, weight, _rgba(self.scheme[color]))
        return width, height

    def _draw_preview(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        ui: dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = box
        _rounded(draw, box, self.scheme["preview"], 20)
        inner = (x1 + 15, y1 + 15, x2 - 15, y2 - 15)
        _rounded(draw, inner, self.scheme["preview"], 14)
        image_box = (inner[0] + 1, inner[1] + 1, inner[2] - 1, inner[3] - 1)

        if self.last_packet is None:
            self._draw_target_icon(
                draw,
                (inner[0] + inner[2]) // 2,
                (inner[1] + inner[3]) // 2 - 15,
                self.scheme["primary"],
                1.2,
            )
            self._center_text(
                draw,
                (inner[0], inner[1] + (inner[3] - inner[1]) // 2 + 28, inner[2], inner[3]),
                "系统准备就绪，等待相机画面…",
                "body",
                "muted",
            )
            return

        frame = self.last_packet.frame
        source_h, source_w = frame.shape[:2]
        px, py, pw, ph, scale = _fit_image(source_w, source_h, image_box)
        image = Image.fromarray(frame).convert("RGB")
        _paste_rounded(
            canvas,
            image,
            (int(px), int(py)),
            (max(1, int(pw)), max(1, int(ph))),
            12,
        )

        hud_text = f"实时画面  ·  前视相机  ·  {source_w}×{source_h}"
        hud_w = int(self.fonts.measure(hud_text, *TYPE["small"]) + 32)
        _rounded(
            draw,
            (inner[0] + 12, inner[1] + 12, inner[0] + 12 + hud_w, inner[1] + 42),
            self.scheme["preview"],
            15,
        )
        draw.ellipse(
            (inner[0] + 22, inner[1] + 24, inner[0] + 28, inner[1] + 30),
            fill=_rgba(self.scheme["primary"]),
        )
        self._text(draw, (inner[0] + 37, inner[1] + 19), hud_text, "small", "primary")

        if self.last_snapshot is not None and self._detection_fresh and ui.get("show_detections", True):
            for xmin, ymin, xmax, ymax, _class_id, confidence, name in self.last_snapshot.boxes:
                accent = "primary" if confidence >= 0.45 else "warning"
                left = _clip(px + xmin * scale, image_box[0], image_box[2])
                top = _clip(py + ymin * scale, image_box[1], image_box[3])
                right = _clip(px + xmax * scale, image_box[0], image_box[2])
                bottom = _clip(py + ymax * scale, image_box[1], image_box[3])
                if right <= left or bottom <= top:
                    continue
                draw.rounded_rectangle(
                    (left, top, right, bottom),
                    radius=4,
                    outline=_rgba(self.scheme[accent]),
                    width=3,
                )
                label = f"{_display_name(name)}  {confidence:.0%}"
                label_w = int(self.fonts.measure(label, *TYPE["small"]) + 18)
                label_h = 25
                label_y = top - label_h - 4 if top - label_h - 4 >= image_box[1] else bottom + 4
                label_y = _clip(label_y, image_box[1], image_box[3] - label_h)
                _rounded(
                    draw,
                    (left, label_y, min(image_box[2], left + label_w), label_y + label_h),
                    self.scheme["surface"],
                    8,
                )
                self._text(draw, (left + 9, label_y + 6), label, "small", accent)

        frame_id = self.last_packet.frame_id
        detection_count = (
            len(self.last_snapshot.boxes)
            if self.last_snapshot is not None and self._detection_fresh
            else 0
        )
        self._chip(draw, inner[0] + 12, inner[3] - 42, f"帧 {frame_id:06d}", fill="preview", color="primary")
        self._chip(draw, inner[0] + 128, inner[3] - 42, f"目标 {detection_count}", fill="preview", color="primary")

    def _status_values(self, ui: dict[str, Any]) -> tuple[str, str, str]:
        camera_ok = bool(ui.get("camera_ok"))
        waypoint = int(ui.get("waypoint_index", 0))
        flight_state = str(ui.get("flight_state", ""))
        if flight_state == "ERROR":
            error = str(ui.get("airsim_error", "")).strip()
            return "● 任务异常", "warning", error or "飞行线程出现异常，请检查终端日志"
        if not ui.get("airsim_connected", False):
            error = str(ui.get("airsim_error", "")).strip()
            subtitle = "未检测到 AirSim 模拟器，请先启动场景"
            if error:
                subtitle = f"连接失败：{error}"
            return "等待 AirSim 信号", "muted", subtitle
        if not ui.get("airsim_ready", False):
            return "等待场景就绪", "muted", "AirSim 已连接，场景加载中…"
        if flight_state == "TAKING_OFF":
            return "● 正在起飞", "primary", "正在爬升至巡航高度"
        if flight_state == "STOPPING":
            return "● 正在停止", "warning", "正在取消航段并准备降落"
        if flight_state == "LANDING":
            return "● 正在降落", "primary", "正在执行安全降落"
        if not ui.get("cruise_started", False):
            return "● 任务待命", "primary", "点击“多航点巡航”开始任务"
        if not camera_ok:
            camera_error = str(ui.get("camera_error", "")).strip()
            return "待命", "muted", f"相机连接失败：{camera_error}" if camera_error else "等待相机连接"
        if self._detection_fresh and self.last_snapshot is not None and self.last_snapshot.boxes:
            return "● 发现目标", "warning", "正在进行目标检测"
        if waypoint > 0:
            return "● 自动巡航", "primary", "实时视觉检测中"
        return "● 监测中", "success", "相机已连接"


# ---------------------------------------------------------------------------
# 颜色与字体辅助（Qt 侧）
# ---------------------------------------------------------------------------


def _hex(name: str) -> str:
    r, g, b = COLORS[name]
    return f"#{r:02X}{g:02X}{b:02X}"


def _mix_white(name: str, alpha: float) -> str:
    """MD3 状态层：在底色上叠加半透明白（hover 8% / pressed 12%）。"""
    r, g, b = COLORS[name]
    r = int(r + (255 - r) * alpha)
    g = int(g + (255 - g) * alpha)
    b = int(b + (255 - b) * alpha)
    return f"#{r:02X}{g:02X}{b:02X}"


def _pick_font(candidates: tuple[str, ...]) -> str:
    families = set(QFontDatabase.families())
    for name in candidates:
        if name in families:
            return name
    return candidates[-1]


_font_names_cache: tuple[str, str] | None = None


def _font_names() -> tuple[str, str]:
    """懒加载字体名：QFontDatabase 需要 QApplication 已存在。"""
    global _font_names_cache
    if _font_names_cache is None:
        _font_names_cache = (
            _pick_font(("Microsoft YaHei", "SimHei", "微软雅黑", "黑体")),
            _pick_font(("Bahnschrift", "Segoe UI")),
        )
    return _font_names_cache


def _make_font(size: int, weight: int = 400, latin: bool = False) -> QFont:
    cjk_name, latin_name = _font_names()
    font = QFont(latin_name if latin else cjk_name)
    font.setPixelSize(size)
    font.setWeight(weight)
    return font


# ---------------------------------------------------------------------------
# Qt 自定义控件
# ---------------------------------------------------------------------------


class RouteWidget(QWidget):
    """航点路线图：水平线 + 圆点标记，当前航点高亮。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total = 1
        self._index = 0
        self.setMinimumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_progress(self, total: int, index: int) -> None:
        total = max(1, int(total))
        index = max(0, min(int(index), total))
        if (total, index) != (self._total, self._index):
            self._total, self._index = total, index
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt 命名)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        line_y = height // 2
        left, right = 14, width - 14

        painter.setPen(QPen(QColor(_hex("border")), 3))
        painter.drawLine(left, line_y, right, line_y)

        step = (right - left) / max(1, self._total - 1)
        for point_index in range(self._total):
            px = left + step * point_index
            active = point_index < self._index
            current = point_index == max(0, self._index - 1)
            if current:
                painter.setPen(QPen(QColor(_hex("primary")), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(int(px) - 11, line_y - 11, 22, 22)
            painter.setPen(QPen(QColor(_hex("primary")), 2))
            painter.setBrush(QColor(_hex("primary")) if active or current else QColor(_hex("surface")))
            radius = 7 if current else 5
            painter.drawEllipse(int(px) - radius, line_y - radius, radius * 2, radius * 2)


class StatusStrip(QWidget):
    """紧凑模式底部状态条：相机 / 航点 / 高度 / 速度 / 目标 五栏。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusStrip")
        self.setStyleSheet(
            f"#statusStrip {{ background: {_hex('surface')}; border-radius: 16px; }}"
        )
        self._labels: dict[str, QLabel] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(24)
        for key, title in (
            ("camera", "相机"),
            ("waypoint", "航点"),
            ("altitude", "高度"),
            ("speed", "速度"),
            ("target", "目标"),
        ):
            column = QVBoxLayout()
            column.setSpacing(2)
            caption = QLabel(title)
            caption.setFont(_make_font(12))
            caption.setStyleSheet(f"color: {_hex('muted')};")
            value = QLabel("—")
            value.setFont(_make_font(18, 700, latin=True))
            value.setStyleSheet(f"color: {_hex('text')};")
            column.addWidget(caption)
            column.addWidget(value)
            layout.addLayout(column)
            self._labels[key] = value
        layout.addStretch(1)

    def set_values(self, ui: dict[str, Any]) -> None:
        index = int(ui.get("waypoint_index", 0))
        total = max(1, int(ui.get("waypoints_total", 1)))
        self._labels["camera"].setText("在线" if ui.get("camera_ok") else "等待")
        self._labels["waypoint"].setText(f"{min(index, total)}/{total}")
        self._labels["altitude"].setText(f"{-float(ui.get('altitude', 0.0)):.1f}")
        self._labels["speed"].setText(f"{float(ui.get('speed', 0.0)):.2f}")
        self._labels["target"].setText(str(ui.get("detections", 0)))


class _MainWindow(QMainWindow):
    """带关闭通知的主窗口。"""

    closed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        self.closed.emit()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Qt 版 DetectionDisplay
# ---------------------------------------------------------------------------


class DetectionDisplay(_PilRenderer):
    """SCD 风格 Qt 控制台：左侧 PIL 视频区 + 右侧 Qt Widgets 面板。"""

    COMPACT_MIN_WIDTH = 1120
    COMPACT_MIN_HEIGHT = 720

    def __init__(self, theme: str = "light") -> None:
        super().__init__()
        self._app = QApplication.instance() or QApplication([])
        self._quit_requested = False
        self._window_shown = False
        self._build_window()

    # ------------------------------------------------------------------
    # 窗口构建
    # ------------------------------------------------------------------
    def _build_window(self) -> None:
        self._window = _MainWindow()
        self._window.closed.connect(self._request_quit)
        QShortcut(QKeySequence("Q"), self._window).activated.connect(self._request_quit)
        self._window.setStyleSheet(self._qss())

        root = QWidget()
        self._window.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # ---- 左侧视频区 ----
        self.video_label = QLabel()
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"background: {_hex('preview')}; border-radius: 20px;")
        root_layout.addWidget(self.video_label, stretch=7)

        # ---- 右侧面板 ----
        self._build_panel()
        root_layout.addWidget(self.right_panel, stretch=3)

        # ---- 紧凑状态条 ----
        self.status_strip = StatusStrip()
        root_layout.addWidget(self.status_strip)
        self.status_strip.hide()

    def _build_panel(self) -> None:
        self.right_panel = QWidget()
        panel = QVBoxLayout(self.right_panel)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(16)

        # ---- 状态卡 ----
        self.status_card = self._make_md3_card(fill="surface")
        status_layout = self.status_card.layout()
        self.status_title = QLabel("● 监测中")
        self.status_title.setFont(_make_font(24, 700))
        self.status_title.setStyleSheet(f"color: {_hex('success')};")
        self.status_subtitle = QLabel("相机已连接")
        self.status_subtitle.setFont(_make_font(13))
        self.status_subtitle.setStyleSheet(f"color: {_hex('muted')};")
        self.status_subtitle.setWordWrap(True)
        self.status_subtitle.setMaximumHeight(42)
        status_layout.addWidget(self.status_title)
        status_layout.addWidget(self.status_subtitle)
        panel.addWidget(self.status_card)

        # ---- 飞行数据卡 ----
        self.data_card = self._make_md3_card()
        data_layout = self.data_card.layout()
        self.card_camera = self._make_info_row()
        self.card_waypoint = self._make_info_row()
        self.card_flight = self._make_info_row()
        self.card_performance = self._make_info_row()
        self.card_performance.setFont(_make_font(13))
        self.card_performance.setStyleSheet(f"color: {_hex('muted')};")
        self.card_drops = self._make_info_row()
        self.card_drops.setFont(_make_font(13))
        self.card_drops.setStyleSheet(f"color: {_hex('muted')};")
        data_layout.addWidget(self.card_camera)
        data_layout.addWidget(self.card_waypoint)
        data_layout.addWidget(self.card_flight)
        data_layout.addWidget(self.card_performance)
        data_layout.addWidget(self.card_drops)
        panel.addWidget(self.data_card)

        # ---- 巡航进度卡 ----
        self.progress_card = self._make_md3_card()
        progress_layout = self.progress_card.layout()
        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)
        progress_caption = QLabel("巡航进度")
        progress_caption.setFont(_make_font(13))
        progress_caption.setStyleSheet(f"color: {_hex('muted')};")
        self.round_label = QLabel("0 / 1 · 第 0 圈")
        self.round_label.setFont(_make_font(13))
        self.round_label.setStyleSheet(f"color: {_hex('muted')};")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setStyleSheet(
            f"QProgressBar {{ background: {_hex('surface_high')}; border: none;"
            f" border-radius: 2px; }}"
            f"QProgressBar::chunk {{ background: {_hex('primary')}; border-radius: 2px; }}"
        )
        progress_row.addWidget(progress_caption)
        progress_row.addWidget(self.progress, stretch=1)
        progress_row.addWidget(self.round_label)
        progress_layout.addLayout(progress_row)
        self.route_widget = RouteWidget()
        progress_layout.addWidget(self.route_widget)
        panel.addWidget(self.progress_card)

        # ---- 目标卡（全部检测目标；固定高度 + 滚动，避免目标过多拉长面板）----
        self.target_card = self._make_md3_card()
        target_layout = self.target_card.layout()
        target_title = QLabel("检测目标")
        target_title.setFont(_make_font(13))
        target_title.setStyleSheet(f"color: {_hex('muted')};")
        target_layout.addWidget(target_title)
        self.target_scroll = QScrollArea()
        self.target_scroll.setWidgetResizable(True)
        self.target_scroll.setFixedHeight(140)
        self.target_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.target_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }"
            f"QScrollBar::handle:vertical {{ background: {_hex('border')};"
            " border-radius: 3px; min-height: 20px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        self.target_label = QLabel("当前画面没有检测目标")
        self.target_label.setFont(_make_font(15))
        self.target_label.setWordWrap(True)
        self.target_label.setStyleSheet(f"color: {_hex('muted_light')};")
        self.target_scroll.setWidget(self.target_label)
        target_layout.addWidget(self.target_scroll)
        panel.addWidget(self.target_card)

        panel.addStretch(1)

        # ---- 操作卡（按钮绑定真实控制：巡航启停 / 检测框显示开关）----
        self.action_card = self._make_md3_card(fill="surface")
        action_layout = self.action_card.layout()
        self.btn_cruise = self._make_button("多航点巡航")
        self.btn_detect = self._make_button("实时视觉检测")
        self.btn_stop = self._make_button("停止任务", danger=True)
        self.btn_cruise.clicked.connect(self._on_cruise_clicked)
        self.btn_detect.clicked.connect(self._on_detect_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        action_layout.addWidget(self.btn_cruise)
        action_layout.addWidget(self.btn_detect)
        action_layout.addWidget(self.btn_stop)
        panel.addWidget(self.action_card)
        self._cruise_ui = None

    def _make_md3_card(self, fill: str = "soft_gray") -> QFrame:
        """MD3 FilledCard：圆角 20px + 布局内边距（QSS padding 对 QFrame 无效）。"""
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: {_hex(fill)}; border: none; border-radius: 20px; }}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        return card

    def _make_info_row(self, text: str = "—") -> QLabel:
        label = QLabel(text)
        label.setFont(_make_font(15))
        label.setStyleSheet(f"color: {_hex('text')};")
        return label

    def _make_button(self, text: str, danger: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFixedHeight(48)
        base = _hex("warning") if danger else _hex("primary")
        hover = _mix_white("warning", 0.08) if danger else _mix_white("primary", 0.08)
        pressed = _mix_white("warning", 0.12) if danger else _mix_white("primary", 0.12)
        button.setStyleSheet(
            f"QPushButton {{ background: {base}; color: white;"
            f" border: none; border-radius: 24px; font-size: 15px; font-weight: 700; }}"
            f"QPushButton:hover {{ background: {hover}; }}"
            f"QPushButton:pressed {{ background: {pressed}; }}"
        )
        return button

    def _qss(self) -> str:
        return (
            f"QMainWindow, QWidget {{ background: {_hex('bg')}; }}"
            f"QLabel {{ background: transparent; }}"
        )

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def _request_quit(self) -> None:
        self._quit_requested = True

    def _push_message(self, ui: dict[str, Any], text: str) -> None:
        print(f"[UI] {text}")
        messages = ui.get("messages")
        if messages is not None:
            messages.push("info", text)

    def _on_cruise_clicked(self) -> None:
        """“多航点巡航”：触发巡航开始（停止后可重新开始）。"""
        ui = self._cruise_ui
        if ui is None:
            return
        if str(ui.get("flight_state", "")) in {"TAKING_OFF", "CRUISING", "STOPPING", "LANDING"}:
            self._push_message(ui, "任务正在执行")
            return
        if ui.get("cruise_started", False):
            self._push_message(ui, "巡航进行中")
            return
        if not ui.get("airsim_connected", False):
            self._push_message(ui, "正在等待 AirSim 信号，请启动模拟器")
            return
        ui["start_cruise"].set()
        ui["stop_cruise"].clear()
        if ui.get("airsim_ready", False):
            self._push_message(ui, "已发送开始指令，即将起飞")
        else:
            self._push_message(ui, "场景加载完成后自动开始巡航")

    def _on_stop_clicked(self) -> None:
        """“停止任务”：停止巡航并降落（程序保持运行）。"""
        ui = self._cruise_ui
        if ui is None:
            return
        if not ui.get("cruise_started", False) and str(ui.get("flight_state", "")) not in {
            "TAKING_OFF",
            "CRUISING",
            "STOPPING",
            "LANDING",
        }:
            self._push_message(ui, "任务未在运行")
            return
        ui["stop_cruise"].set()
        self._push_message(ui, "正在停止巡航并降落…")

    def _on_detect_clicked(self) -> None:
        """“实时视觉检测”：切换检测框显示。"""
        ui = self._cruise_ui
        if ui is None:
            return
        show = not ui.get("show_detections", True)
        ui["show_detections"] = show
        self._push_message(ui, f"检测框显示已{'开启' if show else '关闭'}")

    # ------------------------------------------------------------------
    # 对外接口（与旧 ui.py 语义一致）
    # ------------------------------------------------------------------
    def show(self, packet: Any, snapshot: Any, args: Any, ui: dict[str, Any]) -> bool:
        if args.no_display:
            return False
        self._cruise_ui = ui
        has_update = packet is not None or snapshot is not None or not self._window_shown
        if packet is not None:
            self.last_packet = packet
            self._frame_history.append(packet)
        if snapshot is not None:
            self.last_snapshot = snapshot
        # 帧号同步：检测结果必须来自已经进入显示历史的帧；未来帧、
        # 过期帧和无法找到原图的结果都不再绘制，避免检测框错位。
        self._detection_packet = None
        if self.last_packet is None or self.last_snapshot is None:
            self._detection_fresh = False
        else:
            current_id = int(self.last_packet.frame_id)
            detection_id = int(self.last_snapshot.frame_id)
            self._detection_packet = next(
                (
                    history_packet
                    for history_packet in reversed(self._frame_history)
                    if int(history_packet.frame_id) == detection_id
                ),
                None,
            )
            self._detection_fresh = (
                self._detection_packet is not None
                and 0 <= current_id - detection_id <= 2
            )
        now = time.monotonic()
        if not has_update or (self._window_shown and now < self._next_render_at):
            return self.process_events()
        if not self._window_shown:
            self._window.resize(max(960, int(args.display_width)), max(600, int(args.display_height)))
            self._window.show()
            self._window_shown = True

        render_started = time.perf_counter()
        self._render_video(ui, save_render=int(getattr(args, "save_ui_every", 0)) > 0)
        self._update_panel(ui)
        self._update_compact()
        render_ms = (time.perf_counter() - render_started) * 1000.0
        self.render_stats.add(render_ms)
        self._render_rate.mark()
        self.render_count += 1
        display_fps = max(1.0, float(getattr(args, "display_fps", 18.0)))
        self._next_render_at = time.monotonic() + 1.0 / display_fps
        self.process_events(force=True)
        return self._quit_requested

    def process_events(self, force: bool = False) -> bool:
        now = time.monotonic()
        # Qt event pumping is useful at about 30 Hz; calling it every 5–10 ms
        # only adds main-thread work when no new frame is available.
        if force or now - self._last_event_pump >= 1.0 / 30.0:
            QApplication.processEvents()
            self._last_event_pump = now
        return self._quit_requested

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def _render_video(self, ui: dict[str, Any], *, save_render: bool = False) -> None:
        """左侧：PIL 渲染（检测框/HUD）到内存图像 -> QPixmap -> QLabel。"""
        size = self.video_label.size()
        width = max(320, size.width())
        height = max(240, size.height())
        canvas = Image.new("RGBA", (width, height), (*COLORS["bg"], 255))
        draw = ImageDraw.Draw(canvas, "RGBA")
        self._draw_preview(canvas, draw, (0, 0, width, height), ui)

        rgb = np.asarray(canvas.convert("RGB"))
        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimage.copy()))

        # Only create the extra BGR copy when UI frame saving is enabled.
        self.last_render = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if save_render else None

    @property
    def render_fps(self) -> float:
        return self._render_rate.rate()

    @property
    def render_performance(self) -> StatsSnapshot:
        return self.render_stats.snapshot()

    def _update_panel(self, ui: dict[str, Any]) -> None:
        status, color_key, subtitle = self._status_values(ui)
        self.status_title.setText(status)
        self.status_title.setStyleSheet(f"color: {_hex(color_key)};")
        self.status_subtitle.setText(subtitle)

        camera_ok = bool(ui.get("camera_ok"))
        index = int(ui.get("waypoint_index", 0))
        total = max(1, int(ui.get("waypoints_total", 1)))
        round_no = int(ui.get("patrol_round", 0))
        altitude = -float(ui.get("altitude", 0.0))
        speed = float(ui.get("speed", 0.0))
        capture_fps = float(ui.get("capture_fps", 0.0))
        detection_fps = float(ui.get("detection_fps", 0.0))
        inference_ms = float(ui.get("inference_ms", 0.0))
        detection_latency_ms = float(ui.get("detection_latency_ms", 0.0))
        capture_rpc_avg_ms = float(ui.get("capture_rpc_avg_ms", ui.get("capture_rpc_ms", 0.0)))
        capture_rpc_max_ms = float(ui.get("capture_rpc_max_ms", 0.0))
        image_parse_avg_ms = float(ui.get("image_parse_avg_ms", 0.0))
        image_parse_max_ms = float(ui.get("image_parse_max_ms", 0.0))
        detection_avg_ms = float(ui.get("detection_avg_ms", inference_ms))
        detection_max_ms = float(ui.get("detection_max_ms", 0.0))
        render_fps = float(ui.get("render_fps", 0.0))
        camera_drops = int(ui.get("camera_drops", 0))
        detection_drops = int(ui.get("detection_drops", 0))
        detection_latency_avg_ms = float(ui.get("detection_latency_avg_ms", detection_latency_ms))
        detection_latency_max_ms = float(ui.get("detection_latency_max_ms", 0.0))
        boxes = (
            self.last_snapshot.boxes
            if self.last_snapshot is not None and self._detection_fresh
            else ()
        )
        # 同步有效目标数（只统计与当前画面匹配的新鲜结果），紧凑状态条据此显示
        ui["detections"] = len(boxes)

        self.card_camera.setText(
            f"相机状态：{'已连接' if camera_ok else '等待连接'}    目标：{len(boxes)} 个"
        )
        self.card_waypoint.setText(f"航点进度：{min(index, total):02d} / {total:02d}    第 {round_no} 圈")
        self.card_flight.setText(f"飞行高度：{altitude:.1f} 米    速度：{speed:.2f} 米/秒")
        self.card_performance.setText(
            f"采集：{capture_fps:.1f} 帧/秒    取图：{capture_rpc_avg_ms:.0f}/{capture_rpc_max_ms:.0f} 毫秒    "
            f"解析：{image_parse_avg_ms:.1f}/{image_parse_max_ms:.1f} 毫秒\n"
            f"推理：{detection_fps:.1f} 帧/秒    YOLO：{detection_avg_ms:.0f}/{detection_max_ms:.0f} 毫秒    "
            f"GUI：{render_fps:.1f} 帧/秒"
        )
        self.card_drops.setText(
            f"累计丢帧：相机 {camera_drops}    检测 {detection_drops}    "
            f"延迟：{detection_latency_avg_ms:.0f}/{detection_latency_max_ms:.0f} 毫秒"
        )

        self.progress.setRange(0, total)
        self.progress.setValue(min(index, total))
        self.round_label.setText(f"{min(index, total):02d} / {total:02d} · 第 {round_no} 圈")
        self.route_widget.set_progress(total, index)

        if boxes and self._detection_fresh:
            # 显示全部检测目标：按置信度降序，逐行着色
            ordered = sorted(boxes, key=lambda box: float(box[5]), reverse=True)
            lines = []
            for box in ordered:
                name = _display_name(box[6])
                confidence = float(box[5])
                color = "warning" if confidence < 0.45 else "primary"
                lines.append(
                    f'<span style="color:{_hex(color)}; font-size:15px; font-weight:600;">'
                    f"{name}  {confidence:.0%}</span>"
                )
            self.target_label.setText("<br>".join(lines))
            self.target_label.setTextFormat(Qt.TextFormat.RichText)
            self.target_label.setWordWrap(True)
            self.target_label.setStyleSheet("")
        elif boxes:
            # 检测结果已过期（帧不同步）：不显示旧目标，避免与画面错位
            self.target_label.setText("检测更新中…")
            self.target_label.setTextFormat(Qt.TextFormat.PlainText)
            self.target_label.setWordWrap(True)
            self.target_label.setStyleSheet(f"color: {_hex('muted_light')};")
        else:
            self.target_label.setText("当前画面没有检测目标")
            self.target_label.setTextFormat(Qt.TextFormat.PlainText)
            self.target_label.setWordWrap(True)
            self.target_label.setStyleSheet(f"color: {_hex('muted_light')};")

        self.status_strip.set_values(ui)

    def _update_compact(self) -> None:
        """窗口过窄/过矮时切换为底部紧凑状态条。"""
        compact = (
            self._window.width() < self.COMPACT_MIN_WIDTH
            or self._window.height() < self.COMPACT_MIN_HEIGHT
        )
        self.right_panel.setVisible(not compact)
        self.status_strip.setVisible(compact)
