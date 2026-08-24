"""PyQt6 前端（自包含）：左侧 PIL 视频区 + 右侧 Qt Widgets 面板。

本模块不再依赖独立的 ``ui.py``（原纯 PIL 前端已移除）：PIL 渲染基础设施
（配色、字体、预览绘制）直接内联在本模块中，仅保留 Qt 版实际用到
的部分。

视觉规范：浅色大圆角扁平风格（纯色无渐变、大胶囊圆角、无文字描线/无聚焦虚框）。
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
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
# PIL 渲染基础设施
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


# 浅色大圆角扁平色板（纯色、无渐变）
COLORS: dict[str, tuple[int, int, int]] = {
    "bg": (244, 245, 247),  # 主背景（浅灰）
    "surface": (255, 255, 255),  # 卡片背景（纯白）
    "surface_sub": (238, 240, 244),  # 次级卡片/浅底
    "preview": (228, 232, 240),  # 视频区未就绪浅底
    "primary": (37, 99, 235),  # 主色调蓝
    "primary_soft": (219, 234, 254),  # 软蓝背景
    "primary_dark": (29, 78, 216),  # Hover 蓝
    "text": (30, 41, 59),  # 主文字（深灰/接近黑）
    "muted": (100, 116, 139),  # 次要文字
    "muted_light": (148, 163, 184),  # 浅灰/占位符
    "border": (226, 232, 240),  # 卡片外边框色
    "success": (22, 163, 74),  # 成功绿
    "warning": (220, 38, 38),  # 警告红
    "warning_soft": (254, 226, 226),  # 警告浅红
    "white": (255, 255, 255),
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


def _rgba(color: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    return color + (alpha,)


def _rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int],
    radius: int = 24,
) -> None:
    # 彻底去掉外侧 outline 描边
    draw.rounded_rectangle(
        tuple(int(value) for value in box),
        radius=int(radius),
        fill=_rgba(fill),
        outline=None,
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
    radius: int = 20,
) -> None:
    resized = source.resize(size, Image.Resampling.BILINEAR).convert("RGBA")
    mask = Image.new("L", size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255
    )
    canvas.paste(resized, xy, mask)


class _PilRenderer:
    """PIL 预览渲染器：精简四角检测框、无描边悬浮 HUD 胶囊。"""

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
        self._frame_history: deque[Any] = deque(maxlen=8)

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

    def _draw_corner_bbox(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[float, float, float, float],
        color: tuple[int, int, int],
        corner_len: int = 12,
        width: int = 3,
    ) -> None:
        x1, y1, x2, y2 = box
        # 四角锚点线段
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=_rgba(color), width=width)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=_rgba(color), width=width)
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill=_rgba(color), width=width)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=_rgba(color), width=width)
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=_rgba(color), width=width)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill=_rgba(color), width=width)
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill=_rgba(color), width=width)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill=_rgba(color), width=width)

    def _chip(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        text: str,
        *,
        fill: str = "surface",
        color: str = "text",
        dot: str | None = None,
    ) -> tuple[int, int]:
        size, weight = TYPE["small"]
        dot_width = 12 if dot else 0
        width = int(self.fonts.measure(text, size, weight) + 24 + dot_width)
        height = 32
        _rounded(draw, (x, y, x + width, y + height), self.scheme[fill], radius=16)
        text_x = x + 12
        if dot:
            draw.ellipse((x + 10, y + 13, x + 16, y + 19), fill=_rgba(self.scheme[dot]))
            text_x += dot_width
        self.fonts.draw(
            draw, (text_x, y + 8), text, size, weight, _rgba(self.scheme[color])
        )
        return width, height

    def _draw_preview(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        ui: dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = box
        _rounded(draw, box, self.scheme["surface"], radius=28)
        inner = (x1 + 12, y1 + 12, x2 - 12, y2 - 12)
        _rounded(draw, inner, self.scheme["preview"], radius=20)
        image_box = (inner[0] + 1, inner[1] + 1, inner[2] - 1, inner[3] - 1)

        if self.last_packet is None:
            self._center_text(
                draw,
                (
                    inner[0],
                    inner[1] + (inner[3] - inner[1]) // 2 - 10,
                    inner[2],
                    inner[3],
                ),
                "系统就绪 · 等待相机画面",
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
            radius=18,
        )

        # 顶部悬浮 HUD（无边框纯白胶囊）
        hud_text = f"前视监控  ·  {source_w}×{source_h}"
        hud_w = int(self.fonts.measure(hud_text, *TYPE["small"]) + 36)
        _rounded(
            draw,
            (inner[0] + 16, inner[1] + 16, inner[0] + 16 + hud_w, inner[1] + 48),
            self.scheme["surface"],
            radius=16,
        )
        draw.ellipse(
            (inner[0] + 26, inner[1] + 29, inner[0] + 32, inner[1] + 35),
            fill=_rgba(self.scheme["success"]),
        )
        self._text(draw, (inner[0] + 40, inner[1] + 23), hud_text, "small", "text")

        if (
            self.last_snapshot is not None
            and self._detection_fresh
            and ui.get("show_detections", True)
        ):
            for (
                xmin,
                ymin,
                xmax,
                ymax,
                _class_id,
                confidence,
                name,
            ) in self.last_snapshot.boxes:
                accent = "primary" if confidence >= 0.45 else "warning"
                left = _clip(px + xmin * scale, image_box[0], image_box[2])
                top = _clip(py + ymin * scale, image_box[1], image_box[3])
                right = _clip(px + xmax * scale, image_box[0], image_box[2])
                bottom = _clip(py + ymax * scale, image_box[1], image_box[3])
                if right <= left or bottom <= top:
                    continue

                self._draw_corner_bbox(
                    draw,
                    (left, top, right, bottom),
                    self.scheme[accent],
                    corner_len=14,
                    width=3,
                )

                label = f"{_display_name(name)} {confidence:.0%}"
                label_w = int(self.fonts.measure(label, *TYPE["small"]) + 18)
                label_h = 24
                label_y = (
                    top - label_h - 4
                    if top - label_h - 4 >= image_box[1]
                    else bottom + 4
                )
                label_y = _clip(label_y, image_box[1], image_box[3] - label_h)

                # 悬浮文字标签（无描边纯白底）
                _rounded(
                    draw,
                    (
                        left,
                        label_y,
                        min(image_box[2], left + label_w),
                        label_y + label_h,
                    ),
                    self.scheme["surface"],
                    radius=12,
                )
                self._text(draw, (left + 9, label_y + 5), label, "small", accent)

        frame_id = self.last_packet.frame_id
        detection_count = (
            len(self.last_snapshot.boxes)
            if self.last_snapshot is not None and self._detection_fresh
            else 0
        )
        self._chip(
            draw,
            inner[0] + 16,
            inner[3] - 48,
            f"帧 {frame_id:06d}",
            fill="surface",
            color="text",
        )
        self._chip(
            draw,
            inner[0] + 130,
            inner[3] - 48,
            f"目标 {detection_count}",
            fill="surface",
            color="primary",
        )

    def _status_values(self, ui: dict[str, Any]) -> tuple[str, str, str]:
        camera_ok = bool(ui.get("camera_ok"))
        waypoint = int(ui.get("waypoint_index", 0))
        flight_state = str(ui.get("flight_state", ""))
        if flight_state == "ERROR":
            error = str(ui.get("airsim_error", "")).strip()
            return "任务异常", "warning", error or "飞行线程异常，请检查终端日志"
        if not ui.get("airsim_connected", False):
            error = str(ui.get("airsim_error", "")).strip()
            subtitle = "未连接模拟器，请启动 AirSim 场景"
            if error:
                subtitle = f"连接失败：{error}"
            return "等待 AirSim 信号", "muted", subtitle
        if not ui.get("airsim_ready", False):
            return "加载场景", "muted", "AirSim 已连接，加载中…"
        if flight_state == "TAKING_OFF":
            return "正在起飞", "primary", "正在爬升至巡航高度"
        if flight_state == "STOPPING":
            return "正在停止", "warning", "取消航段中，准备降落"
        if flight_state == "LANDING":
            return "正在降落", "primary", "执行安全降落"
        if not ui.get("cruise_started", False):
            return "系统待命", "primary", "点击下方“多航点巡航”开始"
        if not camera_ok:
            camera_error = str(ui.get("camera_error", "")).strip()
            return (
                "待命",
                "muted",
                f"相机连接失败：{camera_error}" if camera_error else "等待相机",
            )
        if (
            self._detection_fresh
            and self.last_snapshot is not None
            and self.last_snapshot.boxes
        ):
            return "发现目标", "warning", "实时目标视觉识别中"
        if waypoint > 0:
            return "自动巡航", "primary", "多航点路线巡航中"
        return "监测中", "success", "设备运行正常"


# ---------------------------------------------------------------------------
# Qt 辅助与组件
# ---------------------------------------------------------------------------


def _hex(name: str) -> str:
    r, g, b = COLORS[name]
    return f"#{r:02X}{g:02X}{b:02X}"


def _pick_font(candidates: tuple[str, ...]) -> str:
    families = set(QFontDatabase.families())
    for name in candidates:
        if name in families:
            return name
    return candidates[-1]


_font_names_cache: tuple[str, str] | None = None


def _font_names() -> tuple[str, str]:
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


class RouteWidget(QWidget):
    """航点路线图。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._total = 1
        self._index = 0
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_progress(self, total: int, index: int) -> None:
        total = max(1, int(total))
        index = max(0, min(int(index), total))
        if (total, index) != (self._total, self._index):
            self._total, self._index = total, index
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        line_y = height // 2
        left, right = 16, width - 16

        painter.setPen(QPen(QColor(_hex("surface_sub")), 4))
        painter.drawLine(left, line_y, right, line_y)

        step = (right - left) / max(1, self._total - 1)
        for point_index in range(self._total):
            px = left + step * point_index
            active = point_index < self._index
            current = point_index == max(0, self._index - 1)

            if current:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(_hex("primary_soft")))
                painter.drawEllipse(int(px) - 10, line_y - 10, 20, 20)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(
                QColor(_hex("primary"))
                if active or current
                else QColor(_hex("muted_light"))
            )
            radius = 6 if current else 4
            painter.drawEllipse(
                int(px) - radius, line_y - radius, radius * 2, radius * 2
            )


class MetricTile(QFrame):
    """大圆角无描线数据小卡片。"""

    def __init__(
        self, title: str, unit: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"MetricTile {{ background: {_hex('surface_sub')}; border-radius: 18px; border: none; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        self.label_title = QLabel(title)
        self.label_title.setFont(_make_font(12))
        self.label_title.setStyleSheet(f"color: {_hex('muted')}; border: none;")

        self.label_val = QLabel("—")
        self.label_val.setFont(_make_font(20, 700, latin=True))
        self.label_val.setStyleSheet(f"color: {_hex('text')}; border: none;")

        self._unit = unit

        layout.addWidget(self.label_title)
        layout.addWidget(self.label_val)

    def set_value(self, val_str: str) -> None:
        if self._unit:
            self.label_val.setText(f"{val_str} {self._unit}")
        else:
            self.label_val.setText(val_str)


class _MainWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# 主 UI 控制台
# ---------------------------------------------------------------------------


class DetectionDisplay(_PilRenderer):
    """极简浅色大圆角 UI 控制台。"""

    def __init__(self, theme: str = "light") -> None:
        super().__init__()
        self._app = QApplication.instance() or QApplication([])
        self._quit_requested = False
        self._window_shown = False
        self._build_window()

    def _build_window(self) -> None:
        self._window = _MainWindow()
        self._window.closed.connect(self._request_quit)
        QShortcut(QKeySequence("Q"), self._window).activated.connect(self._request_quit)
        self._window.setStyleSheet(self._qss())

        root = QWidget()
        self._window.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(20)

        # ---- 左侧视频区 ----
        self.video_label = QLabel()
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            f"background: {_hex('surface')}; border-radius: 28px; border: 1px solid {_hex('border')};"
        )
        root_layout.addWidget(self.video_label, stretch=7)

        # ---- 右侧面板 ----
        self._build_panel()
        root_layout.addWidget(self.right_panel, stretch=3)

    def _build_panel(self) -> None:
        self.right_panel = QWidget()
        panel = QVBoxLayout(self.right_panel)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(16)

        # 1. 顶部 Hero 核心状态卡片
        self.status_card = self._make_card(radius=24)
        status_layout = self.status_card.layout()

        top_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setFont(_make_font(18))
        self.status_dot.setStyleSheet(f"color: {_hex('success')}; border: none;")

        self.status_title = QLabel("监测中")
        self.status_title.setFont(_make_font(20, 700))
        self.status_title.setStyleSheet(f"color: {_hex('text')}; border: none;")

        top_row.addWidget(self.status_dot)
        top_row.addWidget(self.status_title)
        top_row.addStretch(1)

        self.status_subtitle = QLabel("相机设备连接正常")
        self.status_subtitle.setFont(_make_font(13))
        self.status_subtitle.setStyleSheet(f"color: {_hex('muted')}; border: none;")
        self.status_subtitle.setWordWrap(True)

        status_layout.addLayout(top_row)
        status_layout.addWidget(self.status_subtitle)
        panel.addWidget(self.status_card)

        # 2. 核心指标 2x2 网格小卡片
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.tile_alt = MetricTile("飞行高度", "m")
        self.tile_spd = MetricTile("飞行速度", "m/s")
        self.tile_fps = MetricTile("推理帧率", "fps")
        self.tile_lat = MetricTile("系统延迟", "ms")

        grid.addWidget(self.tile_alt, 0, 0)
        grid.addWidget(self.tile_spd, 0, 1)
        grid.addWidget(self.tile_fps, 1, 0)
        grid.addWidget(self.tile_lat, 1, 1)
        panel.addWidget(grid_widget)

        # 3. 巡航进度卡片
        self.progress_card = self._make_card(radius=24)
        progress_layout = self.progress_card.layout()

        prog_header = QHBoxLayout()
        prog_title = QLabel("巡航进度")
        prog_title.setFont(_make_font(13, 700))
        prog_title.setStyleSheet(f"color: {_hex('text')}; border: none;")

        self.round_label = QLabel("0 / 0 航点")
        self.round_label.setFont(_make_font(12, latin=True))
        self.round_label.setStyleSheet(f"color: {_hex('muted')}; border: none;")

        prog_header.addWidget(prog_title)
        prog_header.addStretch(1)
        prog_header.addWidget(self.round_label)
        progress_layout.addLayout(prog_header)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.setStyleSheet(
            f"QProgressBar {{ background: {_hex('surface_sub')}; border: none; border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background: {_hex('primary')}; border-radius: 4px; }}"
        )
        progress_layout.addWidget(self.progress)

        self.route_widget = RouteWidget()
        progress_layout.addWidget(self.route_widget)
        panel.addWidget(self.progress_card)

        # 4. 检测目标展示卡
        self.target_card = self._make_card(radius=24)
        target_layout = self.target_card.layout()

        target_title = QLabel("检测目标")
        target_title.setFont(_make_font(13, 700))
        target_title.setStyleSheet(f"color: {_hex('text')}; border: none;")
        target_layout.addWidget(target_title)

        self.target_scroll = QScrollArea()
        self.target_scroll.setWidgetResizable(True)
        self.target_scroll.setFixedHeight(95)
        self.target_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.target_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: transparent; width: 4px; margin: 0; }"
            f"QScrollBar::handle:vertical {{ background: {_hex('border')}; border-radius: 2px; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )

        self.target_label = QLabel("当前画面无检测目标")
        self.target_label.setFont(_make_font(13))
        self.target_label.setWordWrap(True)
        self.target_label.setStyleSheet(f"color: {_hex('muted_light')}; border: none;")
        self.target_scroll.setWidget(self.target_label)
        target_layout.addWidget(self.target_scroll)
        panel.addWidget(self.target_card)

        panel.addStretch(1)

        # 5. 操作按钮卡片
        self.action_card = self._make_card(radius=24)
        action_layout = self.action_card.layout()

        self.btn_cruise = self._make_button("多航点巡航", primary=True)
        self.btn_detect = self._make_button("实时视觉检测", primary=False)
        self.btn_stop = self._make_button("停止任务", danger=True)

        self.btn_cruise.clicked.connect(self._on_cruise_clicked)
        self.btn_detect.clicked.connect(self._on_detect_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        action_layout.addWidget(self.btn_cruise)
        action_layout.addWidget(self.btn_detect)
        action_layout.addWidget(self.btn_stop)
        panel.addWidget(self.action_card)

        self._cruise_ui = None

    def _make_card(self, radius: int = 24) -> QFrame:
        """带纤细边框的大圆角卡片。"""
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: {_hex('surface')}; border: 1px solid {_hex('border')}; border-radius: {radius}px; }}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        return card

    def _make_button(
        self, text: str, primary: bool = False, danger: bool = False
    ) -> QPushButton:
        """胶囊按钮（彻底移除聚焦虚线框 outline / focus border）。"""
        button = QPushButton(text)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFixedHeight(52)

        if danger:
            bg = _hex("warning_soft")
            text_color = _hex("warning")
            hover_bg = _hex("warning")
            hover_text = _hex("white")
        elif primary:
            bg = _hex("primary")
            text_color = _hex("white")
            hover_bg = _hex("primary_dark")
            hover_text = _hex("white")
        else:
            bg = _hex("surface_sub")
            text_color = _hex("text")
            hover_bg = _hex("border")
            hover_text = _hex("text")

        button.setStyleSheet(
            f"QPushButton {{ background: {bg}; color: {text_color}; border: none; outline: none; border-radius: 26px;"
            f" font-size: 15px; font-weight: 700; }}"
            f"QPushButton:focus {{ outline: none; border: none; }}"
            f"QPushButton:hover {{ background: {hover_bg}; color: {hover_text}; }}"
        )
        return button

    def _qss(self) -> str:
        # 全局消除聚焦蓝/黑边虚线框
        return (
            f"QMainWindow, QWidget {{ background: {_hex('bg')}; outline: none; }}"
            f"QLabel {{ background: transparent; border: none; outline: none; }}"
            f"*:focus {{ outline: none; border: none; }}"
        )

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def _request_quit(self) -> None:
        self._quit_requested = True

    def _push_message(self, text: str) -> None:
        print(f"[UI] {text}")

    def _on_cruise_clicked(self) -> None:
        ui = self._cruise_ui
        if ui is None:
            return
        if str(ui.get("flight_state", "")) in {
            "TAKING_OFF",
            "CRUISING",
            "STOPPING",
            "LANDING",
        }:
            self._push_message("任务正在执行中")
            return
        if ui.get("cruise_started", False):
            self._push_message("巡航运行中")
            return
        if not ui.get("airsim_connected", False):
            self._push_message("等待 AirSim 连接中")
            return
        ui["start_cruise"].set()
        ui["stop_cruise"].clear()
        if ui.get("airsim_ready", False):
            self._push_message("发送巡航指令，即将起飞")
        else:
            self._push_message("场景就绪后自动起飞")

    def _on_stop_clicked(self) -> None:
        ui = self._cruise_ui
        if ui is None:
            return
        if not ui.get("cruise_started", False) and str(
            ui.get("flight_state", "")
        ) not in {
            "TAKING_OFF",
            "CRUISING",
            "STOPPING",
            "LANDING",
        }:
            self._push_message("任务未在运行")
            return
        ui["stop_cruise"].set()
        self._push_message("停止巡航并准备降落…")

    def _on_detect_clicked(self) -> None:
        ui = self._cruise_ui
        if ui is None:
            return
        show = not ui.get("show_detections", True)
        ui["show_detections"] = show
        self._push_message(f"检测框显示已{'开启' if show else '关闭'}")

    # ------------------------------------------------------------------
    # 渲染与更新
    # ------------------------------------------------------------------
    def show(self, packet: Any, snapshot: Any, args: Any, ui: dict[str, Any]) -> bool:
        if args.no_display:
            return False
        self._cruise_ui = ui
        has_update = (
            packet is not None or snapshot is not None or not self._window_shown
        )
        if packet is not None:
            self.last_packet = packet
            self._frame_history.append(packet)
        if snapshot is not None:
            self.last_snapshot = snapshot

        if self.last_packet is None or self.last_snapshot is None:
            self._detection_fresh = False
        else:
            detection_id = int(self.last_snapshot.frame_id)
            detection_packet = next(
                (
                    history_packet
                    for history_packet in reversed(self._frame_history)
                    if int(history_packet.frame_id) == detection_id
                ),
                None,
            )
            # 方案 C：以“检测帧已进入显示历史”为唯一判据。
            # _frame_history 只保留最近到达的 8 帧（约 320-580ms），本身就是时间窗：
            # 未来帧不可能出现在 history 中；过期结果会随 history 滚动而滑出窗口。
            # 原来硬编码的 2 帧上限（≈80ms）在推理较慢（CPU/大 imgsz）时会让
            # 检测框和目标列表永远不显示，这里放宽为 history 覆盖的时间范围。
            self._detection_fresh = detection_packet is not None
        now = time.monotonic()
        if not has_update or (self._window_shown and now < self._next_render_at):
            return self.process_events()
        if not self._window_shown:
            self._window.resize(
                max(960, int(args.display_width)), max(600, int(args.display_height))
            )
            self._window.show()
            self._window_shown = True

        render_started = time.perf_counter()
        self._render_video(ui, save_render=int(getattr(args, "save_ui_every", 0)) > 0)
        self._update_panel(ui)
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
        if force or now - self._last_event_pump >= 1.0 / 30.0:
            QApplication.processEvents()
            self._last_event_pump = now
        return self._quit_requested

    def _render_video(self, ui: dict[str, Any], *, save_render: bool = False) -> None:
        size = self.video_label.size()
        width = max(320, size.width())
        height = max(240, size.height())
        canvas = Image.new("RGBA", (width, height), (*COLORS["bg"], 255))
        draw = ImageDraw.Draw(canvas, "RGBA")
        self._draw_preview(canvas, draw, (0, 0, width, height), ui)

        rgb = np.asarray(canvas.convert("RGB"))
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.shape[1] * 3,
            QImage.Format.Format_RGB888,
        )
        self.video_label.setPixmap(QPixmap.fromImage(qimage.copy()))
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
        self.status_dot.setStyleSheet(f"color: {_hex(color_key)}; border: none;")
        self.status_subtitle.setText(subtitle)

        index = int(ui.get("waypoint_index", 0))
        total = max(1, int(ui.get("waypoints_total", 1)))
        round_no = int(ui.get("patrol_round", 0))
        altitude = -float(ui.get("altitude", 0.0))
        speed = float(ui.get("speed", 0.0))
        detection_fps = float(ui.get("detection_fps", 0.0))
        detection_latency_ms = float(ui.get("detection_latency_ms", 0.0))

        boxes = (
            self.last_snapshot.boxes
            if self.last_snapshot is not None and self._detection_fresh
            else ()
        )
        # 小卡片赋值
        self.tile_alt.set_value(f"{altitude:.1f}")
        self.tile_spd.set_value(f"{speed:.2f}")
        self.tile_fps.set_value(f"{detection_fps:.1f}")
        self.tile_lat.set_value(f"{detection_latency_ms:.0f}")

        # 进度更新
        self.progress.setRange(0, total)
        self.progress.setValue(min(index, total))
        self.round_label.setText(f"第 {round_no} 圈 · {min(index, total)}/{total} 航点")
        self.route_widget.set_progress(total, index)

        # 目标展示 Tag (纯色充填，彻底无描边)。
        # boxes 已按新鲜度过滤：非空 = 新鲜且有目标。
        # 三态：新鲜有目标 → 标签；无快照或新鲜但无目标 → “无检测目标”；
        # 有快照但检测帧已滑出显示历史（推理过慢/长时间无新结果）→ “检测更新中”。
        if boxes:
            ordered = sorted(boxes, key=lambda box: float(box[5]), reverse=True)
            tags = []
            for box in ordered:
                name = _display_name(box[6])
                confidence = float(box[5])

                if confidence >= 0.45:
                    bg_color = _hex("primary_soft")
                    text_color = _hex("primary")
                else:
                    bg_color = _hex("warning_soft")
                    text_color = _hex("warning")

                tags.append(
                    f'<span style="background-color:{bg_color}; color:{text_color};'
                    f' padding: 4px 10px; border-radius: 12px; font-weight:600; font-size:12px;">'
                    f"{name} {confidence:.0%}</span>"
                )

            self.target_label.setText("  ".join(tags))
            self.target_label.setTextFormat(Qt.TextFormat.RichText)
            self.target_label.setWordWrap(True)
            self.target_label.setStyleSheet("border: none;")
        elif self.last_snapshot is None or self._detection_fresh:
            self.target_label.setText("当前画面无检测目标")
            self.target_label.setTextFormat(Qt.TextFormat.PlainText)
            self.target_label.setStyleSheet(
                f"color: {_hex('muted_light')}; border: none;"
            )
        else:
            self.target_label.setText("检测更新中…")
            self.target_label.setTextFormat(Qt.TextFormat.PlainText)
            self.target_label.setStyleSheet(
                f"color: {_hex('muted_light')}; border: none;"
            )
