"""AirSim PyQt6 前端：左侧 QPainter 视频区 + 右侧任务控制面板。

主题令牌和复用组件分别位于 ``ui_theme.py``、``ui_components.py``；本模块
负责视频帧合成、状态映射、窗口组装以及与主循环的兼容接口。

视觉规范：浅色大圆角扁平风格（纯色无渐变、大胶囊圆角、无文字描线/无聚焦虚框）。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGraphicsDropShadowEffect,
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

try:
    from qfluentwidgets import PrimaryPushButton, Theme, setTheme

    FLUENT_WIDGETS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional fallback for existing installs
    PrimaryPushButton = QPushButton
    Theme = None
    setTheme = None
    FLUENT_WIDGETS_AVAILABLE = False

from performance import RateWindow, RollingStats, StatsSnapshot
from ui_components import (
    DetectionTargetList,
    MetricTile,
    RouteWidget,
)
from ui_theme import _hex, _make_font

WINDOW_TITLE = "AirSim 无人机目标检测系统"

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


def _fit_image(source_w: int, source_h: int, box: QRectF) -> tuple[float, float, float, float, float]:
    scale = min(box.width() / max(1, source_w), box.height() / max(1, source_h))
    width = source_w * scale
    height = source_h * scale
    return (
        box.left() + (box.width() - width) / 2,
        box.top() + (box.height() - height) / 2,
        width,
        height,
        scale,
    )

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

class VideoWidget(QWidget):
    """QPainter 自绘视频区：等比图像、检测框、HUD 和底部 chips。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frame: np.ndarray | None = None
        self._boxes: list[tuple[float, float, float, float, int, float, str]] = []
        self._frame_id = 0
        self._detection_fresh = False

    def set_frame(
        self,
        frame: np.ndarray | None,
        boxes: Any,
        frame_id: int,
        detection_fresh: bool,
    ) -> None:
        self._frame = frame
        self._boxes = list(boxes or ())
        self._frame_id = int(frame_id)
        self._detection_fresh = bool(detection_fresh)
        self.update()

    def _draw_corner_bbox(
        self, painter: QPainter, box: tuple[float, float, float, float], color: str
    ) -> None:
        x1, y1, x2, y2 = box
        painter.setPen(QPen(QColor(_hex(color)), 3))
        painter.drawLine(QPointF(x1, y1), QPointF(x1 + 14, y1))
        painter.drawLine(QPointF(x1, y1), QPointF(x1, y1 + 14))
        painter.drawLine(QPointF(x2, y1), QPointF(x2 - 14, y1))
        painter.drawLine(QPointF(x2, y1), QPointF(x2, y1 + 14))
        painter.drawLine(QPointF(x1, y2), QPointF(x1 + 14, y2))
        painter.drawLine(QPointF(x1, y2), QPointF(x1, y2 - 14))
        painter.drawLine(QPointF(x2, y2), QPointF(x2 - 14, y2))
        painter.drawLine(QPointF(x2, y2), QPointF(x2, y2 - 14))

    def _draw_chip(
        self, painter: QPainter, rect: QRectF, text: str, color: str
    ) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(_hex("surface")))
        painter.drawRoundedRect(rect, 16, 16)
        painter.setFont(_make_font(12))
        painter.setPen(QColor(_hex(color)))
        painter.drawText(
            rect.adjusted(12, 0, -12, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def paintEvent(self, _event) -> None:  # noqa: N802
        width = max(320, self.width())
        height = max(240, self.height())
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        )

        painter.setPen(QPen(QColor(_hex("border")), 1))
        painter.setBrush(QColor(_hex("surface")))
        painter.drawRoundedRect(QRectF(0.5, 0.5, width - 1, height - 1), 28, 28)

        inner = QRectF(12, 12, max(1, width - 24), max(1, height - 24))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(_hex("preview")))
        painter.drawRoundedRect(inner, 20, 20)
        image_box = inner.adjusted(1, 1, -1, -1)

        if self._frame is None:
            painter.setFont(_make_font(14))
            painter.setPen(QColor(_hex("muted")))
            empty_rect = QRectF(
                inner.left(), height / 2 - 22, inner.width(), 24
            )
            painter.drawText(empty_rect, Qt.AlignmentFlag.AlignCenter, "系统就绪 · 等待相机画面")
            painter.end()
            return

        frame = np.ascontiguousarray(self._frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            painter.end()
            return
        source_h, source_w = frame.shape[:2]
        px, py, pw, ph, scale = _fit_image(source_w, source_h, image_box)
        qimage = QImage(
            frame.data,
            source_w,
            source_h,
            source_w * 3,
            QImage.Format.Format_RGB888,
        ).copy()

        image_path = QPainterPath()
        image_path.addRoundedRect(QRectF(px, py, pw, ph), 18, 18)
        painter.save()
        painter.setClipPath(image_path)
        painter.drawImage(QRectF(px, py, pw, ph), qimage)
        painter.restore()

        hud_text = f"前视监控  ·  {source_w}×{source_h}"
        painter.setFont(_make_font(12))
        hud_width = painter.fontMetrics().horizontalAdvance(hud_text) + 36
        hud_rect = QRectF(inner.left() + 16, inner.top() + 16, hud_width, 32)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(_hex("surface")))
        painter.drawRoundedRect(hud_rect, 16, 16)
        painter.setBrush(QColor(_hex("success")))
        painter.drawEllipse(
            QRectF(inner.left() + 26, inner.top() + 29, 6, 6)
        )
        painter.setPen(QColor(_hex("text")))
        painter.drawText(QPointF(inner.left() + 40, inner.top() + 23), hud_text)

        if self._detection_fresh and self._boxes:
            painter.setFont(_make_font(12))
            metrics = painter.fontMetrics()
            for xmin, ymin, xmax, ymax, _class_id, confidence, name in self._boxes:
                accent = "primary" if confidence >= 0.45 else "warning"
                left = _clip(px + xmin * scale, image_box.left(), image_box.right())
                top = _clip(py + ymin * scale, image_box.top(), image_box.bottom())
                right = _clip(px + xmax * scale, image_box.left(), image_box.right())
                bottom = _clip(py + ymax * scale, image_box.top(), image_box.bottom())
                if right <= left or bottom <= top:
                    continue
                self._draw_corner_bbox(painter, (left, top, right, bottom), accent)
                label = f"{_display_name(name)} {confidence:.0%}"
                label_height = 24
                label_width = metrics.horizontalAdvance(label) + 18
                label_y = top - label_height - 4
                if label_y < image_box.top():
                    label_y = bottom + 4
                label_y = _clip(
                    label_y, image_box.top(), image_box.bottom() - label_height
                )
                # 标签水平方向同样钳制进图像区，避免靠近左/右边缘时溢出控件。
                label_x = _clip(
                    left, image_box.left(), image_box.right() - label_width
                )
                label_width = min(label_width, image_box.right() - label_x)
                label_rect = QRectF(label_x, label_y, label_width, label_height)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(_hex("surface")))
                painter.drawRoundedRect(label_rect, 12, 12)
                painter.setPen(QColor(_hex(accent)))
                painter.drawText(
                    QRectF(
                        label_x + 9,
                        label_y,
                        max(1, label_width - 9),
                        label_height,
                    ),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )

        detection_count = len(self._boxes) if self._detection_fresh else 0
        self._draw_chip(
            painter,
            QRectF(inner.left() + 16, inner.bottom() - 48, 106, 32),
            f"帧 {self._frame_id:06d}",
            "text",
        )
        self._draw_chip(
            painter,
            QRectF(inner.left() + 130, inner.bottom() - 48, 106, 32),
            f"目标 {detection_count}",
            "primary",
        )
        painter.end()

class _PilRenderer:
    """UI 状态适配与渲染统计基类，保留原有外部兼容接口。"""

    def __init__(self) -> None:
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
        self._ui_lock = threading.Lock()

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
            return "系统待命", "primary", "等待自动起飞"
        if not camera_ok:
            camera_error = str(ui.get("camera_error", "")).strip()
            return (
                "待命",
                "muted",
                f"相机连接失败：{camera_error}" if camera_error else "等待相机",
            )
        if self.last_snapshot is not None and self.last_snapshot.boxes:
            return "发现目标", "warning", "实时目标视觉识别中"
        if waypoint > 0:
            return "自动巡航", "primary", "多航点路线巡航中"
        return "监测中", "success", "设备运行正常"


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
        self._window.setMinimumSize(960, 600)
        self._window.closed.connect(self._request_quit)
        QShortcut(QKeySequence("Q"), self._window).activated.connect(self._request_quit)
        self._window.setStyleSheet(self._qss())
        if FLUENT_WIDGETS_AVAILABLE and setTheme is not None and Theme is not None:
            setTheme(Theme.LIGHT)

        root = QWidget()
        self._window.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(14)

        # ---- 顶部应用栏 ----
        root_layout.addWidget(self._build_app_bar())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(20)

        # ---- 左侧视频区 ----
        self.video_widget = VideoWidget()
        body_layout.addWidget(self.video_widget, stretch=7)

        # ---- 右侧面板 ----
        self._build_panel()
        self.right_panel.setMinimumWidth(360)
        self.right_panel_scroll = QScrollArea()
        self.right_panel_scroll.setWidgetResizable(True)
        self.right_panel_scroll.setMinimumWidth(376)
        self.right_panel_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.right_panel_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.right_panel_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.right_panel_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: transparent; width: 6px; margin: 4px 0; }"
            f"QScrollBar::handle:vertical {{ background: {_hex('border')}; border-radius: 3px; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        self.right_panel_scroll.setWidget(self.right_panel)
        body_layout.addWidget(self.right_panel_scroll, stretch=3)
        root_layout.addWidget(body, stretch=1)

    def _build_app_bar(self) -> QFrame:
        """构建统一的 Fluent 风格应用栏和任务标题。"""
        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame {{ background: {_hex('surface')}; border: 1px solid {_hex('border')}; border-radius: 20px; }}"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(12)

        mark = QLabel("AIRSIM")
        mark.setFont(_make_font(12, 700, latin=True))
        mark.setStyleSheet(
            f"color: {_hex('primary')}; background: {_hex('primary_soft')}; "
            "border: none; border-radius: 10px; padding: 6px 10px;"
        )
        layout.addWidget(mark)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("无人机目标检测控制台")
        title.setFont(_make_font(16, 700))
        title.setStyleSheet(f"color: {_hex('text')}; border: none;")
        subtitle = QLabel("实时巡航 · 视觉识别 · 任务监控")
        subtitle.setFont(_make_font(11))
        subtitle.setStyleSheet(f"color: {_hex('muted')}; border: none;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)
        layout.addStretch(1)

        mode = QLabel("实时任务")
        mode.setFont(_make_font(11, 700))
        mode.setStyleSheet(
            f"color: {_hex('success')}; background: {_hex('surface_sub')}; "
            "border: none; border-radius: 10px; padding: 7px 12px;"
        )
        layout.addWidget(mode)
        self._apply_card_shadow(bar)
        return bar

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

        # 2. 核心性能指标 2x3 网格小卡片
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.tile_capture = MetricTile("采集帧率", "fps")
        self.tile_rpc = MetricTile("取图耗时", "ms")
        self.tile_inference = MetricTile("推理帧率", "fps")
        self.tile_yolo = MetricTile("YOLO 耗时", "ms")
        self.tile_gui = MetricTile("GUI 帧率", "fps")
        self.tile_drops = MetricTile("丢帧（相机/检测）")

        grid.addWidget(self.tile_capture, 0, 0)
        grid.addWidget(self.tile_rpc, 0, 1)
        grid.addWidget(self.tile_inference, 1, 0)
        grid.addWidget(self.tile_yolo, 1, 1)
        grid.addWidget(self.tile_gui, 2, 0)
        grid.addWidget(self.tile_drops, 2, 1)
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
        self.target_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.target_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: transparent; width: 4px; margin: 0; }"
            f"QScrollBar::handle:vertical {{ background: {_hex('border')}; border-radius: 2px; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        self.target_scroll.setFixedHeight(166)
        self.target_list = DetectionTargetList(tuple(DISPLAY_NAME_MAP.values()))
        self.target_scroll.setWidget(self.target_list)
        target_layout.addWidget(self.target_scroll)
        panel.addWidget(self.target_card)

        panel.addStretch(1)

        # 5. 操作按钮卡片
        self.action_card = self._make_card(radius=24)
        action_layout = self.action_card.layout()

        self.btn_stop = self._make_button("停止任务", danger=True)

        self.btn_stop.clicked.connect(self._on_stop_clicked)

        action_layout.addWidget(self.btn_stop)
        panel.addWidget(self.action_card)

        self._cruise_ui = None

    def _make_card(self, radius: int = 24) -> QFrame:
        """带纤细边框的大圆角卡片。"""
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: {_hex('surface')}; border: 1px solid {_hex('border')}; border-radius: {radius}px; }}"
        )
        self._apply_card_shadow(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        return card

    @staticmethod
    def _apply_card_shadow(widget: QFrame) -> None:
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(30, 41, 59, 18))
        widget.setGraphicsEffect(shadow)

    def _make_button(
        self, text: str, primary: bool = False, danger: bool = False
    ) -> QPushButton:
        """胶囊按钮（彻底移除聚焦虚线框 outline / focus border）。"""
        button = PrimaryPushButton(text) if FLUENT_WIDGETS_AVAILABLE else QPushButton(text)
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
            f"QPushButton:pressed {{ background: {hover_bg}; padding-top: 2px; }}"
            f"QPushButton:disabled {{ background: {_hex('surface_sub')}; color: {_hex('muted_light')}; }}"
        )
        return button

    def _qss(self) -> str:
        # 全局消除聚焦蓝/黑边虚线框
        return (
            f"QMainWindow, QWidget {{ background: {_hex('bg')}; outline: none; }}"
            f"QLabel {{ background: transparent; border: none; outline: none; }}"
            f"QToolTip {{ background: {_hex('text')}; color: {_hex('white')}; border: none; padding: 6px 8px; }}"
            f"*:focus {{ outline: none; border: none; }}"
        )

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def _request_quit(self) -> None:
        self._quit_requested = True

    def _push_message(self, text: str) -> None:
        print(f"[UI] {text}")

    def _on_stop_clicked(self) -> None:
        ui = self._cruise_ui
        if ui is None:
            return
        with self._ui_lock:
            cruise_started = ui.get("cruise_started", False)
            flight_state = str(ui.get("flight_state", ""))
        if not cruise_started and flight_state not in {
            "TAKING_OFF",
            "CRUISING",
            "STOPPING",
            "LANDING",
        }:
            self._push_message("任务未在运行")
            return
        ui["stop_cruise"].set()  # Event 线程安全，无需持锁
        self._push_message("停止巡航并准备降落…")

    # ------------------------------------------------------------------
    # 渲染与更新
    # ------------------------------------------------------------------
    def show(
        self,
        packet: Any,
        snapshot: Any,
        args: Any,
        ui: dict[str, Any],
        state_lock: threading.Lock | None = None,
    ) -> bool:
        if args.no_display:
            return False
        if state_lock is not None:
            self._ui_lock = state_lock
        self._cruise_ui = ui
        has_update = (
            packet is not None or snapshot is not None or not self._window_shown
        )
        if packet is not None:
            self.last_packet = packet
            self._frame_history.append(packet)
        if snapshot is not None:
            self.last_snapshot = snapshot

        # 检测框必须画在“与结果同源”的帧上，否则飞行器移动时会把旧帧的框
        # 叠加到新帧上造成错位。策略：快照存在且其帧仍在显示历史内时，回退
        # 显示“被检测帧”并叠加框（自洽，但画面最多滞后 history 覆盖的时间窗）；
        # 快照帧已滑出历史则显示最新帧、不叠加（该结果已过期）。
        if self.last_snapshot is None:
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
            if detection_packet is not None:
                self.last_packet = detection_packet
                self._detection_fresh = True
            else:
                self._detection_fresh = False
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
        boxes = (
            self.last_snapshot.boxes
            if self.last_snapshot is not None and self._detection_fresh
            else ()
        )
        self.video_widget.set_frame(
            self.last_packet.frame if self.last_packet is not None else None,
            boxes,
            self.last_packet.frame_id if self.last_packet is not None else 0,
            self._detection_fresh,
        )
        self._update_panel(ui)
        if int(getattr(args, "save_ui_every", 0)) > 0:
            self._capture_render()
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

    def _capture_render(self) -> None:
        """Capture the painted video widget for the existing save-ui contract."""
        if not self._window_shown:
            self.last_render = None
            return
        self.video_widget.repaint()
        pixmap = self.video_widget.grab()
        if pixmap.isNull():
            self.last_render = None
            return
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        if image.isNull():
            self.last_render = None
            return
        height, width = image.height(), image.width()
        row_bytes = image.bytesPerLine()
        byte_count = height * row_bytes
        pixels = image.constBits()
        pixels.setsize(byte_count)
        rgb = np.frombuffer(pixels, dtype=np.uint8, count=byte_count)
        if rgb.size != byte_count or row_bytes < width * 3:
            self.last_render = None
            return
        rgb = rgb.reshape(height, row_bytes)[:, : width * 3]
        rgb = rgb.reshape(height, width, 3).copy()
        self.last_render = rgb[:, :, ::-1].copy()

    @property
    def render_fps(self) -> float:
        return self._render_rate.rate()

    @property
    def render_performance(self) -> StatsSnapshot:
        return self.render_stats.snapshot()

    def _update_panel(self, ui: dict[str, Any]) -> None:
        # 在工作线程持锁写入的同一把锁下取一次性快照，避免 GUI 侧无锁读竞态。
        with self._ui_lock:
            ui = dict(ui)
        status, color_key, subtitle = self._status_values(ui)
        self.status_title.setText(status)
        self.status_dot.setStyleSheet(f"color: {_hex(color_key)}; border: none;")
        self.status_subtitle.setText(subtitle)

        index = int(ui.get("waypoint_index", 0))
        total = max(1, int(ui.get("waypoints_total", 1)))
        round_no = int(ui.get("patrol_round", 0))
        capture_fps = float(ui.get("capture_fps", 0.0))
        capture_rpc_avg_ms = float(ui.get("capture_rpc_avg_ms", 0.0))
        capture_rpc_max_ms = float(ui.get("capture_rpc_max_ms", 0.0))
        detection_fps = float(ui.get("detection_fps", 0.0))
        detection_avg_ms = float(ui.get("detection_avg_ms", 0.0))
        detection_max_ms = float(ui.get("detection_max_ms", 0.0))
        render_fps = float(ui.get("render_fps", 0.0))
        camera_drops = int(ui.get("camera_drops", 0))
        detection_drops = int(ui.get("detection_drops", 0))

        # 目标列表展示最近一次检测结果（与当前是否叠加框无关）。
        boxes = self.last_snapshot.boxes if self.last_snapshot is not None else ()
        # 小卡片赋值
        self.tile_capture.set_value(f"{capture_fps:.1f}")
        self.tile_rpc.set_value(f"{capture_rpc_avg_ms:.0f}/{capture_rpc_max_ms:.0f}")
        self.tile_inference.set_value(f"{detection_fps:.1f}")
        self.tile_yolo.set_value(f"{detection_avg_ms:.0f}/{detection_max_ms:.0f}")
        self.tile_gui.set_value(f"{render_fps:.1f}")
        self.tile_drops.set_value(f"{camera_drops}/{detection_drops}")

        # 进度更新
        self.progress.setRange(0, total)
        self.progress.setValue(min(index, total))
        self.round_label.setText(f"第 {round_no} 圈 · {min(index, total)}/{total} 航点")
        self.route_widget.set_progress(total, index)

        # 目标展示固定类别卡片；只按限频策略更新数量和置信度进度。
        display_boxes = [(*box[:6], _display_name(box[6])) for box in boxes]
        self.target_list.set_targets(display_boxes)
