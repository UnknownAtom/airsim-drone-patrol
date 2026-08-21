"""PyQt6 套壳前端：左侧 PIL 视频区 + 右侧 Qt Widgets 面板。

架构（与 ui.py 的接口保持一致，simu.py 无需修改调用方式）：

- 左侧视频区：复用 ``ui.py`` 的 PIL 渲染逻辑（检测框、HUD、chips），
  渲染结果转为 ``QPixmap`` 显示在 ``QLabel`` 上；
- 右侧面板：全部使用 Qt Widgets（状态区、信息卡片、进度条、航点路线、
  目标区、操作按钮、紧凑状态条）；
- ``show(packet, snapshot, args, ui) -> bool`` 与 ``process_events()``
  签名/语义不变：返回 True 表示用户请求退出（Q 键或关闭窗口）。

视觉规范（SCD 风格）：浅灰背景 #F2F2F7、白卡片圆角 12px、深蓝主色
#324CB4、进度条淡蓝轨道 #EEF1FB、成功绿 / 警告红语义色。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui import (
    COLORS,
    UIMessages,
    DetectionDisplay as PilDetectionDisplay,
    _display_name,
)

WINDOW_TITLE = "AirSim 无人机目标检测系统"

# ---------------------------------------------------------------------------
# 颜色与字体辅助
# ---------------------------------------------------------------------------


def _hex(name: str) -> str:
    """COLORS 中的 RGB 三元组 -> '#RRGGBB'（QSS 用）。"""
    r, g, b = COLORS[name]
    return f"#{r:02X}{g:02X}{b:02X}"


def _hover_hex(name: str, factor: float = 1.1) -> str:
    """按比例提亮后的颜色（按钮 hover 用）。"""
    r, g, b = COLORS[name]
    r, g, b = (min(255, int(v * factor)) for v in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def _pick_font(candidates: tuple[str, ...]) -> str:
    families = set(QFontDatabase.families())
    for name in candidates:
        if name in families:
            return name
    return candidates[-1]


_font_names_cache: tuple[str, str] | None = None


def _font_names() -> tuple[str, str]:
    """懒加载字体名：QFontDatabase 需要 QApplication 已存在，
    因此不能在模块导入时调用。"""
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
# 自定义控件
# ---------------------------------------------------------------------------


class RouteWidget(QWidget):
    """航点路线图：水平线 + 圆点标记，当前航点高亮（与原 PIL 版本一致）。"""

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

        # 基线
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


class DetectionDisplay(PilDetectionDisplay):
    """SCD 风格 Qt 控制台：左侧 PIL 视频区 + 右侧 Qt Widgets 面板。

    复用 ``ui.py`` 的 PIL 渲染（``_draw_preview``：检测框、HUD、chips）、
    状态计算（``_status_values``）与字体/配色体系；右侧信息面板全部由
    Qt Widgets 实现。
    """

    # 布局阈值：低于该尺寸时切换为底部紧凑状态条
    COMPACT_MIN_WIDTH = 1120
    COMPACT_MIN_HEIGHT = 720

    def __init__(self, theme: str = "light") -> None:
        super().__init__(theme)  # 复用配色/字体/状态
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
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(14)

        # ---- 左侧视频区 ----
        self.video_label = QLabel()
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"background: {_hex('preview')}; border-radius: 12px;")
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
        panel.setSpacing(12)

        # 状态标题区
        self.status_title = QLabel("● 监测中")
        self.status_title.setFont(_make_font(26, 700))
        self.status_title.setStyleSheet(f"color: {_hex('success')};")
        self.status_subtitle = QLabel("相机已连接")
        self.status_subtitle.setFont(_make_font(13))
        self.status_subtitle.setStyleSheet(f"color: {_hex('muted')};")
        panel.addWidget(self.status_title)
        panel.addWidget(self.status_subtitle)

        # 信息卡片（3 行）
        self.card_camera = self._make_card()
        self.card_waypoint = self._make_card()
        self.card_flight = self._make_card()
        panel.addWidget(self.card_camera)
        panel.addWidget(self.card_waypoint)
        panel.addWidget(self.card_flight)

        # 巡航进度
        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)
        progress_caption = QLabel("巡航进度")
        progress_caption.setFont(_make_font(13))
        progress_caption.setStyleSheet(f"color: {_hex('muted')};")
        self.round_label = QLabel("第 0 圈")
        self.round_label.setFont(_make_font(13))
        self.round_label.setStyleSheet(f"color: {_hex('muted')};")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(14)
        self.progress.setStyleSheet(
            f"QProgressBar {{ background: {_hex('primary_soft')}; border: none;"
            f" border-radius: 3px; color: {_hex('muted')}; font-size: 10px; }}"
            f"QProgressBar::chunk {{ background: {_hex('primary')}; border-radius: 3px; }}"
        )
        progress_row.addWidget(progress_caption)
        progress_row.addWidget(self.progress, stretch=1)
        progress_row.addWidget(self.round_label)
        panel.addLayout(progress_row)

        # 航点路线
        self.route_widget = RouteWidget()
        panel.addWidget(self.route_widget)

        # 当前目标
        self.target_label = QLabel("当前画面没有检测目标")
        self.target_label.setFont(_make_font(15))
        self.target_label.setStyleSheet(
            f"background: {_hex('soft_gray')}; border-radius: 10px; padding: 12px;"
            f" color: {_hex('muted_light')};"
        )
        panel.addWidget(self.target_label)

        panel.addStretch(1)

        # 操作按钮（预留 clicked 信号，后续可绑定实际功能）
        self.btn_cruise = self._make_button("多航点巡航")
        self.btn_detect = self._make_button("实时视觉检测")
        self.btn_stop = self._make_button("按 Q 键停止任务", danger=True)
        self.btn_cruise.clicked.connect(lambda: self._on_action("cruise"))
        self.btn_detect.clicked.connect(lambda: self._on_action("detect"))
        self.btn_stop.clicked.connect(self._request_quit)
        panel.addWidget(self.btn_cruise)
        panel.addWidget(self.btn_detect)
        panel.addWidget(self.btn_stop)

        hint = QLabel("窗口操作提示：按 Q 键安全停止并降落")
        hint.setFont(_make_font(12))
        hint.setStyleSheet(f"color: {_hex('muted')};")
        panel.addWidget(hint)

    def _make_card(self) -> QLabel:
        label = QLabel("—")
        label.setFont(_make_font(15))
        label.setStyleSheet(
            f"background: {_hex('soft_gray')}; border-radius: 12px; padding: 14px;"
            f" color: {_hex('text')};"
        )
        return label

    def _make_button(self, text: str, danger: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFixedHeight(46)
        if danger:
            style = (
                f"QPushButton {{ background: {_hex('warning')}; color: white;"
                f" border: none; border-radius: 8px; font-size: 15px; font-weight: 700; }}"
                f"QPushButton:hover {{ background: {_hover_hex('warning')}; }}"
                f"QPushButton:pressed {{ background: {_hex('warning')}; }}"
            )
        else:
            style = (
                f"QPushButton {{ background: {_hex('primary')}; color: white;"
                f" border: none; border-radius: 8px; font-size: 15px; font-weight: 700; }}"
                f"QPushButton:hover {{ background: {_hover_hex('primary')}; }}"
                f"QPushButton:pressed {{ background: {_hex('primary')}; }}"
            )
        button.setStyleSheet(style)
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

    def _on_action(self, action: str) -> None:
        """操作按钮预留接口：后续在此绑定巡航/检测控制。"""
        print(f"[UI] 按钮动作（预留）: {action}")

    # ------------------------------------------------------------------
    # 对外接口（与 ui.py 语义一致）
    # ------------------------------------------------------------------
    def show(self, packet: Any, snapshot: Any, args: Any, ui: dict[str, Any]) -> bool:
        if args.no_display:
            return False
        if packet is not None:
            self.last_packet = packet
        if snapshot is not None:
            self.last_snapshot = snapshot
        if not self._window_shown:
            self._window.resize(max(960, int(args.display_width)), max(600, int(args.display_height)))
            self._window.show()
            self._window_shown = True

        self._render_video(ui)
        self._update_panel(ui)
        self._update_compact()
        QApplication.processEvents()
        return self._quit_requested

    def process_events(self) -> bool:
        QApplication.processEvents()
        return self._quit_requested

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def _render_video(self, ui: dict[str, Any]) -> None:
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

        # 供 --save-ui-every 保存界面帧
        self.last_render = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _update_panel(self, ui: dict[str, Any]) -> None:
        status, color_key, subtitle = self._status_values(ui)
        self.status_title.setText(status)
        self.status_title.setStyleSheet(f"color: {_hex(color_key)};")
        self.status_subtitle.setText(subtitle)

        camera_ok = bool(ui.get("camera_ok"))
        detections = int(ui.get("detections", 0))
        index = int(ui.get("waypoint_index", 0))
        total = max(1, int(ui.get("waypoints_total", 1)))
        round_no = int(ui.get("patrol_round", 0))
        altitude = -float(ui.get("altitude", 0.0))
        speed = float(ui.get("speed", 0.0))

        self.card_camera.setText(f"相机状态：{'已连接' if camera_ok else '等待连接'}    目标：{detections} 个")
        self.card_waypoint.setText(f"航点进度：{min(index, total):02d} / {total:02d}    第 {round_no} 圈")
        self.card_flight.setText(f"飞行高度：{altitude:.1f} 米    速度：{speed:.2f} 米/秒")

        self.progress.setRange(0, total)
        self.progress.setValue(min(index, total))
        self.round_label.setText(f"第 {round_no} 圈")
        self.route_widget.set_progress(total, index)

        boxes = self.last_snapshot.boxes if self.last_snapshot is not None else ()
        if boxes:
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
            self.target_label.setStyleSheet(
                f"background: {_hex('soft_gray')}; border-radius: 10px; padding: 12px;"
            )
        else:
            self.target_label.setText("当前画面没有检测目标")
            self.target_label.setTextFormat(Qt.TextFormat.PlainText)
            self.target_label.setWordWrap(False)
            self.target_label.setStyleSheet(
                f"background: {_hex('soft_gray')}; border-radius: 10px; padding: 12px;"
                f" color: {_hex('muted_light')};"
            )

        self.status_strip.set_values(ui)

    def _update_compact(self) -> None:
        """窗口过窄/过矮时切换为底部紧凑状态条。"""
        compact = (
            self._window.width() < self.COMPACT_MIN_WIDTH
            or self._window.height() < self.COMPACT_MIN_HEIGHT
        )
        self.right_panel.setVisible(not compact)
        self.status_strip.setVisible(compact)
