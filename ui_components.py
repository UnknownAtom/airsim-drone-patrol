"""Reusable Qt components for the AirSim monitoring console."""

from __future__ import annotations

import time
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui_theme import _hex, _make_font

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
        self._last_value: str | None = None

        layout.addWidget(self.label_title)
        layout.addWidget(self.label_val)

    def set_value(self, val_str: str) -> None:
        value = f"{val_str} {self._unit}" if self._unit else val_str
        if value != self._last_value:
            self.label_val.setText(value)
            self._last_value = value


class DetectionTargetCard(QFrame):
    """固定类别目标卡片；运行中只更新数量和置信度进度。"""

    def __init__(self, name: str, count: int, confidence: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background: {_hex('surface_sub')}; border: none; border-radius: 14px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.title_label = QLabel(name)
        self.title_label.setFont(_make_font(12, 700))
        self.title_label.setStyleSheet(f"color: {_hex('text')}; border: none;")
        self.count_label = QLabel()
        self.count_label.setFont(_make_font(11, 700, latin=True))
        self.count_label.setStyleSheet(f"color: {_hex('muted')}; border: none;")
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.count_label)
        layout.addLayout(header)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setFixedHeight(4)
        layout.addWidget(self.confidence_bar)
        self.confidence_bar.setStyleSheet(
            f"QProgressBar {{ background: {_hex('border')}; border: none; border-radius: 2px; }}"
            f"QProgressBar::chunk {{ background: {_hex('primary')}; border-radius: 2px; }}"
        )
        self._last_count = -1
        self._last_progress = -1
        self.set_values(count, confidence)

    def set_values(self, count: int, confidence: float) -> None:
        confidence = max(0.0, min(1.0, float(confidence)))
        count = int(count)
        progress = round(confidence * 100)
        if count != self._last_count:
            self.count_label.setText(f"{count} 个")
            self._last_count = count
        if progress != self._last_progress:
            self.confidence_bar.setValue(progress)
            self._last_progress = progress


class DetectionTargetList(QWidget):
    """固定类别检测目标列表，限频更新且不在运行中重排/重建卡片。"""

    def __init__(self, categories: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._categories = tuple(categories)
        self._cards: dict[str, DetectionTargetCard] = {}
        self._next_update_at = 0.0
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 4, 0)
        self._layout.setSpacing(8)
        for name in self._categories:
            card = DetectionTargetCard(name, 0, 0.0, self)
            self._cards[name] = card
            self._layout.addWidget(card)
        self._layout.addStretch(1)

    def set_targets(self, boxes: Any, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._next_update_at:
            return
        grouped: dict[str, list[float]] = {}
        for box in boxes or ():
            grouped.setdefault(str(box[6]), []).append(float(box[5]))
        for name, card in self._cards.items():
            values = grouped.get(name, ())
            card.set_values(len(values), max(values, default=0.0))
        self._next_update_at = now + 0.25
