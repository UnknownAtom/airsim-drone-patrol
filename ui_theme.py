"""Shared visual tokens and font helpers for the AirSim desktop UI."""

from __future__ import annotations

from PyQt6.QtGui import QFont, QFontDatabase


COLORS: dict[str, tuple[int, int, int]] = {
    "bg": (244, 245, 247),
    "surface": (255, 255, 255),
    "surface_sub": (238, 240, 244),
    "preview": (228, 232, 240),
    "primary": (37, 99, 235),
    "primary_soft": (219, 234, 254),
    "primary_dark": (29, 78, 216),
    "text": (30, 41, 59),
    "muted": (100, 116, 139),
    "muted_light": (148, 163, 184),
    "border": (226, 232, 240),
    "success": (22, 163, 74),
    "warning": (220, 38, 38),
    "warning_soft": (254, 226, 226),
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
