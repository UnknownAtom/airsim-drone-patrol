"""SCD-style visual frontend for the AirSim patrol program.

The window follows the visual language of a local SCD-style reference
application: a quiet light-gray workspace, a large pale-blue preview surface,
a white analysis card, blue action buttons, and compact status strips. This
module only renders the latest data; AirSim, camera capture, YOLO inference,
and keyboard handling remain in ``simu.py``.
"""

from __future__ import annotations

import ctypes
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WINDOW_TITLE = "AirSim 无人机目标检测系统"
WINDOW_NAME = "AirSim Detection Console"


def fix_window_title() -> None:
    """Set a Chinese title after HighGUI creates the ASCII-registered window."""
    try:
        import ctypes.wintypes as wt

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        process_id = kernel32.GetCurrentProcessId()

        @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def enum_window(hwnd, _lparam):
            window_pid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
            if window_pid.value != process_id:
                return True
            class_name = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_name, 256)
            if class_name.value == "Main HighGUI class":
                user32.SetWindowTextW(hwnd, WINDOW_TITLE)
            return True

        user32.EnumWindows(enum_window, 0)
    except Exception:
        # Linux/macOS or a HighGUI build without Win32 APIs: the ASCII title
        # is still valid and the rendering path continues normally.
        pass


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


# Palette copied from the reference application's visual language. The pale
# blue preview surface and dark-blue actions are the visual anchors.
COLORS: dict[str, tuple[int, int, int]] = {
    "bg": (242, 242, 247),
    "surface": (255, 255, 255),
    "preview": (206, 229, 237),
    "primary": (50, 76, 180),
    "primary_soft": (238, 241, 251),
    "text": (26, 26, 46),
    "muted": (92, 96, 112),
    "muted_light": (118, 121, 136),
    "border": (225, 226, 235),
    "soft_gray": (242, 242, 247),
    "soft_blue": (238, 241, 251),
    "success": (56, 142, 60),
    "warning": (211, 47, 47),
    "warning_soft": (252, 235, 235),
    "white": (255, 255, 255),
    "image_border": (255, 255, 255),
}

# Compatibility alias for code that may still import the old theme map. Both
# options deliberately render this same reference-style light interface.
SCHEMES = {"light": COLORS, "dark": COLORS}


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


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x2E80 <= code <= 0x9FFF
        or 0x3000 <= code <= 0x303F
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF
    )


class UIFonts:
    """Load a Chinese-first font pairing with Windows fallbacks."""

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
        # Use one Chinese-capable font for the entire run. Mixing Microsoft
        # YaHei and Bahnschrift per character makes Latin digits sit at a
        # different vertical height from Chinese glyphs in the same card.
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
    """Thread-safe short-lived messages supplied by the flight workers."""

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


class DetectionDisplay:
    """Render the SCD-inspired AirSim analysis window."""

    def __init__(self, theme: str = "light") -> None:
        self.scheme = COLORS
        self.fonts = UIFonts()
        self.last_packet: Any = None
        self.last_snapshot: Any = None
        self.last_render: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Small reusable pieces matching the reference frontend
    # ------------------------------------------------------------------
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

    def _draw_camera_icon(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        draw.rounded_rectangle(
            (x, y + 4, x + 27, y + 22), radius=4, outline=_rgba(color), width=2
        )
        draw.line(
            ((x + 27, y + 9), (x + 34, y + 6), (x + 34, y + 20), (x + 27, y + 17)),
            fill=_rgba(color),
            width=2,
            joint="curve",
        )
        draw.ellipse((x + 9, y + 8, x + 18, y + 17), outline=_rgba(color), width=2)

    def _draw_route_icon(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        points = ((x + 2, y + 19), (x + 12, y + 7), (x + 23, y + 16), (x + 33, y + 4))
        draw.line(points, fill=_rgba(color), width=2, joint="curve")
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=_rgba(color))

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

    def _info_card(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        label: str,
        value: str,
        *,
        value_color: str = "text",
    ) -> None:
        _rounded(draw, box, self.scheme["soft_gray"], 15)
        self._text(draw, (box[0] + 14, box[1] + 8), label, "small", "muted")
        self._text(draw, (box[0] + 14, box[1] + 23), value, "value", value_color)

    def _info_row(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        text: str,
        *,
        color: str = "text",
    ) -> None:
        """Draw one reference-style, single-line information card."""
        _rounded(draw, box, self.scheme["soft_gray"], 12)
        center_y = (box[1] + box[3]) // 2
        draw.ellipse(
            (box[0] + 14, center_y - 4, box[0] + 22, center_y + 4),
            fill=_rgba(self.scheme[color]),
        )
        size, weight = TYPE["card"]
        text_y = center_y - int(size * 0.48)
        self.fonts.draw(
            draw,
            (box[0] + 34, text_y),
            text,
            size,
            weight,
            _rgba(self.scheme[color]),
        )

    def _button(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        text: str,
        icon: str,
        *,
        fill: str = "primary",
    ) -> None:
        _rounded(draw, box, self.scheme[fill], 12)
        color = self.scheme["white"]
        icon_x = box[0] + 20
        icon_y = (box[1] + box[3]) // 2 - 13
        if icon == "camera":
            self._draw_camera_icon(draw, icon_x, icon_y, color)
        elif icon == "route":
            self._draw_route_icon(draw, icon_x, icon_y, color)
        else:
            draw.ellipse(
                (icon_x + 2, icon_y + 2, icon_x + 25, icon_y + 25),
                outline=_rgba(color),
                width=2,
            )
            draw.line(
                (icon_x + 10, icon_y + 10, icon_x + 17, icon_y + 17),
                fill=_rgba(color),
                width=2,
            )
            draw.line(
                (icon_x + 17, icon_y + 10, icon_x + 10, icon_y + 17),
                fill=_rgba(color),
                width=2,
            )
        self.fonts.draw(
            draw,
            (icon_x + 47, box[1] + 18),
            text,
            *TYPE["button"],
            _rgba(color),
        )

    # ------------------------------------------------------------------
    # Main content areas
    # ------------------------------------------------------------------
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
        # Keep the pale-blue media surface visible around a keep-aspect image,
        # matching the reference app instead of adding another white card.
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

        if self.last_snapshot is not None:
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
        detection_count = len(self.last_snapshot.boxes) if self.last_snapshot is not None else 0
        self._chip(draw, inner[0] + 12, inner[3] - 42, f"帧 {frame_id:06d}", fill="preview", color="primary")
        self._chip(draw, inner[0] + 128, inner[3] - 42, f"目标 {detection_count}", fill="preview", color="primary")

    def _status_values(self, ui: dict[str, Any]) -> tuple[str, str, str]:
        camera_ok = bool(ui.get("camera_ok"))
        detections = int(ui.get("detections", 0))
        waypoint = int(ui.get("waypoint_index", 0))
        if not ui.get("airsim_connected", False):
            return "等待 AirSim 信号", "muted", "未检测到 AirSim 模拟器，请先启动场景"
        if not camera_ok:
            return "待命", "muted", "等待相机连接"
        if detections > 0:
            return "● 发现目标", "warning", "正在进行目标检测"
        if waypoint > 0:
            return "● 自动巡航", "primary", "实时视觉检测中"
        return "● 监测中", "success", "相机已连接"

    def _draw_result_card(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        ui: dict[str, Any],
    ) -> None:
        """Draw the same one-column information rhythm as the reference UI."""
        x1, y1, x2, y2 = box
        _rounded(draw, box, self.scheme["surface"], 20)
        left, right = x1 + 20, x2 - 20

        self._draw_target_icon(draw, left + 15, y1 + 30, self.scheme["primary"], 0.75)
        self._text(draw, (left + 38, y1 + 19), "实时分析结果", "section", "primary")

        status, status_color, subtitle = self._status_values(ui)
        self._center_text(draw, (left, y1 + 70, right, y1 + 108), status, "status", status_color)
        self._center_text(draw, (left, y1 + 108, right, y1 + 130), subtitle, "small", "muted")

        index = int(ui.get("waypoint_index", 0))
        total = max(1, int(ui.get("waypoints_total", 1)))
        camera_ok = bool(ui.get("camera_ok"))
        camera_color = "success" if camera_ok else "muted"
        detections = int(ui.get("detections", 0))
        altitude = -float(ui.get("altitude", 0.0))
        speed = float(ui.get("speed", 0.0))

        row_left, row_right = left, right
        row_h, row_gap = 44, 9
        row_y = y1 + 145
        rows = (
            (f"相机状态：{'已连接' if camera_ok else '等待连接'}    目标：{detections} 个", camera_color),
            (f"航点进度：{min(index, total):02d} / {total:02d}    第 {int(ui.get('patrol_round', 0))} 圈", "primary"),
            (f"飞行高度：{altitude:.1f} 米    速度：{speed:.2f} 米/秒", "text"),
        )
        for text, color in rows:
            self._info_row(draw, (row_left, row_y, row_right, row_y + row_h), text, color=color)
            row_y += row_h + row_gap

        progress_y = row_y + 7
        self._text(draw, (left, progress_y), "巡航进度", "body", "muted")
        round_text = f"第 {int(ui.get('patrol_round', 0))} 圈"
        round_w = self.fonts.measure(round_text, *TYPE["small"])
        self._text(draw, (right - round_w, progress_y + 1), round_text, "small", "muted")
        track_y = progress_y + 27
        _rounded(draw, (left, track_y, right, track_y + 6), self.scheme["soft_blue"], 3)
        progress = _clip(index / total, 0.0, 1.0)
        if progress > 0:
            _rounded(draw, (left, track_y, left + int((right - left) * progress), track_y + 6), self.scheme["primary"], 3)

        route_y = track_y + 34
        self._text(draw, (left, route_y), "巡航路线", "body", "muted")
        points = list(ui.get("waypoints", ()))
        point_count = max(2, len(points) if points else total)
        line_y = route_y + 28
        line_left, line_right = left + 8, right - 8
        draw.line((line_left, line_y, line_right, line_y), fill=_rgba(self.scheme["border"]), width=3)
        for point_index in range(point_count):
            px = line_left + (line_right - line_left) * point_index / max(1, point_count - 1)
            active = point_index < index
            current = point_index == max(0, index - 1)
            fill = self.scheme["primary"] if active or current else self.scheme["surface"]
            radius = 7 if current else 5
            if current:
                draw.ellipse((px - 11, line_y - 11, px + 11, line_y + 11), outline=_rgba(self.scheme["primary"]), width=2)
            draw.ellipse((px - radius, line_y - radius, px + radius, line_y + radius), fill=_rgba(fill), outline=_rgba(self.scheme["primary"]), width=2)

        separator_y = min(y2 - 93, route_y + 65)
        draw.line((left, separator_y, right, separator_y), fill=_rgba(self.scheme["border"]), width=1)
        self._text(draw, (left, separator_y + 14), "当前目标", "body", "muted")
        count_text = str(len(self.last_snapshot.boxes) if self.last_snapshot is not None else 0)
        count_w = int(self.fonts.measure(count_text, *TYPE["small"]) + 24)
        self._chip(draw, right - count_w, separator_y + 8, count_text, fill="soft_blue", color="primary")
        detections_list = list(self.last_snapshot.boxes) if self.last_snapshot is not None else []
        if detections_list:
            target = detections_list[0]
            name = _display_name(target[6])
            target_color = "warning" if target[5] < 0.45 else "primary"
            self._text(draw, (left, separator_y + 45), f"{name}  {target[5]:.0%}", "body", target_color)
        else:
            self._text(draw, (left, separator_y + 45), "当前画面没有检测目标", "body", "muted_light")

    def _draw_action_panel(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        ui: dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = box
        self._button(draw, (x1, y1, x2, y1 + 54), "多航点巡航", "route", fill="primary")
        self._button(draw, (x1, y1 + 67, x2, y1 + 121), "实时视觉检测", "camera", fill="primary")
        self._button(draw, (x1, y1 + 134, x2, y1 + 188), "按 Q 键停止任务", "stop", fill="warning")
        self._text(draw, (x1 + 3, min(y2 - 22, y1 + 211)), "窗口操作提示：按 Q 键安全停止并降落", "small", "muted")

    def _draw_compact_status(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        ui: dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = box
        _rounded(draw, box, self.scheme["surface"], 20)
        index = int(ui.get("waypoint_index", 0))
        total = max(1, int(ui.get("waypoints_total", 1)))
        items = (
            ("相机", "在线" if ui.get("camera_ok") else "等待"),
            ("航点", f"{min(index, total)}/{total}"),
            ("高度", f"{-float(ui.get('altitude', 0.0)):.1f} m"),
            ("速度", f"{float(ui.get('speed', 0.0)):.2f} m/s"),
            ("目标", str(ui.get("detections", 0))),
        )
        item_w = (x2 - x1 - 32 - 4 * 12) / 5
        for idx, (label, value) in enumerate(items):
            x = x1 + 16 + idx * (item_w + 12)
            self._text(draw, (x, y1 + 13), label, "small", "muted")
            self._text(draw, (x, y1 + 33), value, "value", "primary" if idx in (0, 1, 4) else "text")

    def _draw_alert(self, draw: ImageDraw.ImageDraw, width: int, height: int, ui: dict[str, Any]) -> None:
        messages = ui.get("messages")
        if not isinstance(messages, UIMessages):
            return
        visible = messages.snapshot()
        if not visible:
            return
        kind, message = visible[-1]
        if kind == "info":
            return
        fill = "warning_soft" if kind == "error" else "soft_blue"
        color = "warning" if kind == "error" else "primary"
        max_width = min(width - 50, 520)
        text_width = min(self.fonts.measure(str(message), *TYPE["small"]), max_width - 50)
        box_w = int(text_width + 44)
        x, y = 25, height - 62
        _rounded(draw, (x, y, x + box_w, y + 34), self.scheme[fill], 17)
        draw.ellipse((x + 12, y + 13, x + 20, y + 21), fill=_rgba(self.scheme[color]))
        self._text(draw, (x + 29, y + 9), str(message), "small", color)

    def show(self, packet: Any, snapshot: Any, args: Any, ui: dict[str, Any]) -> bool:
        if packet is not None:
            self.last_packet = packet
        if snapshot is not None:
            self.last_snapshot = snapshot
        if args.no_display:
            return False

        width = max(960, int(args.display_width))
        height = max(600, int(args.display_height))
        canvas = Image.new("RGBA", (width, height), _rgba(self.scheme["bg"]))
        draw = ImageDraw.Draw(canvas, "RGBA")

        margin = 25 if width >= 1200 else 18
        gap = 20
        content_top = margin
        content_bottom = height - margin
        content_height = content_bottom - content_top
        content_width = width - 2 * margin

        compact = width < 1120 or height < 720
        if compact:
            status_h = 82
            video_box = (margin, content_top, width - margin, content_bottom - status_h - gap)
            status_box = (margin, content_bottom - status_h, width - margin, content_bottom)
            self._draw_preview(canvas, draw, video_box, ui)
            self._draw_compact_status(draw, status_box, ui)
        else:
            side_width = max(340, int(content_width * 0.30))
            left_width = content_width - side_width - gap
            left_box = (margin, content_top, margin + left_width, content_bottom)
            right_x = margin + left_width + gap
            result_height = int(content_height * 0.69)
            result_box = (right_x, content_top, width - margin, content_top + result_height)
            action_box = (right_x, content_top + result_height + gap, width - margin, content_bottom)
            self._draw_preview(canvas, draw, left_box, ui)
            self._draw_result_card(draw, result_box, ui)
            self._draw_action_panel(draw, action_box, ui)

        self._draw_alert(draw, width, height, ui)
        frame = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGBA2BGR)
        self.last_render = frame
        cv2.imshow(WINDOW_NAME, frame)
        return cv2.waitKey(1) & 0xFF == ord("q")

    @staticmethod
    def process_events() -> bool:
        return cv2.waitKey(1) & 0xFF == ord("q")
