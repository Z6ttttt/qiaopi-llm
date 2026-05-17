#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pipeline/render_cover.py

封面渲染模块。

输入：
    cover_fields = {
        "right_text": "汕头埠慈亲大人收",
        "center_text": "家慈亲大人安启",
        "left_text": "外付银二十元男阿明寄"
    }

输出：
    cover.png

说明：
    本文件由早期 Windows 测试脚本 cover.py 改造而来。
    原脚本已经包含竖排文字、随机笔迹扰动、墨迹效果、字段 bbox 等逻辑。
    这里主要做了工程化：
    1. 去掉 Windows 硬编码路径；
    2. 改成可被 pipeline 调用的 render_cover() 函数；
    3. 默认使用项目路径：
       data/cover_template.png
       fonts/MasaFont-Regular.ttf
"""

import math
import random
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from pipeline.text_convert import clean_render_text




def to_traditional_for_render(text: str) -> str:
    text = str(text or "")
    if _OPENCC is not None:
        return _OPENCC.convert(text)
    return text.translate(S2T_FALLBACK_MAP)



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVER_TEMPLATE_PATH = PROJECT_ROOT / "data" / "cover_template.png"
DEFAULT_FONT_PATH = PROJECT_ROOT / "fonts" / "MasaFont-Regular.ttf"


# 原 Windows 测试脚本中的 bbox 以约 440x995 的模板调参。
# 如果服务器模板尺寸不同，会按比例缩放 bbox。
DESIGN_SIZE = (440, 995)


COVER_CFG = {
    "fields": {
        "left_text": {
            "bbox": [10, 25, 100, 970],
            "font_size": 80,
        },
        "center_text": {
            "bbox": [122, 25, 312, 970],
            "font_size": 100,
        },
        "right_text": {
            "bbox": [332, 25, 430, 970],
            "font_size": 80,
        },
    }
}


BASE_STYLE = {
    "size_jitter": 2,
    "x_jitter": 3,
    "y_jitter": 4,
    "rotation_jitter": 2.5,
    "char_spacing": 7,
    "col_spacing": 18,
    "blur_prob": 0.18,
    "dilate_prob": 0.08,
    "erode_prob": 0.08,
    "x_offset": -10,
}


FIELD_STYLES = {
    "left_text": {
        "weight_mode": "light",
        "ink_gray_range": (25, 55),
        "opacity_range": (180, 225),
        "x_jitter": 4,
        "y_jitter": 4,
        "rotation_jitter": 2.8,
        "dilate_prob": 0.03,
        "erode_prob": 0.14,
        "x_offset": -7,
    },
    "center_text": {
        "weight_mode": "medium",
        "ink_gray_range": (15, 40),
        "opacity_range": (215, 250),
        "x_jitter": 2,
        "y_jitter": 3,
        "rotation_jitter": 1.8,
        "blur_prob": 0.12,
        "dilate_prob": 0.12,
        "erode_prob": 0.04,
        "x_offset": -40,
    },
    "right_text": {
        "weight_mode": "regular",
        "ink_gray_range": (18, 45),
        "opacity_range": (195, 235),
        "x_jitter": 3,
        "y_jitter": 4,
        "rotation_jitter": 2.4,
        "dilate_prob": 0.06,
        "erode_prob": 0.08,
        "x_offset": -10,
    },
}


def get_field_style(field_name: str) -> dict:
    style = BASE_STYLE.copy()
    style.update(FIELD_STYLES.get(field_name, {}))
    return style


def ensure_rgba(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGBA")
    return img


def crop_transparent(img: Image.Image) -> Image.Image:
    bbox = img.getbbox()
    if bbox is None:
        return img
    return img.crop(bbox)


def clean_text(text: str) -> str:
    return clean_render_text(text)


def load_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not font_path.exists():
        raise FileNotFoundError(f"找不到字体文件: {font_path}")
    return ImageFont.truetype(str(font_path), size)


def scale_bbox(bbox, img_w: int, img_h: int):
    design_w, design_h = DESIGN_SIZE
    sx = img_w / design_w
    sy = img_h / design_h
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * sx),
        int(y1 * sy),
        int(x2 * sx),
        int(y2 * sy),
    ]


def scale_font_size(size: int, img_h: int) -> int:
    _, design_h = DESIGN_SIZE
    return max(24, int(size * img_h / design_h))


def apply_ink_effect(glyph: Image.Image, style: dict) -> Image.Image:
    if random.random() < style["blur_prob"]:
        glyph = glyph.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.55)))

    if random.random() < style["dilate_prob"]:
        glyph = glyph.filter(ImageFilter.MaxFilter(size=3))

    if random.random() < style["erode_prob"]:
        glyph = glyph.filter(ImageFilter.MinFilter(size=3))

    return glyph


def random_ink_color(style: dict):
    ink = random.randint(*style["ink_gray_range"])
    opacity = random.randint(*style["opacity_range"])

    # 稍微带一点暖墨色，而不是纯黑
    r = min(ink + random.randint(1, 8), 255)
    g = min(ink + random.randint(0, 5), 255)
    b = max(ink - random.randint(0, 3), 0)

    return (r, g, b, opacity)


def draw_text_once(draw: ImageDraw.ImageDraw, pos, ch: str, font, fill):
    draw.text(pos, ch, font=font, fill=fill)


def render_single_char(ch: str, font_size: int, style: dict, font_path: Path) -> Image.Image:
    font = load_font(font_path, font_size)

    canvas_size = font_size * 4
    glyph = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(glyph)

    bbox = draw.textbbox((0, 0), ch, font=font)
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    x = (canvas_size - char_w) // 2 - bbox[0]
    y = (canvas_size - char_h) // 2 - bbox[1]

    fill = random_ink_color(style)
    weight_mode = style.get("weight_mode", "regular")

    if weight_mode == "light":
        draw_text_once(draw, (x, y), ch, font, fill)

    elif weight_mode == "regular":
        draw_text_once(draw, (x, y), ch, font, fill)
        if random.random() < 0.20:
            fill2 = (fill[0], fill[1], fill[2], max(fill[3] - 45, 120))
            draw_text_once(draw, (x + 1, y), ch, font, fill2)

    elif weight_mode == "medium":
        draw_text_once(draw, (x, y), ch, font, fill)
        fill2 = (fill[0], fill[1], fill[2], max(fill[3] - 35, 150))
        draw_text_once(draw, (x + 1, y), ch, font, fill2)

    else:
        draw_text_once(draw, (x, y), ch, font, fill)

    glyph = crop_transparent(glyph)

    angle = random.uniform(-style["rotation_jitter"], style["rotation_jitter"])
    glyph = glyph.rotate(
        angle,
        expand=True,
        resample=Image.Resampling.BICUBIC,
    )

    glyph = apply_ink_effect(glyph, style)
    glyph = crop_transparent(glyph)

    return glyph


def fit_font_size(text: str, bbox, base_font_size: int, style: dict):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    size = base_font_size
    while size >= 24:
        step_y = size + style["char_spacing"]
        rows_per_col = max(1, int((height - 10) / step_y))
        cols = math.ceil(len(text) / rows_per_col)
        required_width = cols * (size + style["col_spacing"])

        if required_width <= width:
            return size, rows_per_col

        size -= 2

    size = 24
    step_y = size + style["char_spacing"]
    rows_per_col = max(1, int((height - 10) / step_y))
    return size, rows_per_col


def draw_vertical_text(
    base: Image.Image,
    text: str,
    bbox,
    base_font_size: int,
    field_name: str,
    font_path: Path,
):
    text = clean_text(text)
    if not text:
        return

    style = get_field_style(field_name)
    x1, y1, x2, y2 = bbox

    font_size, rows_per_col = fit_font_size(text, bbox, base_font_size, style)

    columns = [
        text[i:i + rows_per_col]
        for i in range(0, len(text), rows_per_col)
    ]

    for col_idx, col_text in enumerate(columns):
        # 从右往左排列
        col_center_x = x2 - font_size / 2 - 4 - col_idx * (font_size + style["col_spacing"])

        col_dx = random.randint(-1, 1)
        col_dy = random.randint(-2, 2)
        curve_strength = random.uniform(-0.08, 0.08)

        for row_idx, ch in enumerate(col_text):
            actual_size = font_size + random.randint(-style["size_jitter"], style["size_jitter"])
            actual_size = max(24, actual_size)

            glyph = render_single_char(ch, actual_size, style, font_path=font_path)

            jitter_x = random.randint(-style["x_jitter"], style["x_jitter"])
            jitter_y = random.randint(-style["y_jitter"], style["y_jitter"])

            if random.random() < 0.10:
                jitter_x += random.randint(-3, 3)
                jitter_y += random.randint(-3, 3)

            curve_offset_x = int(curve_strength * ((row_idx + 1) ** 1.2))

            cx = int(
                col_center_x
                + col_dx
                + curve_offset_x
                + jitter_x
                + style["x_offset"]
            )
            y = int(
                y1 + 6
                + row_idx * (font_size + style["char_spacing"])
                + col_dy
                + jitter_y
            )

            x = int(cx - glyph.width / 2)
            base.alpha_composite(glyph, (x, y))


def draw_debug_bbox(base: Image.Image, fields_cfg: dict):
    draw = ImageDraw.Draw(base)
    for field_name, cfg in fields_cfg.items():
        bbox = cfg["bbox"]
        draw.rectangle(bbox, outline=(255, 0, 0, 180), width=2)
        draw.text((bbox[0], bbox[1]), field_name, fill=(255, 0, 0, 220))


def render_cover(
    cover_fields: Dict[str, str],
    output_path: str | Path,
    template_path: str | Path = DEFAULT_COVER_TEMPLATE_PATH,
    font_path: str | Path = DEFAULT_FONT_PATH,
    debug: bool = False,
    seed: Optional[int] = 2026,
) -> Path:
    """
    渲染封面。

    Args:
        cover_fields: 包含 right_text / center_text / left_text 的字典。
        output_path: 输出图片路径。
        template_path: 封面模板路径，默认 data/cover_template.png。
        font_path: 字体路径，默认 fonts/MasaFont-Regular.ttf。
        debug: 是否绘制 bbox 调试框。
        seed: 随机种子。设为 None 可每次生成不同笔迹。

    Returns:
        Path: 输出文件路径。
    """
    if seed is not None:
        random.seed(seed)

    template_path = Path(template_path)
    font_path = Path(font_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not template_path.exists():
        raise FileNotFoundError(f"找不到封面模板: {template_path}")
    if not font_path.exists():
        raise FileNotFoundError(f"找不到字体文件: {font_path}")

    base = ensure_rgba(Image.open(template_path))
    w, h = base.size

    # 按模板尺寸缩放 bbox / 字号。
    fields_cfg = {}
    for field_name, cfg in COVER_CFG["fields"].items():
        fields_cfg[field_name] = {
            "bbox": scale_bbox(cfg["bbox"], w, h),
            "font_size": scale_font_size(cfg["font_size"], h),
        }

    if debug:
        draw_debug_bbox(base, fields_cfg)

    for field_name, field_cfg in fields_cfg.items():
        text = cover_fields.get(field_name, "")
        draw_vertical_text(
            base=base,
            text=text,
            bbox=field_cfg["bbox"],
            base_font_size=field_cfg["font_size"],
            field_name=field_name,
            font_path=font_path,
        )

    base.save(output_path)
    return output_path


if __name__ == "__main__":
    demo_fields = {
        "right_text": "汕头埠慈亲大人收",
        "center_text": "家慈亲大人安启",
        "left_text": "外付银二十元男阿明寄",
    }

    out = PROJECT_ROOT / "outputs" / "demo_cover.png"
    render_cover(demo_fields, out, debug=False)
    print(f"[OK] 封面已生成: {out}")
