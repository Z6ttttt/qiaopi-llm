#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pipeline/render_text.py

正文渲染模块。

输入：
    body_fields = {
        "salutation": "慈亲大人膝下敬禀者",
        "body_text": "...",
        "closing": "专此奉闻，顺叩福安",
        "date": "民国三十四年春月",
        "signature": "男阿明"
    }

输出：
    text.png

说明：
    本文件由早期 Windows 测试脚本 text.py 改造而来。
    原脚本已经包含正文模板栏位计算、竖排文字、手写扰动和墨迹效果。
    这里主要做了工程化：
    1. 去掉 Windows 硬编码路径；
    2. 改成可调用的 render_body() 函数；
    3. 默认使用项目路径：
       data/text_template.png
       fonts/MasaFont-Regular.ttf
"""

import random
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from pipeline.text_convert import clean_render_text








PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT_TEMPLATE_PATH = PROJECT_ROOT / "data" / "text_template.png"
DEFAULT_FONT_PATH = PROJECT_ROOT / "fonts" / "MasaFont-Regular.ttf"


BODY_CFG = {
    # 页边距
    "top_margin": 70,
    "bottom_margin": 70,
    "left_margin": 45,
    "right_margin": 45,

    # 模板里红色竖线的条数。
    # 若你的模板不是 15 条红线，可改这里。
    "num_guides": 15,

    # 字体
    "font_size": 42,
    "size_jitter": 2,

    # 竖排字距
    "char_spacing": 10,

    # 署名和日期占用的列数
    "reserve_cols_for_tail": 2,

    # 扰动
    "x_jitter": 2,
    "y_jitter": 3,
    "rotation_jitter": 2.0,

    # 墨色
    "ink_gray_range": (18, 45),
    "opacity_range": (205, 245),

    # 墨迹效果
    "blur_prob": 0.15,
    "dilate_prob": 0.08,
    "erode_prob": 0.06,

    # 整体左右微调
    "x_offset": -2,
}


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


def apply_ink_effect(glyph: Image.Image) -> Image.Image:
    if random.random() < BODY_CFG["blur_prob"]:
        glyph = glyph.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.45)))

    if random.random() < BODY_CFG["dilate_prob"]:
        glyph = glyph.filter(ImageFilter.MaxFilter(size=3))

    if random.random() < BODY_CFG["erode_prob"]:
        glyph = glyph.filter(ImageFilter.MinFilter(size=3))

    return glyph


def random_ink_color():
    ink = random.randint(*BODY_CFG["ink_gray_range"])
    opacity = random.randint(*BODY_CFG["opacity_range"])

    r = min(ink + random.randint(1, 8), 255)
    g = min(ink + random.randint(0, 5), 255)
    b = max(ink - random.randint(0, 3), 0)

    return (r, g, b, opacity)


def draw_text_once(draw: ImageDraw.ImageDraw, pos, ch: str, font, fill):
    draw.text(pos, ch, font=font, fill=fill)


def render_single_char(ch: str, font_size: int, font_path: Path) -> Image.Image:
    font = load_font(font_path, font_size)

    canvas_size = font_size * 4
    glyph = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(glyph)

    bbox = draw.textbbox((0, 0), ch, font=font)
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    x = (canvas_size - char_w) // 2 - bbox[0]
    y = (canvas_size - char_h) // 2 - bbox[1]

    fill = random_ink_color()

    draw_text_once(draw, (x, y), ch, font, fill)

    # 低概率轻微叠一次，让少数字略重一点
    if random.random() < 0.15:
        fill2 = (fill[0], fill[1], fill[2], max(fill[3] - 40, 120))
        draw_text_once(draw, (x + 1, y), ch, font, fill2)

    glyph = crop_transparent(glyph)

    angle = random.uniform(-BODY_CFG["rotation_jitter"], BODY_CFG["rotation_jitter"])
    glyph = glyph.rotate(
        angle,
        expand=True,
        resample=Image.Resampling.BICUBIC,
    )

    glyph = apply_ink_effect(glyph)
    glyph = crop_transparent(glyph)
    return glyph


def compute_column_centers(img_w: int):
    """
    根据模板宽度和 guide 数量，计算每个可写竖栏的中心点。
    - 红线一共 num_guides 条
    - 相邻两条红线之间，是一个可写栏位
    - 可写栏位数 = num_guides - 1
    """
    left = BODY_CFG["left_margin"]
    right = img_w - BODY_CFG["right_margin"]

    num_guides = BODY_CFG["num_guides"]
    guide_gap = (right - left) / (num_guides - 1)

    guide_xs = [left + i * guide_gap for i in range(num_guides)]

    col_centers = []
    for i in range(len(guide_xs) - 1):
        cx = (guide_xs[i] + guide_xs[i + 1]) / 2
        col_centers.append(cx)

    # 竖排正文从右往左写
    return col_centers[::-1]


def split_text_to_columns(text: str, rows_per_col: int, max_cols: int):
    text = clean_text(text)
    if not text:
        return [], ""

    chunks = [
        text[i:i + rows_per_col]
        for i in range(0, len(text), rows_per_col)
    ]

    visible = chunks[:max_cols]
    overflow_chunks = chunks[max_cols:]
    overflow = "".join(overflow_chunks)
    return visible, overflow


def draw_vertical_column(
    base: Image.Image,
    text: str,
    center_x: float,
    top_y: int,
    font_size: int,
    font_path: Path,
):
    text = clean_text(text)
    if not text:
        return

    col_dx = random.randint(-1, 1)
    col_dy = random.randint(-2, 2)
    curve_strength = random.uniform(-0.05, 0.05)

    for row_idx, ch in enumerate(text):
        actual_size = font_size + random.randint(-BODY_CFG["size_jitter"], BODY_CFG["size_jitter"])
        actual_size = max(28, actual_size)

        glyph = render_single_char(ch, actual_size, font_path=font_path)

        jitter_x = random.randint(-BODY_CFG["x_jitter"], BODY_CFG["x_jitter"])
        jitter_y = random.randint(-BODY_CFG["y_jitter"], BODY_CFG["y_jitter"])

        curve_offset_x = int(curve_strength * ((row_idx + 1) ** 1.2))

        cx = int(center_x + BODY_CFG["x_offset"] + col_dx + curve_offset_x + jitter_x)
        y = int(top_y + row_idx * (font_size + BODY_CFG["char_spacing"]) + col_dy + jitter_y)

        x = int(cx - glyph.width / 2)
        base.alpha_composite(glyph, (x, y))


def compose_main_text(body_fields: Dict[str, str]) -> str:
    salutation = clean_text(body_fields.get("salutation", ""))
    body_text = clean_text(body_fields.get("body_text", ""))
    closing = clean_text(body_fields.get("closing", ""))

    # 正文渲染为连续竖排，保持三个部分的顺序。
    return salutation + body_text + closing


def render_body_page(
    main_text: str,
    date_text: str,
    signature_text: str,
    output_path: str | Path,
    template_path: str | Path = DEFAULT_TEXT_TEMPLATE_PATH,
    font_path: str | Path = DEFAULT_FONT_PATH,
    debug: bool = False,
    seed: Optional[int] = 2026,
) -> Path:
    """
    渲染正文页。

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
        raise FileNotFoundError(f"找不到正文模板: {template_path}")
    if not font_path.exists():
        raise FileNotFoundError(f"找不到字体文件: {font_path}")

    base = ensure_rgba(Image.open(template_path))
    w, h = base.size

    col_centers = compute_column_centers(w)

    top_y = BODY_CFG["top_margin"]
    bottom_y = h - BODY_CFG["bottom_margin"]
    usable_h = bottom_y - top_y

    font_size = BODY_CFG["font_size"]
    rows_per_col = max(1, int(usable_h / (font_size + BODY_CFG["char_spacing"])))

    reserve_cols = BODY_CFG["reserve_cols_for_tail"]
    main_cols_available = max(1, len(col_centers) - reserve_cols)

    main_columns, overflow = split_text_to_columns(main_text, rows_per_col, main_cols_available)

    for i, col_text in enumerate(main_columns):
        draw_vertical_column(
            base=base,
            text=col_text,
            center_x=col_centers[i],
            top_y=top_y,
            font_size=font_size,
            font_path=font_path,
        )

    # 左边预留两列写日期和署名
    if reserve_cols >= 2 and len(col_centers) >= 2:
        date_col_x = col_centers[-2]
        sign_col_x = col_centers[-1]

        draw_vertical_column(
            base=base,
            text=date_text,
            center_x=date_col_x,
            top_y=top_y,
            font_size=font_size,
            font_path=font_path,
        )

        draw_vertical_column(
            base=base,
            text=signature_text,
            center_x=sign_col_x,
            top_y=top_y,
            font_size=font_size,
            font_path=font_path,
        )

    if debug:
        draw = ImageDraw.Draw(base)
        for cx in col_centers:
            draw.line([(cx, top_y), (cx, bottom_y)], fill=(0, 0, 255, 120), width=1)

    base.save(output_path)

    if overflow:
        print(f"[WARN] 正文过长，当前单页模板未显示 {len(overflow)} 个字符。后续可扩展分页。")

    return output_path


def render_body(
    body_fields: Dict[str, str],
    output_path: str | Path,
    template_path: str | Path = DEFAULT_TEXT_TEMPLATE_PATH,
    font_path: str | Path = DEFAULT_FONT_PATH,
    debug: bool = False,
    seed: Optional[int] = 2026,
) -> Path:
    """
    根据 body_fields 渲染正文图。
    """
    main_text = compose_main_text(body_fields)
    date_text = clean_text(body_fields.get("date", ""))
    signature_text = clean_text(body_fields.get("signature", ""))

    return render_body_page(
        main_text=main_text,
        date_text=date_text,
        signature_text=signature_text,
        output_path=output_path,
        template_path=template_path,
        font_path=font_path,
        debug=debug,
        seed=seed,
    )


if __name__ == "__main__":
    demo_body_fields = {
        "salutation": "慈亲大人膝下敬禀者",
        "body_text": "男在星洲，近来身体康健。今寄上银二十元，为家中买米之用。母亲年迈，望多加保重。",
        "closing": "专此奉闻，顺叩福安",
        "date": "民国三十四年春月",
        "signature": "男阿明",
    }

    out = PROJECT_ROOT / "outputs" / "demo_text.png"
    render_body(demo_body_fields, out, debug=False)
    print(f"[OK] 正文已生成: {out}")
