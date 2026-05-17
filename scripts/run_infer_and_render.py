#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/run_infer_and_render.py

小样本完整推理 + 渲染脚本。

流程：
    用户输入
    → QiaopiInferencePipeline.generate()
    → result.json
    → render_cover()
    → render_body()
    → cover.png / text.png

用法：
    cd /data/luozetong/qiaoxiang/qiaopi

    PYTHONPATH=. python scripts/run_infer_and_render.py \
      --query "1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。"

如果你已经在脚本中加入 ROOT 到 sys.path，也可以直接：
    python scripts/run_infer_and_render.py --query "..."
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


# 确保从 scripts/ 目录运行时也能 import pipeline.*
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.inference import QiaopiInferencePipeline
from pipeline.render_cover import render_cover
from pipeline.render_text import render_body


DEFAULT_BASE_MODEL = "/data/luozetong/models/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER = "outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter"
DEFAULT_COVER_TEMPLATE = "data/cover_template.png"
DEFAULT_TEXT_TEMPLATE = "data/text_template.png"
DEFAULT_FONT = "fonts/MasaFont-Regular.ttf"


def safe_case_dir_name(prefix: str = "demo") -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。",
        help="用户自然语言输入。",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Qwen2.5-7B-Instruct 基座路径。",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help="LoRA adapter 路径。",
    )
    parser.add_argument(
        "--cover_template",
        type=str,
        default=DEFAULT_COVER_TEMPLATE,
        help="封面模板路径。",
    )
    parser.add_argument(
        "--text_template",
        type=str,
        default=DEFAULT_TEXT_TEMPLATE,
        help="正文模板路径。",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=DEFAULT_FONT,
        help="字体路径。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录。默认 outputs/render_demo_时间戳。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="渲染调试模式，显示 bbox / 栏位线。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="随机种子。设为 -1 表示每次随机。",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / safe_case_dir_name("render_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = None if args.seed == -1 else args.seed

    print("[INFO] 加载推理 pipeline...")
    pipe = QiaopiInferencePipeline(
        base_model=args.base_model,
        adapter=args.adapter,
    ).load()

    print("[INFO] 用户输入:")
    print(args.query)

    result = pipe.generate(args.query)

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] JSON 已保存: {result_path}")

    if result.get("action") != "generate":
        print("[INFO] 当前结果不是 generate，不进行图片渲染。")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    cover_fields = result.get("cover_fields", {})
    body_fields = result.get("body_fields", {})

    cover_path = output_dir / "cover.png"
    text_path = output_dir / "text.png"

    render_cover(
        cover_fields=cover_fields,
        output_path=cover_path,
        template_path=ROOT / args.cover_template,
        font_path=ROOT / args.font,
        debug=args.debug,
        seed=seed,
    )

    render_body(
        body_fields=body_fields,
        output_path=text_path,
        template_path=ROOT / args.text_template,
        font_path=ROOT / args.font,
        debug=args.debug,
        seed=seed,
    )

    print(f"[OK] 封面图已保存: {cover_path}")
    print(f"[OK] 正文图已保存: {text_path}")

    print("\n[FINAL RESULT]")
    print(json.dumps({
        "result_json": str(result_path),
        "cover_image": str(cover_path),
        "text_image": str(text_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
