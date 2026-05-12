#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the formal Qiaopi inference pipeline.

Usage:
  cd /data/luozetong/qiaoxiang/qiaopi

  python scripts/test_pipeline_infer.py \
    --base_model /data/luozetong/models/Qwen2.5-7B-Instruct \
    --adapter outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter
"""

import argparse
import json

from pipeline.inference import QiaopiInferencePipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/data/luozetong/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter")
    args = parser.parse_args()

    pipe = QiaopiInferencePipeline(
        base_model=args.base_model,
        adapter=args.adapter,
    ).load()

    test_queries = [
        "1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。",
        "我想写一封从新加坡寄往汕头的侨批，主要是报平安和寄点家用。",
        "帮我生成一个侨批。",
        "1930年代，阿顺在曼谷写给潮州母亲，寄十元买药，并问母亲腰痛好了吗。",
    ]

    for i, q in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"Test {i}: {q}")
        print("=" * 80)
        result = pipe.generate(q)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
