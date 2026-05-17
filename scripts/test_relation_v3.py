#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/test_relation_v3.py

v3 关系增强模型测试脚本。

目标：
1. 测试 v3 adapter 是否能正确处理多种人物关系；
2. 检查 relationship / sender_role_old / receiver_role_old / salutation / signature / tags；
3. 测试信息不足时是否 ask_clarification；
4. 可选：对部分 generate 样本渲染 cover.png / text.png。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.inference import QiaopiInferencePipeline
from pipeline.render_cover import render_cover
from pipeline.render_text import render_body


TEST_CASES = [
    {
        "id": "mother",
        "query": "1930年代，阿明从新加坡寄二十元给汕头母亲，报平安并让家里买米。",
        "expect_action": "generate",
        "expect_relationship": "母子",
        "expect_sender_role_old": "男",
        "expect_receiver_role_old": "慈亲大人",
        "expect_tags": ["报平安", "寄款", "家用"],
    },
    {
        "id": "father",
        "query": "大牛从槟城寄十五元给澄海父亲，让家里还陈伯的钱。",
        "expect_action": "generate",
        "expect_relationship": "父子",
        "expect_sender_role_old": "男",
        "expect_receiver_role_old": "严亲大人",
        "expect_tags": ["寄款", "债务"],
    },
    {
        "id": "parents",
        "query": "阿顺从星洲寄二十五元给汕头父母，报平安并问父亲腰痛和母亲眼疾好了没。",
        "expect_action": "generate",
        "expect_relationship": "父母子",
        "expect_sender_role_old": "男",
        "expect_receiver_role_old": "双亲大人",
        "expect_tags": ["寄款", "问候父母", "生病问候"],
    },
    {
        "id": "wife",
        "query": "阿旺从新加坡寄十二元给汕头妻子作家用，并让她照顾家中子女。",
        "expect_action": "generate",
        "expect_relationship": "夫妻",
        "expect_sender_role_old": "夫",
        "expect_receiver_role_old": "贤妻",
        "expect_tags": ["寄款", "家用", "问候家中"],
    },
    {
        "id": "brother",
        "query": "阿发从新加坡寄十元给汕头兄长，问母亲咳嗽好了没有。",
        "expect_action": "generate",
        "expect_relationship": "兄弟",
        "expect_sender_role_old": "弟",
        "expect_receiver_role_old": "兄长大人",
        "expect_tags": ["寄款", "生病问候"],
    },
    {
        "id": "uncle",
        "query": "炳辉从曼谷寄二十元给汕头叔父，贺寿并作家用。",
        "expect_action": "generate",
        "expect_relationship": "叔侄",
        "expect_sender_role_old": "侄",
        "expect_receiver_role_old": "叔父大人",
        "expect_tags": ["寄款", "节庆问候"],
    },
    {
        "id": "grandfather",
        "query": "阿林从新加坡寄八元给汕头祖父，表达孝心并问候身体。",
        "expect_action": "generate",
        "expect_relationship": "祖孙",
        "expect_sender_role_old": "孙",
        "expect_receiver_role_old": "祖父大人",
        "expect_tags": ["寄款", "思亲"],
    },
    {
        "id": "friend",
        "query": "阿友从新加坡寄十元给汕头的朋友，问候近况。",
        "expect_action": "generate",
        "expect_relationship": "朋友",
        "expect_sender_role_old": "弟",
        "expect_receiver_role_old": "仁兄",
        "expect_tags": ["寄款"],
    },
    {
        "id": "ask_too_vague",
        "query": "帮我生成一个侨批。",
        "expect_action": "ask_clarification",
    },
    {
        "id": "ask_missing_place",
        "query": "帮我写一封给母亲的侨批，内容是报平安和寄家用。",
        "expect_action": "ask_clarification",
    },
    {
        "id": "safe_partial_default_name",
        "query": "我想写一封从新加坡寄往汕头的侨批，收批人是母亲，寄二十元，主要内容是报平安、寄钱、作家用。",
        "expect_action": "generate",
        "expect_relationship": "母子",
        "expect_sender_name": "某某",
        "expect_receiver_role_old": "慈亲大人",
        "expect_tags": ["报平安", "寄款", "家用"],
    },
]


def check_case(result: Dict[str, Any], case: Dict[str, Any]) -> List[str]:
    errors = []
    action = result.get("action")
    if action != case["expect_action"]:
        errors.append(f"action: expected {case['expect_action']}, got {action}")
        return errors

    if action == "ask_clarification":
        if "missing_fields" not in result:
            errors.append("missing missing_fields")
        if "question" not in result:
            errors.append("missing question")
        return errors

    metadata = result.get("metadata", {})
    body_fields = result.get("body_fields", {})
    cover_fields = result.get("cover_fields", {})

    checks = [
        ("relationship", metadata.get("relationship"), case.get("expect_relationship")),
        ("sender_role_old", metadata.get("sender_role_old"), case.get("expect_sender_role_old")),
        ("receiver_role_old", metadata.get("receiver_role_old"), case.get("expect_receiver_role_old")),
        ("sender_name", metadata.get("sender_name"), case.get("expect_sender_name")),
    ]

    for name, actual, expected in checks:
        if expected is not None and actual != expected:
            errors.append(f"{name}: expected {expected}, got {actual}")

    receiver_role_old = case.get("expect_receiver_role_old")
    if receiver_role_old and receiver_role_old not in body_fields.get("salutation", ""):
        errors.append(f"salutation does not contain {receiver_role_old}: {body_fields.get('salutation')}")

    sender_role_old = case.get("expect_sender_role_old")
    if sender_role_old and not body_fields.get("signature", "").startswith(sender_role_old):
        errors.append(f"signature should start with {sender_role_old}: {body_fields.get('signature')}")

    if receiver_role_old and receiver_role_old not in cover_fields.get("right_text", ""):
        errors.append(f"cover right_text does not contain {receiver_role_old}: {cover_fields.get('right_text')}")
    if sender_role_old and sender_role_old not in cover_fields.get("left_text", ""):
        errors.append(f"cover left_text does not contain {sender_role_old}: {cover_fields.get('left_text')}")

    tags = metadata.get("tags", [])
    for tag in case.get("expect_tags", []):
        if tag not in tags:
            errors.append(f"missing tag: {tag}; actual tags={tags}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/data/luozetong/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter")
    parser.add_argument("--output_dir", type=str, default="outputs/v3_relation_eval")
    parser.add_argument("--render_first_n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = QiaopiInferencePipeline(
        base_model=args.base_model,
        adapter=args.adapter,
    ).load()

    records = []
    passed = 0
    render_count = 0

    for i, case in enumerate(TEST_CASES, 1):
        result = pipe.generate(case["query"])
        errors = check_case(result, case)
        ok = len(errors) == 0
        passed += int(ok)

        record = {
            "id": case["id"],
            "query": case["query"],
            "ok": ok,
            "errors": errors,
            "result": result,
        }
        records.append(record)

        print("\n" + "=" * 100)
        print(f"[{i:02d}] {case['id']} | {'PASS' if ok else 'FAIL'}")
        print("QUERY:", case["query"])
        if errors:
            print("ERRORS:")
            for e in errors:
                print(" -", e)
        print("RESULT:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if result.get("action") == "generate" and render_count < args.render_first_n:
            case_dir = out_dir / f"render_{case['id']}"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            render_cover(result.get("cover_fields", {}), case_dir / "cover.png", seed=args.seed)
            render_body(result.get("body_fields", {}), case_dir / "text.png", seed=args.seed)
            render_count += 1

    summary = {
        "total": len(TEST_CASES),
        "passed": passed,
        "failed": len(TEST_CASES) - passed,
        "adapter": args.adapter,
        "output_dir": str(out_dir),
    }

    (out_dir / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "#" * 100)
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] results saved to: {out_dir / 'results.json'}")
    print(f"[OK] summary saved to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
