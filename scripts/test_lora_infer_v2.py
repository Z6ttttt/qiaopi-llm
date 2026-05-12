#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust inference test for Qwen2.5-7B-Instruct + Qiaopi LoRA adapter.

This version is stronger than the first test script:
1. Uses task-specific rules in the prompt, matching the v2 training idea.
2. Uses temperature=0.0 by default to reduce random hallucination.
3. Extracts the first balanced JSON object, so trailing text or an extra brace is less likely to break parsing.
4. Prints simple schema warnings for qiaopi_tagging / user_to_qiaopi_body / ask_clarification.

Usage:
  cd /data/luozetong/qiaoxiang/qiaopi

  python scripts/test_lora_infer_v2.py \
    --base_model /data/luozetong/models/Qwen2.5-7B-Instruct \
    --adapter outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter

Interactive:
  python scripts/test_lora_infer_v2.py --interactive
"""

import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


FIXED_TAGS = [
    "报平安", "寄款", "思亲", "家用", "问候父母", "问候家中",
    "劝学", "劝勤俭", "说明近况", "工作谋生", "生病问候",
    "婚嫁", "丧事", "添丁", "建房修屋", "债务", "收成田园",
    "节庆问候", "托人带信", "承诺再寄"
]

SYSTEM_PROMPT = (
    "你是侨批生成系统。你必须根据任务要求输出严格 JSON，不要输出解释。"
    "不要输出 Markdown 代码块。字段名必须保持稳定。"
    "tags 只能从固定标签表中选择；extra_tags 用于具体细节。"
)


def load_model(base_model: str, adapter: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def find_first_balanced_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object substring."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Direct parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Parse first balanced object. This fixes cases like:
    # {"a": 1}}   or   {"a": 1}, ["extra"]
    candidate = find_first_balanced_json_object(text)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def validate_output(task_type: str, payload: Optional[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []

    if payload is None:
        return ["JSON 解析失败"]

    if task_type == "qiaopi_tagging":
        required = ["relationship", "tags", "tag_details", "extra_tags", "modern_explanation"]
        for k in required:
            if k not in payload:
                warnings.append(f"缺少字段: {k}")

        if "tags" in payload and isinstance(payload["tags"], list):
            invalid = [t for t in payload["tags"] if t not in FIXED_TAGS]
            if invalid:
                warnings.append(f"存在非法 tags: {invalid}")

        if "tag_details" in payload and isinstance(payload["tag_details"], list):
            for i, item in enumerate(payload["tag_details"]):
                if not isinstance(item, dict):
                    warnings.append(f"tag_details[{i}] 不是对象")
                    continue
                if "tag" not in item:
                    warnings.append(f"tag_details[{i}] 缺少 tag")
                if "evidence" not in item:
                    warnings.append(f"tag_details[{i}] 缺少 evidence")

    elif task_type == "user_to_qiaopi_body":
        action = payload.get("action")
        if action == "ask_clarification":
            for k in ["missing_fields", "question"]:
                if k not in payload:
                    warnings.append(f"ask_clarification 缺少字段: {k}")
        elif action == "generate":
            for k in ["metadata", "body_fields"]:
                if k not in payload:
                    warnings.append(f"generate 缺少字段: {k}")
            if isinstance(payload.get("metadata"), dict):
                for k in ["sender_place_modern", "receiver_place_modern", "receiver_role_modern", "relationship", "tags", "extra_tags"]:
                    if k not in payload["metadata"]:
                        warnings.append(f"metadata 缺少字段: {k}")
                if isinstance(payload["metadata"].get("tags"), list):
                    invalid = [t for t in payload["metadata"]["tags"] if t not in FIXED_TAGS]
                    if invalid:
                        warnings.append(f"存在非法 tags: {invalid}")
            if isinstance(payload.get("body_fields"), dict):
                for k in ["salutation", "body_text", "closing", "date", "signature"]:
                    if k not in payload["body_fields"]:
                        warnings.append(f"body_fields 缺少字段: {k}")
        else:
            warnings.append("缺少 action，或 action 不是 generate/ask_clarification")

    return warnings


def chat_generate(
    tokenizer,
    model,
    user_content: str,
    task_type: str,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.02,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    parsed = extract_json(raw)
    warnings = validate_output(task_type, parsed)

    return {
        "raw": raw,
        "json": parsed,
        "warnings": warnings,
    }


def make_user_to_body_prompt(user_query: str) -> str:
    payload = {
        "task_type": "user_to_qiaopi_body",
        "user_query": user_query,
        "fixed_tags": FIXED_TAGS,
        "rules": [
            "如果用户输入严重不足，例如只说“帮我生成一个侨批”“写一封侨批”“帮我写一封家书”，必须输出 action=ask_clarification，不要编造地点、人物、金额或正文。",
            "如果用户没有提供寄出地、收批地、收批人身份、主要主题中的大部分信息，应输出 action=ask_clarification。",
            "如果用户给出具体信息，必须优先使用用户信息，不要改成其他地点、人物或金额。",
            "如果用户没有提供姓名，sender_name 使用“某某”，不要随机编造姓名。",
            "如果用户没有提供金额但可以默认补全，默认 amount_modern 使用“二十元”，amount_old 使用“银二十元”。",
            "如果出现寄钱、家用、买米，应在 tags 中包含“寄款”和“家用”。",
            "输出必须严格匹配 generate 或 ask_clarification 的 JSON 结构。"
        ],
        "output_schema_generate": {
            "action": "generate",
            "completion_mode": "user_specified 或 auto_default",
            "metadata": {
                "sender_name": "",
                "sender_place_modern": "",
                "sender_place_old": "",
                "sender_role_modern": "",
                "sender_role_old": "",
                "receiver_place_modern": "",
                "receiver_place_old": "",
                "receiver_role_modern": "",
                "receiver_role_old": "",
                "relationship": "",
                "amount_modern": "",
                "amount_old": "",
                "amount_value": 0,
                "tags": [],
                "extra_tags": []
            },
            "body_fields": {
                "salutation": "",
                "body_text": "",
                "closing": "",
                "date": "",
                "signature": ""
            }
        },
        "output_schema_ask_clarification": {
            "action": "ask_clarification",
            "missing_fields": [],
            "question": ""
        }
    }
    return "任务类型：user_to_qiaopi_body\n输入：" + json.dumps(payload, ensure_ascii=False, indent=2)


def make_tagging_prompt(body_fields: Dict[str, str], case_id: str = "manual_test") -> str:
    payload = {
        "case_id": case_id,
        "body_fields": body_fields,
        "fixed_tags": FIXED_TAGS,
        "rules": [
            "必须输出 relationship、tags、tag_details、extra_tags、modern_explanation。",
            "tags 必须是数组，只能从 fixed_tags 中选择。",
            "tag_details 必须为每个 tag 单独给出 evidence。",
            "evidence 应尽量引用 body_text 中能支持该标签的原句或短句。",
            "modern_explanation 应用现代白话概括正文含义。",
            "输出必须是严格 JSON，不要输出解释。"
        ],
        "output_schema": {
            "relationship": "",
            "tags": [],
            "tag_details": [
                {"tag": "", "evidence": ""}
            ],
            "extra_tags": [],
            "modern_explanation": ""
        }
    }
    return "任务类型：qiaopi_tagging\n输入：" + json.dumps(payload, ensure_ascii=False, indent=2)


def print_result(title: str, task_type: str, result: Dict[str, Any]):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("\n[RAW OUTPUT]")
    print(result["raw"])

    print("\n[PARSED JSON]")
    if result["json"] is None:
        print("JSON 解析失败")
    else:
        print(json.dumps(result["json"], ensure_ascii=False, indent=2))

    print("\n[SCHEMA WARNINGS]")
    if result["warnings"]:
        for w in result["warnings"]:
            print("-", w)
    else:
        print("无")


def run_default_tests(tokenizer, model):
    tests: List[Tuple[str, str, str]] = [
        (
            "Test 1: user_to_qiaopi_body 完整输入",
            "user_to_qiaopi_body",
            make_user_to_body_prompt("1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。"),
        ),
        (
            "Test 2: user_to_qiaopi_body 不完整但可默认补全",
            "user_to_qiaopi_body",
            make_user_to_body_prompt("我想写一封从新加坡寄往汕头的侨批，主要是报平安和寄点家用。"),
        ),
        (
            "Test 3: qiaopi_tagging 正文标签提取",
            "qiaopi_tagging",
            make_tagging_prompt(
                {
                    "salutation": "严亲大人膝下敬禀者",
                    "body_text": (
                        "男在太平埠，近以树胶落价，园内停工，儿已失业旬余。"
                        "今仅能寄上银六元，系从前余蓄，心内惭恨。"
                        "现托人寻觅新工，一旦有就，当即多寄。父亲且免愁烦。"
                    ),
                    "closing": "专此奉闻，顺叩福安",
                    "date": "民国三十一年春月",
                    "signature": "男振业",
                },
                case_id="manual_test_001",
            ),
        ),
        (
            "Test 4: ask_clarification 信息严重不足",
            "user_to_qiaopi_body",
            make_user_to_body_prompt("帮我生成一个侨批。"),
        ),
    ]

    for title, task_type, prompt in tests:
        result = chat_generate(tokenizer, model, prompt, task_type=task_type, max_new_tokens=768, temperature=0.0)
        print_result(title, task_type, result)


def interactive_loop(tokenizer, model):
    print("\n进入交互模式。输入 exit 退出。")
    print("默认任务：user_to_qiaopi_body")
    while True:
        user_query = input("\n请输入用户需求：").strip()
        if user_query.lower() in {"exit", "quit", "q"}:
            break
        if not user_query:
            continue
        prompt = make_user_to_body_prompt(user_query)
        result = chat_generate(tokenizer, model, prompt, task_type="user_to_qiaopi_body", max_new_tokens=768, temperature=0.0)
        print_result("Interactive user_to_qiaopi_body", "user_to_qiaopi_body", result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/data/luozetong/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading base model:", args.base_model)
    print("Loading adapter:", args.adapter)
    tokenizer, model = load_model(args.base_model, args.adapter)
    if args.interactive:
        interactive_loop(tokenizer, model)
    else:
        run_default_tests(tokenizer, model)


if __name__ == "__main__":
    main()
