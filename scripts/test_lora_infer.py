#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test inference for Qwen2.5-7B-Instruct + Qiaopi LoRA adapter.

Usage:
  cd /data/luozetong/qiaoxiang/qiaopi

  python scripts/test_lora_infer.py \
    --base_model /data/luozetong/models/Qwen2.5-7B-Instruct \
    --adapter outputs/qwen25-7b-qiaopi-qlora-v1/final_adapter

Optional:
  python scripts/test_lora_infer.py --interactive
"""

import argparse
import json
import re
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "你是侨批生成系统。你必须根据任务要求输出严格 JSON，不要输出解释。"
    "字段名必须保持稳定；tags 只能从固定标签表中选择；extra_tags 用于具体细节。"
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


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def chat_generate(
    tokenizer,
    model,
    user_content: str,
    max_new_tokens: int = 768,
    temperature: float = 0.1,
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
        "repetition_penalty": 1.05,
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
    return {"raw": raw, "json": extract_json(raw)}


def make_user_to_body_prompt(user_query: str) -> str:
    payload = {
        "task_type": "user_to_qiaopi_body",
        "user_query": user_query,
    }
    return "任务类型：user_to_qiaopi_body\n输入：" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def make_tagging_prompt(body_fields: Dict[str, str], case_id: str = "manual_test") -> str:
    payload = {
        "case_id": case_id,
        "body_fields": body_fields,
    }
    return "任务类型：qiaopi_tagging\n输入：" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def print_result(title: str, result: Dict[str, Any]):
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


def run_default_tests(tokenizer, model):
    tests = [
        (
            "Test 1: user_to_qiaopi_body 完整输入",
            make_user_to_body_prompt(
                "1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。"
            ),
        ),
        (
            "Test 2: user_to_qiaopi_body 不完整但可默认补全",
            make_user_to_body_prompt(
                "我想写一封从新加坡寄往汕头的侨批，主要是报平安和寄点家用。"
            ),
        ),
        (
            "Test 3: qiaopi_tagging 正文标签提取",
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
            make_user_to_body_prompt("帮我生成一个侨批。"),
        ),
    ]

    for title, prompt in tests:
        result = chat_generate(
            tokenizer,
            model,
            user_content=prompt,
            max_new_tokens=768,
            temperature=0.1,
        )
        print_result(title, result)


def interactive_loop(tokenizer, model):
    print("\n进入交互模式。输入 exit 退出。")
    print("默认任务：user_to_qiaopi_body")
    while True:
        user_query = input("\n请输入用户需求：").strip()
        if user_query.lower() in {"exit", "quit", "q"}:
            break
        if not user_query:
            continue
        result = chat_generate(
            tokenizer,
            model,
            user_content=make_user_to_body_prompt(user_query),
            max_new_tokens=768,
            temperature=0.1,
        )
        print_result("Interactive user_to_qiaopi_body", result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="/data/luozetong/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="outputs/qwen25-7b-qiaopi-qlora-v1/final_adapter",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive user_to_qiaopi_body inference.",
    )
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
