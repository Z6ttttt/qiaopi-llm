#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qiaopi inference pipeline.

功能：
1. 用户输入前置规则判断：明显信息不足时直接 ask_clarification，不调用模型。
2. 加载 Qwen2.5 base + LoRA adapter。
3. 调用 user_to_qiaopi_body 任务生成 JSON。
4. 解析模型输出 JSON。
5. 后处理修复：
   - 用户未提供姓名时，sender_name 统一为“某某”
   - 补齐/修正 tags
   - 生成 cover_fields
   - 补齐 rendering
6. 输出最终可供渲染代码使用的结构化 JSON。

推荐用法：
  from pipeline.inference import QiaopiInferencePipeline

  pipe = QiaopiInferencePipeline(
      base_model="/data/luozetong/models/Qwen2.5-7B-Instruct",
      adapter="outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter",
  )
  result = pipe.generate("1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。")
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


FIXED_TAGS = [
    "报平安",
    "寄款",
    "思亲",
    "家用",
    "问候父母",
    "问候家中",
    "劝学",
    "劝勤俭",
    "说明近况",
    "工作谋生",
    "生病问候",
    "婚嫁",
    "丧事",
    "添丁",
    "建房修屋",
    "债务",
    "收成田园",
    "节庆问候",
    "托人带信",
    "承诺再寄",
]

SYSTEM_PROMPT = (
    "你是侨批生成系统。你必须根据任务要求输出严格 JSON，不要输出解释。"
    "不要输出 Markdown 代码块。字段名必须保持稳定。"
    "tags 只能从固定标签表中选择；extra_tags 用于具体细节。"
)

DEFAULT_CLARIFICATION_RESPONSE = {
    "action": "ask_clarification",
    "missing_fields": ["sender_place", "receiver_place", "receiver_role", "theme"],
    "question": "请补充寄出地、收批地、收批人身份和主要内容。例如：新加坡寄给汕头母亲，寄二十元，报平安。",
}


PLACE_KEYWORDS = [
    "新加坡", "汕头", "潮州", "潮汕", "澄海", "揭阳", "潮阳", "梅州", "兴化",
    "曼谷", "泰国", "暹罗", "暹京", "槟城", "槟榔屿", "马来西亚", "吉隆坡",
    "雅加达", "印尼", "爪哇", "棉兰", "巴城", "西贡", "越南", "仰光", "缅甸",
    "菲律宾", "马尼拉", "太平埠", "怡保", "砂拉越", "沙巴",
]

RECEIVER_KEYWORDS = [
    "母亲", "父亲", "父母", "双亲", "祖母", "祖父", "哥哥", "兄长", "弟弟",
    "妻子", "贤妻", "朋友", "仁兄", "叔父", "伯父", "家里", "家中", "亲人",
]

THEME_KEYWORDS = [
    "报平安", "平安", "寄", "寄钱", "寄款", "汇", "银", "元", "家用", "买米",
    "问候", "保重", "劝学", "读书", "勤俭", "生病", "买药", "医药", "修屋",
    "还债", "债", "丧事", "添丁", "婚嫁", "收成", "田园", "节庆", "托人带信",
    "再寄", "补寄", "工作", "做工", "近况",
]

COMMON_NAME_KEYWORDS = [
    "阿明", "阿发", "阿顺", "阿牛", "大牛", "振业", "炳辉", "瑞昌", "福来",
    "其昌", "大志", "阿强", "阿成", "阿水", "阿德", "某某",
]


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def has_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def should_ask_clarification(user_query: str) -> bool:
    """规则前置判断：明显信息不足时直接追问。"""
    q = norm_text(user_query)

    if not q:
        return True

    vague_exact = {
        "帮我生成一个侨批",
        "帮我生成一封侨批",
        "写一封侨批",
        "帮我写一封侨批",
        "帮我写一封家书",
        "生成侨批",
        "我要一封侨批",
        "我想要一封侨批",
        "写个侨批",
        "帮我写侨批",
    }

    q_no_punc = re.sub(r"[。！!？?\s]", "", q)
    vague_no_punc = {re.sub(r"[。！!？?\s]", "", x) for x in vague_exact}

    if q_no_punc in vague_no_punc:
        return True

    has_place = has_any(q, PLACE_KEYWORDS)
    has_receiver = has_any(q, RECEIVER_KEYWORDS)
    has_theme = has_any(q, THEME_KEYWORDS)

    # 太短且缺核心信息
    if len(q_no_punc) <= 12 and not (has_place and has_receiver and has_theme):
        return True

    # 三类核心信息缺两类以上
    missing_count = int(not has_place) + int(not has_receiver) + int(not has_theme)
    return missing_count >= 2


def find_first_balanced_json_object(text: str) -> Optional[str]:
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

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    candidate = find_first_balanced_json_object(text)
    if candidate:
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def make_user_to_body_prompt(user_query: str) -> str:
    payload = {
        "task_type": "user_to_qiaopi_body",
        "user_query": user_query,
        "fixed_tags": FIXED_TAGS,
        "rules": [
            "如果用户输入严重不足，例如只说“帮我生成一个侨批”“写一封侨批”“帮我写一封家书”，必须输出 action=ask_clarification，不要编造地点、人物、金额或正文。",
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


def user_has_explicit_name(user_query: str, generated_name: str = "") -> bool:
    q = user_query or ""
    if generated_name and generated_name in q:
        return True
    return any(name in q for name in COMMON_NAME_KEYWORDS)


def query_has_explicit_receiver_role(user_query: str) -> bool:
    """判断用户是否明确指定了收件人身份。"""
    q = user_query or ""
    role_words = [
        "母亲", "父亲", "父母", "双亲", "祖母", "祖父", "哥哥", "兄长", "弟弟",
        "妻子", "贤妻", "朋友", "仁兄", "叔父", "伯父", "家里", "家中", "亲人"
    ]
    return any(x in q for x in role_words)


def force_default_receiver_if_missing(result: Dict[str, Any], user_query: str) -> None:
    """
    第一版默认规则：
    如果用户没有明确说收件人身份，则默认“儿子写给母亲”。
    这样避免模型在不完整输入里随机补成父亲/朋友等关系。
    """
    if query_has_explicit_receiver_role(user_query):
        return

    metadata = result.setdefault("metadata", {})
    body_fields = result.setdefault("body_fields", {})

    old_receiver_role = metadata.get("receiver_role_modern", "")
    old_receiver_role_old = metadata.get("receiver_role_old", "")

    metadata["sender_role_modern"] = metadata.get("sender_role_modern") or "儿子"
    metadata["sender_role_old"] = metadata.get("sender_role_old") or "男"
    metadata["receiver_role_modern"] = "母亲"
    metadata["receiver_role_old"] = "慈亲大人"
    metadata["relationship"] = "母子"

    # 同步修复 salutation。
    body_fields["salutation"] = "慈亲大人膝下敬禀者"

    # 尽量替换正文里的父亲称呼，避免 metadata/body 不一致。
    replace_pairs = [
        ("父亲年事渐高", "母亲年事渐高"),
        ("父亲年事已高", "母亲年事已高"),
        ("父亲", "母亲"),
        ("严亲大人", "慈亲大人"),
        ("严亲", "慈亲"),
    ]
    for key in ["body_text"]:
        if key in body_fields and isinstance(body_fields[key], str):
            for a, b in replace_pairs:
                body_fields[key] = body_fields[key].replace(a, b)


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,，/、;；\s]+", value)
        return [x.strip() for x in parts if x.strip()]
    return []


def add_tag(tags: List[str], tag: str):
    if tag in FIXED_TAGS and tag not in tags:
        tags.append(tag)


def infer_tags_from_text(tags: List[str], user_query: str, body_text: str) -> List[str]:
    """
    依据用户输入和正文补齐必要 tags。
    注意：第一版要避免“过度打标签”，所以弱标签主要看 user_query；
    body_text 只用于补寄款/家用/问候这类强信号。
    """
    tags = [t for t in tags if t in FIXED_TAGS]
    user = user_query or ""
    body = body_text or ""
    joined = f"{user} {body}"

    # 强信号：只要有金额/寄款表达，补寄款。
    if any(k in joined for k in ["寄", "寄上", "汇", "银", "元", "奉上"]):
        add_tag(tags, "寄款")

    # 强信号：家用、买米、柴米、日用。
    if any(k in joined for k in ["家用", "买米", "柴米", "日用", "补家"]):
        add_tag(tags, "家用")

    # 强信号：报平安。
    if any(k in joined for k in ["报平安", "平安", "安好", "身体", "粗安", "一切如常"]):
        add_tag(tags, "报平安")

    # 问候父母。
    if any(k in joined for k in ["保重", "母亲年迈", "母亲年事", "父亲年事", "严亲", "慈亲"]):
        add_tag(tags, "问候父母")

    # 思亲：尽量要求明确表达挂念/思念，避免所有家书都标。
    if any(k in joined for k in ["挂念", "思念", "甚念", "勿念", "勿悬念"]):
        add_tag(tags, "思亲")

    # 以下两个标签容易被模型因“橡胶园/近来”误触发，必须用户输入有明确主题才补。
    if any(k in user for k in ["工作", "做工", "工钱", "谋生", "工厂", "胶园", "橡胶园", "失业", "停工"]):
        add_tag(tags, "工作谋生")

    if any(k in user for k in ["近况", "说明近况", "失业", "停工", "工钱减少", "生意", "做工"]):
        add_tag(tags, "说明近况")

    if any(k in joined for k in ["再寄", "补寄", "下月再补", "当即多寄"]):
        add_tag(tags, "承诺再寄")

    # 如果用户没有提工作/近况，但模型生成 body_text 时自己写了橡胶园/做工，不要保留这两个弱标签。
    if not any(k in user for k in ["工作", "做工", "工钱", "谋生", "工厂", "胶园", "橡胶园", "失业", "停工", "近况"]):
        tags = [t for t in tags if t not in {"工作谋生", "说明近况"}]

    # extra_tags 才放细节，核心 tags 去重保序。
    seen = set()
    cleaned = []
    for t in tags:
        if t in FIXED_TAGS and t not in seen:
            cleaned.append(t)
            seen.add(t)
    return cleaned




def generate_cover_fields(result: Dict[str, Any]) -> Dict[str, str]:
    metadata = result.get("metadata", {})
    receiver_place_old = metadata.get("receiver_place_old", "")
    receiver_role_old = metadata.get("receiver_role_old", "")
    sender_role_old = metadata.get("sender_role_old", "")
    sender_name = metadata.get("sender_name", "") or "某某"
    amount_old = metadata.get("amount_old", "")

    right_text = f"{receiver_place_old}{receiver_role_old}收" if (receiver_place_old or receiver_role_old) else ""
    center_text = f"家{receiver_role_old}安启" if receiver_role_old else ""

    if amount_old:
        left_text = f"外付{amount_old}{sender_role_old}{sender_name}寄"
    else:
        sender_place_old = metadata.get("sender_place_old", "")
        left_text = f"{sender_place_old}{sender_role_old}{sender_name}寄"

    return {
        "right_text": right_text,
        "center_text": center_text,
        "left_text": left_text,
    }


def postprocess_generate_result(result: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return DEFAULT_CLARIFICATION_RESPONSE.copy()

    if result.get("action") == "ask_clarification":
        return result

    # 如果模型没有 action，但像 generate 结构，补 action。
    if "action" not in result and ("metadata" in result or "body_fields" in result):
        result["action"] = "generate"

    if result.get("action") != "generate":
        return result

    metadata = result.setdefault("metadata", {})
    body_fields = result.setdefault("body_fields", {})

    # 修复用户未提供姓名时模型乱编姓名。
    generated_name = norm_text(metadata.get("sender_name", ""))
    if not user_has_explicit_name(user_query, generated_name):
        if generated_name:
            # 替换 signature/body_text 中的姓名。
            for key in ["signature", "body_text"]:
                if key in body_fields and isinstance(body_fields[key], str):
                    body_fields[key] = body_fields[key].replace(generated_name, "某某")
        metadata["sender_name"] = "某某"

    # 如果用户未明确收件人身份，默认儿子写给母亲，并同步修复称谓。
    force_default_receiver_if_missing(result, user_query)

    # 统一 tags。
    tags = ensure_list(metadata.get("tags"))
    extra_tags = ensure_list(metadata.get("extra_tags"))

    body_text = norm_text(body_fields.get("body_text", ""))
    tags = infer_tags_from_text(tags, user_query=user_query, body_text=body_text)

    # extra_tags 只放具体细节，不要放固定核心 tags。
    extra_tags = [x for x in extra_tags if x not in FIXED_TAGS]

    metadata["tags"] = tags
    metadata["extra_tags"] = extra_tags

    # 如果没金额但生成了默认场景，可以补默认金额；这里只在空时补，避免覆盖用户/模型已有信息。
    if not metadata.get("amount_modern") and any(k in user_query + body_text for k in ["寄", "家用", "买米"]):
        metadata["amount_modern"] = "二十元"
        metadata["amount_old"] = "银二十元"
        metadata["amount_value"] = 20

    # closing 默认值。
    body_fields.setdefault("closing", "专此奉闻，顺叩福安")

    # 生成 cover_fields。
    result["cover_fields"] = generate_cover_fields(result)

    # 渲染控制字段。
    result.setdefault("rendering", {
        "cover_template": "cover_vertical_v1",
        "body_template": "body_vertical_v1",
        "writing_direction": "vertical",
        "pagination": "auto",
    })

    return result


@dataclass
class QiaopiInferencePipeline:
    base_model: str
    adapter: str
    max_new_tokens: int = 768
    temperature: float = 0.0

    tokenizer: Any = None
    model: Any = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.model = PeftModel.from_pretrained(base, self.adapter)
        self.model.eval()
        return self

    def _generate_raw(self, user_content: str) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

        do_sample = self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.02,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def generate(self, user_query: str) -> Dict[str, Any]:
        if should_ask_clarification(user_query):
            return DEFAULT_CLARIFICATION_RESPONSE.copy()

        prompt = make_user_to_body_prompt(user_query)
        raw = self._generate_raw(prompt)
        parsed = extract_json(raw)

        if parsed is None:
            return {
                "action": "error",
                "error": "model_output_json_parse_failed",
                "raw_output": raw,
            }

        return postprocess_generate_result(parsed, user_query=user_query)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/data/luozetong/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter")
    parser.add_argument("--query", type=str, default="1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。")
    args = parser.parse_args()

    pipe = QiaopiInferencePipeline(
        base_model=args.base_model,
        adapter=args.adapter,
    ).load()

    result = pipe.generate(args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
