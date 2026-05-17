#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patch pipeline/inference.py for v3 relation-enhanced inference.

修复内容：
1. 用户输入完全没有地点时，强制 ask_clarification，避免模型乱补新加坡/汕头。
2. 从 user_query 中用正则提取显式姓名，避免把“大牛/阿顺/炳辉”等错误改成“某某”。
3. 补强 tags 后处理：生病问候、节庆问候。
4. 过滤模型生成的 *_explanation / *_region_old 等非 schema 扩展字段，让输出更干净。

用法：
  cd /data/luozetong/qiaoxiang/qiaopi
  python scripts/patch_inference_v3.py
"""

from pathlib import Path
import re


TARGET = Path("pipeline/inference.py")


NEW_SHOULD_ASK = """def should_ask_clarification(user_query: str) -> bool:
    \"\"\"规则前置判断：明显信息不足时直接追问。

    v3 规则：
    - 完全没有地点时，一律追问。因为封面和正文都依赖寄出地/收批地，
      不能让模型自由补成“新加坡/汕头”。
    - 有地点、收件人、主题中的大部分信息时，允许模型默认补姓名/金额。
    \"\"\"
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
        "请帮我写一封侨批",
        "请生成一封侨批",
    }

    q_no_punc = re.sub(r"[。！!？?，,\\s]", "", q)
    vague_no_punc = {re.sub(r"[。！!？?，,\\s]", "", x) for x in vague_exact}

    if q_no_punc in vague_no_punc:
        return True

    has_place = has_any(q, PLACE_KEYWORDS)
    has_receiver = has_any(q, RECEIVER_KEYWORDS)
    has_theme = has_any(q, THEME_KEYWORDS)

    # v3 新增：完全没有地点时直接追问。
    # 例如：“帮我写一封给母亲的侨批，内容是报平安和寄家用。”
    # 不能让模型自动补新加坡/汕头。
    if not has_place:
        return True

    # 太短且缺核心信息
    if len(q_no_punc) <= 12 and not (has_place and has_receiver and has_theme):
        return True

    # 三类核心信息缺两类以上
    missing_count = int(not has_place) + int(not has_receiver) + int(not has_theme)
    return missing_count >= 2
"""


HELPERS = """

def extract_sender_name_from_query(user_query: str) -> str:
    \"\"\"从用户输入中抽取显式寄批人姓名。

    用于修复 v3 测试中“大牛/阿顺/炳辉”等被后处理改成“某某”的问题。
    只做轻量规则，不追求覆盖所有中文姓名。
    \"\"\"
    q = user_query or ""

    patterns = [
        r"([阿][\\u4e00-\\u9fa5]{1,2})从",
        r"([阿][\\u4e00-\\u9fa5]{1,2})在",
        r"([阿][\\u4e00-\\u9fa5]{1,2})寄",
        r"([阿][\\u4e00-\\u9fa5]{1,2})写",
        r"([阿][\\u4e00-\\u9fa5]{1,2})给",
        r"([\\u4e00-\\u9fa5]{2,3})从",
        r"([\\u4e00-\\u9fa5]{2,3})在",
        r"([\\u4e00-\\u9fa5]{2,3})寄",
    ]

    stop_words = {
        "我想", "帮我", "请帮", "生成", "写一封", "母亲", "父亲", "父母",
        "妻子", "兄长", "叔父", "祖父", "祖母", "朋友", "家里", "家中",
        "新加坡", "槟城", "曼谷", "汕头", "澄海", "潮州",
    }

    for pat in patterns:
        m = re.search(pat, q)
        if m:
            name = m.group(1)
            if name not in stop_words and not any(sw in name for sw in stop_words):
                return name

    return ""


def clean_metadata_schema(metadata: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"过滤模型偶尔生成的额外解释字段，使输出 schema 更稳定。\"\"\"
    allowed = {
        "sender_name",
        "sender_place_modern",
        "sender_place_old",
        "sender_role_modern",
        "sender_role_old",
        "receiver_place_modern",
        "receiver_place_old",
        "receiver_role_modern",
        "receiver_role_old",
        "relationship",
        "amount_modern",
        "amount_old",
        "amount_value",
        "tags",
        "extra_tags",
    }
    return {k: v for k, v in metadata.items() if k in allowed}
"""


NEW_INFER_TAGS = """def infer_tags_from_text(tags: List[str], user_query: str, body_text: str) -> List[str]:
    \"\"\"
    依据用户输入和正文补齐必要 tags。
    v3 重点补强：
    - 生病问候：腰痛、眼疾、咳嗽、牙痛、腿痛、气喘、服药等。
    - 节庆问候：贺寿、寿辰、冬节、春节、中秋等。
    - 工作谋生/说明近况仍然保持谨慎，只在用户明确提到时补。
    \"\"\"
    tags = [t for t in tags if t in FIXED_TAGS]
    user = user_query or ""
    body = body_text or ""
    joined = f"{user} {body}"

    # 强信号：寄款
    if any(k in joined for k in ["寄", "寄上", "汇", "汇上", "银", "元", "奉上", "托带"]):
        add_tag(tags, "寄款")

    # 强信号：家用
    if any(k in joined for k in ["家用", "买米", "柴米", "日用", "帮贴", "帮补", "作家用"]):
        add_tag(tags, "家用")

    # 强信号：报平安
    if any(k in joined for k in ["报平安", "平安", "安好", "身体", "粗安", "尚健", "一切如常", "一切顺遂"]):
        add_tag(tags, "报平安")

    # 问候父母
    if any(k in joined for k in ["母亲", "父亲", "双亲", "严亲", "慈亲", "保重"]):
        add_tag(tags, "问候父母")

    # v3 新增：生病问候
    if any(k in joined for k in [
        "病", "痛", "腰痛", "眼疾", "咳嗽", "咳症", "气喘", "头晕",
        "牙痛", "腿痛", "服药", "买药", "医药", "全愈", "愈否",
        "调治", "染恙", "微恙", "旧疾", "痧症"
    ]):
        add_tag(tags, "生病问候")

    # v3 新增：节庆问候
    if any(k in joined for k in [
        "贺寿", "寿辰", "寿诞", "祝寿", "寿敬", "冬节", "春节",
        "中秋", "新年", "节庆", "寒衣"
    ]):
        add_tag(tags, "节庆问候")

    # 思亲：要求明确挂念/思念表达
    if any(k in joined for k in ["挂念", "思念", "甚念", "悬念", "勿念", "勿悬念", "远念", "孝思"]):
        add_tag(tags, "思亲")

    # 问候家中
    if any(k in joined for k in ["家中大小", "子女", "弟妹", "家中均安", "家中可好"]):
        add_tag(tags, "问候家中")

    # 债务
    if any(k in joined for k in ["还债", "旧债", "欠", "偿还", "代垫"]):
        add_tag(tags, "债务")

    # 劝学
    if any(k in joined for k in ["读书", "学费", "劝学", "用功", "夜学", "识字"]):
        add_tag(tags, "劝学")

    # 承诺再寄
    if any(k in joined for k in ["再寄", "补寄", "下月再补", "当即多寄", "月内当再寄", "得工多寄"]):
        add_tag(tags, "承诺再寄")

    # 工作谋生 / 说明近况：避免因为模型自己写“杂货店帮工”而过度打标签。
    if any(k in user for k in ["工作", "做工", "工钱", "谋生", "工厂", "胶园", "橡胶园", "失业", "停工", "洋行", "码头", "米行"]):
        add_tag(tags, "工作谋生")

    if any(k in user for k in ["近况", "说明近况", "失业", "停工", "工钱减少", "生意", "做工", "米市", "胶价"]):
        add_tag(tags, "说明近况")

    if not any(k in user for k in ["工作", "做工", "工钱", "谋生", "工厂", "胶园", "橡胶园", "失业", "停工", "近况", "洋行", "码头", "米行"]):
        tags = [t for t in tags if t not in {"工作谋生", "说明近况"}]

    # 去重保序
    seen = set()
    cleaned = []
    for t in tags:
        if t in FIXED_TAGS and t not in seen:
            cleaned.append(t)
            seen.add(t)
    return cleaned
"""


NEW_NAME_BLOCK = """    # 修复 sender_name：
    # 1. 如果用户输入显式包含姓名，则优先使用该姓名；
    # 2. 如果用户没有提供姓名，则统一使用“某某”，不要让模型随机编。
    explicit_name = extract_sender_name_from_query(user_query)
    generated_name = norm_text(metadata.get("sender_name", ""))

    if explicit_name:
        old_name = generated_name
        metadata["sender_name"] = explicit_name

        # 同步修复正文和署名。
        for key in ["signature", "body_text"]:
            if key in body_fields and isinstance(body_fields[key], str):
                if old_name and old_name != explicit_name:
                    body_fields[key] = body_fields[key].replace(old_name, explicit_name)

        sender_role_old = metadata.get("sender_role_old", "")
        if sender_role_old:
            body_fields["signature"] = f"{sender_role_old}{explicit_name}"

    else:
        if generated_name:
            for key in ["signature", "body_text"]:
                if key in body_fields and isinstance(body_fields[key], str):
                    body_fields[key] = body_fields[key].replace(generated_name, "某某")
        metadata["sender_name"] = "某某"
"""


def main():
    if not TARGET.exists():
        raise FileNotFoundError(f"找不到目标文件: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(".py.bak_v3_patch")
    backup.write_text(text, encoding="utf-8")
    print(f"[OK] backup saved: {backup}")

    # 1. replace should_ask_clarification
    pattern = r"def should_ask_clarification\(user_query: str\) -> bool:\n.*?\n\ndef find_first_balanced_json_object"
    text2, n = re.subn(pattern, lambda m: NEW_SHOULD_ASK + "\n\ndef find_first_balanced_json_object", text, flags=re.S)
    print(f"[PATCH] should_ask_clarification replacements: {n}")
    text = text2

    # 2. insert helpers after user_has_explicit_name if missing
    if "def extract_sender_name_from_query" not in text:
        marker_pattern = r"def user_has_explicit_name\(user_query: str, generated_name: str = \"\"\) -> bool:\n.*?\n\s*return any\(name in q for name in COMMON_NAME_KEYWORDS\)\n"
        text2, n = re.subn(marker_pattern, lambda m: m.group(0) + HELPERS, text, flags=re.S)
        if n == 0:
            text = text.replace("\ndef ensure_list", HELPERS + "\n\ndef ensure_list")
            print("[PATCH] helpers inserted before ensure_list")
        else:
            text = text2
            print("[PATCH] helpers inserted after user_has_explicit_name")
    else:
        print("[SKIP] helpers already exist")

    # 3. replace infer_tags_from_text
    pattern = r"def infer_tags_from_text\(tags: List\[str\], user_query: str, body_text: str\) -> List\[str\]:\n.*?\n\ndef generate_cover_fields"
    text2, n = re.subn(pattern, lambda m: NEW_INFER_TAGS + "\n\ndef generate_cover_fields", text, flags=re.S)
    print(f"[PATCH] infer_tags_from_text replacements: {n}")
    text = text2

    # 4. replace name block
    pattern = r"    # 修复用户未提供姓名时模型乱编姓名。\n.*?\n(?=    # 如果用户未明确收件人身份)"
    text2, n = re.subn(pattern, lambda m: NEW_NAME_BLOCK + "\n", text, flags=re.S)
    print(f"[PATCH] name postprocess replacements: {n}")
    text = text2

    # 5. ensure clean metadata schema after tags assignment
    if "metadata = clean_metadata_schema(metadata)" not in text:
        marker = '    # 生成 cover_fields。\n    result["cover_fields"] = generate_cover_fields(result)\n'
        repl = '    # 清理模型偶尔生成的非 schema 字段。\n    metadata = clean_metadata_schema(metadata)\n    result["metadata"] = metadata\n\n' + marker
        text = text.replace(marker, repl)
        print("[PATCH] metadata schema cleaner inserted")
    else:
        print("[SKIP] metadata schema cleaner already inserted")

    TARGET.write_text(text, encoding="utf-8")
    print(f"[OK] patched: {TARGET}")


if __name__ == "__main__":
    main()
