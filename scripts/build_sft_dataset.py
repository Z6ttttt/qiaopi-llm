# -*- coding: utf-8 -*-
r"""
build_sft_dataset.py

将侨批 Excel 数据集转换为 Qwen / ChatML 风格 SFT JSONL。

默认输入：D:\qiao\qiaopi\data\qiaopi_dataset_100_clean_v1.xlsx
默认输出：D:\qiao\qiaopi\data\processed

数据集约定包含 4 个 sheet：
  - Main
  - tags
  - relationship_rules
  - greeting_phrase

第一版训练任务：
  1. qiaopi_tagging: body_fields -> relationship / tags / extra_tags / modern_explanation
  2. user_to_qiaopi_body: user_query -> metadata / body_fields
  3. ask_clarification: 信息严重不足时追问，少量辅助样本

运行示例：
  python scripts/build_sft_dataset.py
  python scripts/build_sft_dataset.py --input "D:\qiao\qiaopi\data\qiaopi_dataset_100_clean_v1.xlsx"
  python scripts/build_sft_dataset.py --input "D:\qiao\qiaopi\data\qiaopi_dataset_100_clean_v1.xlsx" --output_dir "D:\qiao\qiaopi\data\processed"

依赖：
  pip install pandas openpyxl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_INPUT = r"D:\qiao\qiaopi\data\qiaopi_dataset_100_clean_v1.xlsx"

SYSTEM_PROMPT = (
    "你是侨批生成系统。你必须根据任务要求输出严格 JSON，不要输出解释。"
    "字段名必须保持稳定；tags 只能从固定标签表中选择；extra_tags 用于具体细节。"
)

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

TAG_ALIASES = {
    "询问家中": "问候家中",
    "问家中": "问候家中",
    "问候家里": "问候家中",
    "问候父母亲": "问候父母",
    "问候母亲": "问候父母",
    "问候父亲": "问候父母",
    "汇款": "寄款",
    "寄银": "寄款",
    "寄钱": "寄款",
    "家庭用度": "家用",
    "家用钱": "家用",
    "想家": "思亲",
    "思念": "思亲",
    "谋生": "工作谋生",
    "近况": "说明近况",
    "病情问候": "生病问候",
    "疾病问候": "生病问候",
    "修屋": "建房修屋",
    "建房": "建房修屋",
    "还债": "债务",
    "欠债": "债务",
}

# 为了兼容 Excel 中可能出现的不同列名，统一用候选列名读取。
COLUMN_CANDIDATES = {
    "case_id": ["case_id", "案例编号", "编号"],
    "entire": ["entire", "完整案例", "完整侨批", "侨批全文"],
    "era": ["era", "年代", "时期"],
    "sender_name": ["sender_name", "寄批人姓名", "寄件人姓名", "寄信人姓名", "sender"],
    "sender_place_modern": ["sender_place_modern", "寄出地现代", "寄出地", "寄批地现代", "寄信地现代"],
    "sender_place_old": ["sender_place_old", "寄出地旧称", "寄批地旧称", "寄信地旧称"],
    "sender_role_modern": ["sender_role_modern", "寄批人现代身份", "寄件人现代身份", "寄信人现代身份"],
    "sender_role_old": ["sender_role_old", "寄批人旧称", "寄件人旧称", "寄信人旧称", "自称"],
    "receiver_name": ["receiver_name", "收批人姓名", "收件人姓名", "收信人姓名", "receiver"],
    "receiver_place_modern": ["receiver_place_modern", "收批地现代", "收件地现代", "收信地现代", "收批地"],
    "receiver_place_old": ["receiver_place_old", "收批地旧称", "收件地旧称", "收信地旧称"],
    "receiver_role_modern": ["receiver_role_modern", "收批人现代身份", "收件人现代身份", "收信人现代身份"],
    "receiver_role_old": ["receiver_role_old", "收批人旧称", "收件人旧称", "收信人旧称", "称谓"],
    "relationship": ["relationship", "人物关系", "关系"],
    "amount_modern": ["amount_modern", "现代金额", "金额现代", "金额"],
    "amount_old": ["amount_old", "旧式金额", "金额旧称", "金额旧式"],
    "tags": ["tags", "标签", "主题标签"],
    "extra_tags": ["extra_tags", "细节标签", "额外标签"],
    "modern_explanation": ["modern_explanation", "modern_explanation_objective", "现代解释", "白话解释", "现代白话解释"],
    "salutation": ["salutation", "称呼", "开头称呼"],
    "body_text": ["body_text", "正文", "正文主体", "侨批正文"],
    "closing": ["closing", "结尾", "结语"],
    "date": ["date", "日期", "落款日期"],
    "signature": ["signature", "署名", "落款", "签名"],
    "right_text": ["right_text", "封面右栏", "右栏"],
    "center_text": ["center_text", "封面中栏", "中栏"],
    "left_text": ["left_text", "封面左栏", "左栏"],
    "user_query_a": ["user_query_a", "完整用户输入", "用户输入完整", "query_a", "user_query_complete"],
    "user_query_b": ["user_query_b", "不完整用户输入", "用户输入不完整", "query_b", "user_query_incomplete"],
    "rewrite_modern_text": ["rewrite_modern_text", "现代话改写输入", "modern_text", "rewrite现代话"],
    "rewrite_qiaopi_text": ["rewrite_qiaopi_style_text", "qiaopi_style_text", "侨批风格改写", "rewrite侨批句"],
    "rewrite_tags": ["rewrite_tags", "改写标签"],
}


def norm_text(value: Any) -> str:
    """清理单元格文本：空值转空串，去掉多余空白。"""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_header(name: Any) -> str:
    return norm_text(name).replace("\n", "").strip()


def get_value(row: pd.Series, key: str, default: str = "") -> str:
    """按候选列名读取字段。"""
    candidates = COLUMN_CANDIDATES.get(key, [key])
    for col in candidates:
        if col in row.index:
            v = norm_text(row[col])
            if v != "":
                return v
    return default


def parse_list_field(value: Any, fixed_only: bool = False) -> List[str]:
    """解析 tags / extra_tags 等列表字段。"""
    text = norm_text(value)
    if not text or text in {"[]", "None", "none", "无", "空", "nan"}:
        return []

    # 支持 JSON list 字符串。
    if text.startswith("[") and text.endswith("]"):
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                raw_items = [norm_text(x) for x in obj]
            else:
                raw_items = [text]
        except Exception:
            raw_items = re.split(r"[,，/、;；|\n]+", text.strip("[]"))
    else:
        raw_items = re.split(r"[,，/、;；|\n]+", text)

    items: List[str] = []
    for item in raw_items:
        item = norm_text(item).strip("'\" ")
        if not item or item in {"None", "none", "无", "空"}:
            continue
        item = TAG_ALIASES.get(item, item)
        if fixed_only and item not in FIXED_TAGS:
            continue
        if item not in items:
            items.append(item)
    return items

# 每个 tag 对应的证据关键词。用于把 tags 拆成更细的 tag_details，
# 让模型学习“正文片段 -> 标签”的对应关系。注意：这是启发式辅助，不改变原始 tags。
TAG_EVIDENCE_KEYWORDS = {
    "报平安": ["身体粗安", "平安", "安好", "尚安", "粗安", "无恙"],
    "寄款": ["寄上", "奉上", "附上", "汇上", "银", "元", "批银", "寄银"],
    "思亲": ["挂念", "思念", "每念", "念及", "离乡", "想念"],
    "家用": ["家用", "日用", "柴米", "家计", "用度", "补家"],
    "问候父母": ["父亲", "母亲", "慈亲", "严亲", "双亲", "二老", "保重", "珍摄"],
    "问候家中": ["家中", "大小", "诸亲", "近况", "可安", "家里"],
    "劝学": ["读书", "学业", "劝学", "勿贪玩", "用功", "求学"],
    "劝勤俭": ["勤俭", "节俭", "俭省", "不可浪费", "省用"],
    "说明近况": ["近来", "近况", "近以", "工事", "停工", "失业", "薪", "生意", "工作"],
    "工作谋生": ["做工", "工事", "谋生", "营生", "帮工", "杂货店", "园内", "树胶", "椰干"],
    "生病问候": ["病", "腰痛", "药", "医", "身体", "痊愈", "病体"],
    "婚嫁": ["婚", "嫁", "娶", "亲事", "成亲"],
    "丧事": ["丧", "亡", "殁", "讣", "祭", "灵"],
    "添丁": ["添丁", "生子", "产", "婴", "孙"],
    "建房修屋": ["修屋", "建房", "屋", "瓦", "墙", "修葺"],
    "债务": ["债", "还", "欠", "陈伯", "偿还"],
    "收成田园": ["田", "园", "收成", "稻", "农", "田园"],
    "节庆问候": ["春节", "新年", "端午", "中秋", "节", "岁暮"],
    "托人带信": ["托人", "托便", "带信", "转交", "来批", "回批"],
    "承诺再寄": ["再寄", "多寄", "续寄", "下月", "日后", "有就", "稍裕"],
}


def split_sentences(text: str) -> List[str]:
    """把正文粗略切成句子，用于提取 tag 证据。"""
    text = norm_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；;])", text)
    return [p.strip() for p in parts if p.strip()]


def infer_tag_evidence(tag: str, body_text: str) -> str:
    """为单个 tag 从正文里找一句最相关的证据。找不到就返回空串。"""
    sentences = split_sentences(body_text)
    keywords = TAG_EVIDENCE_KEYWORDS.get(tag, [])
    for sent in sentences:
        if any(k in sent for k in keywords):
            return sent
    return ""


def build_tag_details(tags: List[str], body_text: str) -> List[Dict[str, str]]:
    """把 tags 拆成 [{tag, evidence}]，用于 qiaopi_tagging 的更细监督。"""
    details: List[Dict[str, str]] = []
    for tag in tags:
        item = {"tag": tag}
        evidence = infer_tag_evidence(tag, body_text)
        if evidence:
            item["evidence"] = evidence
        details.append(item)
    return details


def maybe_int_from_amount(text: str) -> Optional[int]:
    """非常轻量的金额数值解析，解析不到就返回 None。"""
    text = norm_text(text)
    if not text:
        return None
    m = re.search(r"(\d+)", text)
    if m:
        return int(m.group(1))

    cn_map = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "两": 2,
    }
    # 只处理常见 1-99，如 二十、十五、二十五。
    s = text.replace("元", "").replace("银", "").replace("洋", "")
    if "十" in s:
        parts = s.split("十", 1)
        tens = cn_map.get(parts[0], 1) if parts[0] else 1
        ones = cn_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    if s in cn_map:
        return cn_map[s]
    return None


def compact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """去掉空字符串、空列表、空 dict。"""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cv = compact_dict(v)
            if cv:
                out[k] = cv
        elif isinstance(v, list):
            if v:
                out[k] = v
        elif v is not None and v != "":
            out[k] = v
    return out


def build_metadata(row: pd.Series) -> Dict[str, Any]:
    amount_modern = get_value(row, "amount_modern")
    amount_value = maybe_int_from_amount(amount_modern)
    tags = parse_list_field(get_value(row, "tags"), fixed_only=True)
    extra_tags = parse_list_field(get_value(row, "extra_tags"), fixed_only=False)

    metadata = {
        "era": get_value(row, "era"),
        "sender_name": get_value(row, "sender_name"),
        "sender_place_modern": get_value(row, "sender_place_modern"),
        "sender_place_old": get_value(row, "sender_place_old"),
        "sender_role_modern": get_value(row, "sender_role_modern"),
        "sender_role_old": get_value(row, "sender_role_old"),
        "receiver_name": get_value(row, "receiver_name"),
        "receiver_place_modern": get_value(row, "receiver_place_modern"),
        "receiver_place_old": get_value(row, "receiver_place_old"),
        "receiver_role_modern": get_value(row, "receiver_role_modern"),
        "receiver_role_old": get_value(row, "receiver_role_old"),
        "relationship": get_value(row, "relationship"),
        "amount_modern": amount_modern,
        "amount_old": get_value(row, "amount_old"),
        "amount_value": amount_value,
        "tags": tags,
        "extra_tags": extra_tags,
    }
    return compact_dict(metadata)


def build_body_fields(row: pd.Series) -> Dict[str, Any]:
    body_fields = {
        "salutation": get_value(row, "salutation"),
        "body_text": get_value(row, "body_text"),
        "closing": get_value(row, "closing"),
        "date": get_value(row, "date"),
        "signature": get_value(row, "signature"),
    }
    return compact_dict(body_fields)


def build_cover_fields(row: pd.Series) -> Dict[str, Any]:
    cover = {
        "right_text": get_value(row, "right_text"),
        "center_text": get_value(row, "center_text"),
        "left_text": get_value(row, "left_text"),
    }
    return compact_dict(cover)


def make_messages(task_type: str, user_payload: Dict[str, Any], assistant_payload: Dict[str, Any]) -> Dict[str, Any]:
    user_content = "任务类型：{}\n输入：{}".format(
        task_type,
        json.dumps(user_payload, ensure_ascii=False, indent=2),
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
        ]
    }


def build_tagging_sample(row: pd.Series) -> Optional[Dict[str, Any]]:
    case_id = get_value(row, "case_id")
    body_fields = build_body_fields(row)
    body_text = body_fields.get("body_text", "")
    if not body_text:
        return None

    tags = parse_list_field(get_value(row, "tags"), fixed_only=True)
    extra_tags = parse_list_field(get_value(row, "extra_tags"), fixed_only=False)
    modern_explanation = get_value(row, "modern_explanation")

    # tags 仍然保留为数组；tag_details 用于把每个 tag 拆开并给出正文证据。
    # 这样不会破坏后续评估，又能增强“正文片段 -> 标签”的监督。
    tag_details = build_tag_details(tags, body_text)

    assistant_payload = compact_dict(
        {
            "relationship": get_value(row, "relationship"),
            "tags": tags,
            "tag_details": tag_details,
            "extra_tags": extra_tags,
            "modern_explanation": modern_explanation,
        }
    )
    if not assistant_payload.get("tags") and not assistant_payload.get("modern_explanation"):
        return None

    sample = make_messages(
        "qiaopi_tagging",
        {"case_id": case_id, "body_fields": body_fields},
        assistant_payload,
    )
    sample["case_id"] = case_id
    sample["task_type"] = "qiaopi_tagging"
    return sample


def build_user_to_body_sample(row: pd.Series, query_key: str, completion_mode: str) -> Optional[Dict[str, Any]]:
    case_id = get_value(row, "case_id")
    user_query = get_value(row, query_key)
    if not user_query:
        return None

    assistant_payload = compact_dict(
        {
            "action": "generate",
            "completion_mode": completion_mode,
            "metadata": build_metadata(row),
            "body_fields": build_body_fields(row),
        }
    )
    if not assistant_payload.get("body_fields", {}).get("body_text"):
        return None

    sample = make_messages(
        "user_to_qiaopi_body",
        {"case_id": case_id, "user_query": user_query},
        assistant_payload,
    )
    sample["case_id"] = case_id
    sample["task_type"] = "user_to_qiaopi_body"
    sample["query_kind"] = query_key
    return sample


def build_rewrite_sample(row: pd.Series) -> Optional[Dict[str, Any]]:
    """可选辅助样本：现代话 -> 侨批风格句。没有相关列时自动跳过。"""
    case_id = get_value(row, "case_id")
    modern_text = get_value(row, "rewrite_modern_text")
    qiaopi_text = get_value(row, "rewrite_qiaopi_text")
    if not modern_text or not qiaopi_text:
        return None
    metadata = build_metadata(row)
    user_payload = compact_dict(
        {
            "case_id": case_id,
            "modern_text": modern_text,
            "metadata": metadata,
            "tags": parse_list_field(get_value(row, "rewrite_tags"), fixed_only=True),
        }
    )
    assistant_payload = {"qiaopi_style_text": qiaopi_text}
    sample = make_messages("style_rewrite", user_payload, assistant_payload)
    sample["case_id"] = case_id
    sample["task_type"] = "style_rewrite"
    return sample


def build_clarification_samples() -> List[Dict[str, Any]]:
    """固定追问样本。case_id 使用 clarify_x，不参与真实案例泄漏。"""
    examples = [
        (
            "clarify_001",
            "帮我生成一个侨批。",
            ["sender_place", "receiver_place", "receiver_role", "theme"],
            "请补充寄出地、收批地、收批人身份和主要内容。例如：新加坡寄给汕头母亲，寄二十元，报平安。",
        ),
        (
            "clarify_002",
            "写一封侨批。",
            ["sender_place", "receiver_place", "receiver_role", "theme"],
            "请补充寄出地、收批地、收批人身份和主要内容。例如：槟城儿子写给澄海父亲，寄十五元作家用。",
        ),
        (
            "clarify_003",
            "帮我写一封寄钱的侨批。",
            ["sender_place", "receiver_place", "receiver_role", "amount", "theme"],
            "请补充寄出地、收批地、收批人身份、金额和用途。例如：曼谷写给潮州母亲，寄十元买药并问候身体。",
        ),
        (
            "clarify_004",
            "我要写一封从南洋寄回家的信。",
            ["sender_place", "receiver_place", "receiver_role", "theme"],
            "请补充具体寄出地、收批地、收批人身份和主要内容。例如：新加坡寄给汕头母亲，报平安并寄家用。",
        ),
        (
            "clarify_005",
            "帮我写一封给家里的侨批。",
            ["sender_place", "receiver_place", "receiver_role", "theme"],
            "请补充寄出地、家乡地点、收批人身份和主要内容。例如：雅加达儿子写给潮阳母亲，寄八元家用并说明近况。",
        ),
    ]

    samples = []
    for case_id, query, missing, question in examples:
        sample = make_messages(
            "user_to_qiaopi_body",
            {"case_id": case_id, "user_query": query},
            {"action": "ask_clarification", "missing_fields": missing, "question": question},
        )
        sample["case_id"] = case_id
        sample["task_type"] = "ask_clarification"
        samples.append(sample)
    return samples


def load_main_sheet(input_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(input_path)
    sheet_name = "Main" if "Main" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(input_path, sheet_name=sheet_name, dtype=str)
    df.columns = [clean_header(c) for c in df.columns]
    # 删除全空行。
    df = df.dropna(how="all")
    # 清洗每个单元格。
    for col in df.columns:
        df[col] = df[col].map(norm_text)
    return df


def collect_samples(df: pd.DataFrame, include_rewrite: bool = True) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        case_id = get_value(row, "case_id")
        if not case_id:
            continue
        for builder in [
            build_tagging_sample,
            lambda r: build_user_to_body_sample(r, "user_query_a", "user_specified"),
            lambda r: build_user_to_body_sample(r, "user_query_b", "auto_default"),
        ]:
            sample = builder(row)
            if sample:
                samples.append(sample)
        if include_rewrite:
            sample = build_rewrite_sample(row)
            if sample:
                samples.append(sample)

    samples.extend(build_clarification_samples())
    return samples


def split_case_ids(case_ids: Sequence[str], train_ratio: float, val_ratio: float, seed: int) -> Tuple[set, set, set]:
    ids = sorted(set(case_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # 保证数据量较大时 val/test 至少有样本。
    if n >= 10:
        n_val = max(1, n_val)
        n_train = min(n - 2, max(1, n_train))
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    return train_ids, val_ids, test_ids


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_txt(path: Path, values: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in sorted(values):
            f.write(v + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert qiaopi Excel dataset to SFT JSONL.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="输入 Excel 文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认 input 同级 processed 目录")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_rewrite", action="store_true", help="不生成 style_rewrite 辅助样本")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_main_sheet(input_path)
    samples = collect_samples(df, include_rewrite=not args.no_rewrite)

    real_case_ids = [s["case_id"] for s in samples if not str(s["case_id"]).startswith("clarify_")]
    train_ids, val_ids, test_ids = split_case_ids(real_case_ids, args.train_ratio, args.val_ratio, args.seed)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for s in samples:
        cid = s["case_id"]
        # 追问样本数量少，全部放到 train；需要评估时可手工加入 val/test。
        if str(cid).startswith("clarify_"):
            train_records.append(s)
        elif cid in train_ids:
            train_records.append(s)
        elif cid in val_ids:
            val_records.append(s)
        else:
            test_records.append(s)

    random.Random(args.seed).shuffle(train_records)
    random.Random(args.seed + 1).shuffle(val_records)
    random.Random(args.seed + 2).shuffle(test_records)

    n_train = write_jsonl(output_dir / "train.jsonl", train_records)
    n_val = write_jsonl(output_dir / "val.jsonl", val_records)
    n_test = write_jsonl(output_dir / "test.jsonl", test_records)
    write_jsonl(output_dir / "all.jsonl", train_records + val_records + test_records)

    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    write_txt(split_dir / "train_case_ids.txt", train_ids)
    write_txt(split_dir / "val_case_ids.txt", val_ids)
    write_txt(split_dir / "test_case_ids.txt", test_ids)

    task_counts: Dict[str, int] = {}
    for s in samples:
        task_counts[s.get("task_type", "unknown")] = task_counts.get(s.get("task_type", "unknown"), 0) + 1

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "num_rows_in_main": int(len(df)),
        "num_samples_total": len(samples),
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "num_real_case_ids": len(set(real_case_ids)),
        "task_counts": task_counts,
        "fixed_tags": FIXED_TAGS,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n转换完成：{output_dir}")


if __name__ == "__main__":
    main()
