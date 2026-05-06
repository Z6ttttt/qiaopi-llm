import json
import re
from typing import Any, Dict, Tuple, Optional


REQUIRED_TOP_KEYS = [
    "action",
    "metadata",
    "cover_fields",
    "body_fields",
    "rendering"
]

REQUIRED_METADATA_KEYS = [
    "era",
    "sender_place_modern",
    "sender_place_old",
    "receiver_place_modern",
    "receiver_place_old",
    "sender_role_modern",
    "sender_role_old",
    "receiver_role_modern",
    "receiver_role_old",
    "relationship",
    "amount_modern",
    "amount_old",
    "tags",
    "extra_tags"
]

REQUIRED_COVER_KEYS = [
    "right_text",
    "center_text",
    "left_text"
]

REQUIRED_BODY_KEYS = [
    "salutation",
    "body_text",
    "closing",
    "date",
    "signature"
]

REQUIRED_RENDERING_KEYS = [
    "cover_template_id",
    "body_template_id",
    "writing_direction",
    "body_pagination",
    "signature_on_last_page"
]


def extract_json_text(raw: str) -> str:
    """
    从模型输出中提取 JSON 部分。
    兼容模型输出前后夹杂解释文字的情况。
    """
    raw = raw.strip()

    # 去掉 markdown code fence
    raw = raw.replace("```json", "").replace("```", "").strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return raw

    return raw[start:end + 1]


def parse_json(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    尝试解析 JSON。
    返回: (obj, error)
    """
    text = extract_json_text(raw)

    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {str(e)}"


def validate_required_keys(obj: Dict[str, Any]) -> list:
    errors = []

    for key in REQUIRED_TOP_KEYS:
        if key not in obj:
            errors.append(f"missing top-level key: {key}")

    if "metadata" in obj and isinstance(obj["metadata"], dict):
        for key in REQUIRED_METADATA_KEYS:
            if key not in obj["metadata"]:
                errors.append(f"missing metadata key: {key}")
    else:
        errors.append("metadata is missing or not dict")

    if "cover_fields" in obj and isinstance(obj["cover_fields"], dict):
        for key in REQUIRED_COVER_KEYS:
            if key not in obj["cover_fields"]:
                errors.append(f"missing cover_fields key: {key}")
    else:
        errors.append("cover_fields is missing or not dict")

    if "body_fields" in obj and isinstance(obj["body_fields"], dict):
        for key in REQUIRED_BODY_KEYS:
            if key not in obj["body_fields"]:
                errors.append(f"missing body_fields key: {key}")
    else:
        errors.append("body_fields is missing or not dict")

    if "rendering" in obj and isinstance(obj["rendering"], dict):
        for key in REQUIRED_RENDERING_KEYS:
            if key not in obj["rendering"]:
                errors.append(f"missing rendering key: {key}")
    else:
        errors.append("rendering is missing or not dict")

    return errors


def infer_relationship(sender_role: str, receiver_role: str) -> Optional[str]:
    """
    简单关系校正。
    后面可以扩展。
    """
    if not sender_role or not receiver_role:
        return None

    if sender_role in ["儿子", "男", "子", "小儿"] and receiver_role in ["母亲", "妈妈", "慈亲大人", "母亲大人", "家慈大人"]:
        return "母子"

    if sender_role in ["儿子", "男", "子", "小儿"] and receiver_role in ["父亲", "爸爸", "父亲大人", "严亲大人", "家父大人"]:
        return "父子"

    if sender_role in ["丈夫", "老公", "夫"] and receiver_role in ["妻子", "老婆", "妻", "内人", "贤妻"]:
        return "夫妻"

    return None


def repair_by_rules(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    用规则做轻量修复。
    不依赖大模型。
    """
    obj = dict(obj)

    metadata = obj.setdefault("metadata", {})
    cover_fields = obj.setdefault("cover_fields", {})
    body_fields = obj.setdefault("body_fields", {})
    rendering = obj.setdefault("rendering", {})

    # 修 relationship
    inferred = infer_relationship(
        metadata.get("sender_role_modern") or metadata.get("sender_role_old"),
        metadata.get("receiver_role_modern") or metadata.get("receiver_role_old")
    )
    if inferred:
        metadata["relationship"] = inferred

    # 补 cover_fields
    receiver_place_old = metadata.get("receiver_place_old") or ""
    receiver_role_old = metadata.get("receiver_role_old") or ""
    sender_role_old = metadata.get("sender_role_old") or ""
    amount_old = metadata.get("amount_old") or metadata.get("amount_modern") or ""

    if not cover_fields.get("right_text") or len(cover_fields.get("right_text", "")) <= 3:
        cover_fields["right_text"] = f"{receiver_place_old}{receiver_role_old}收"

    if not cover_fields.get("center_text") or len(cover_fields.get("center_text", "")) <= 3:
        cover_fields["center_text"] = f"家{receiver_role_old}安启"

    if not cover_fields.get("left_text") or len(cover_fields.get("left_text", "")) <= 3:
        cover_fields["left_text"] = f"外付{amount_old}{sender_role_old}寄"

    # 补正文必要字段
    if not body_fields.get("salutation"):
        body_fields["salutation"] = f"{receiver_role_old}膝下敬禀者"

    if not body_fields.get("closing"):
        body_fields["closing"] = "专此奉闻，顺叩福安"

    if not body_fields.get("date"):
        body_fields["date"] = metadata.get("era", "民国年间")

    if not body_fields.get("signature"):
        body_fields["signature"] = f"{sender_role_old}某某谨禀"

    # 补 rendering
    rendering.setdefault("cover_template_id", "八百长春")
    rendering.setdefault("body_template_id", "body_001")
    rendering.setdefault("writing_direction", "vertical")
    rendering.setdefault("body_pagination", "auto")
    rendering.setdefault("signature_on_last_page", True)

    return obj


def validate_semantics(obj: Dict[str, Any]) -> list:
    """
    检查一些语义问题。
    """
    errors = []
    metadata = obj.get("metadata", {})
    cover_fields = obj.get("cover_fields", {})
    body_fields = obj.get("body_fields", {})

    # relationship 检查
    inferred = infer_relationship(
        metadata.get("sender_role_modern") or metadata.get("sender_role_old"),
        metadata.get("receiver_role_modern") or metadata.get("receiver_role_old")
    )
    if inferred and metadata.get("relationship") != inferred:
        errors.append(
            f"relationship mismatch: got {metadata.get('relationship')}, expected {inferred}"
        )

    # cover_fields 太短
    for k in REQUIRED_COVER_KEYS:
        if len(str(cover_fields.get(k, ""))) <= 3:
            errors.append(f"cover_fields.{k} too short: {cover_fields.get(k)}")

    # body_text 太短
    body_text = str(body_fields.get("body_text", ""))
    if len(body_text) < 30:
        errors.append("body_fields.body_text too short")

    return errors


def validate_and_repair(raw: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    总入口：
    1. 解析 JSON
    2. 检查字段
    3. 规则修复
    4. 再检查
    """
    report = {
        "parse_ok": False,
        "before_errors": [],
        "after_errors": [],
        "repaired": False
    }

    obj, parse_error = parse_json(raw)
    if obj is None:
        report["before_errors"].append(parse_error)
        return None, report

    report["parse_ok"] = True

    before_errors = []
    before_errors.extend(validate_required_keys(obj))
    before_errors.extend(validate_semantics(obj))
    report["before_errors"] = before_errors

    repaired = repair_by_rules(obj)
    report["repaired"] = repaired != obj

    after_errors = []
    after_errors.extend(validate_required_keys(repaired))
    after_errors.extend(validate_semantics(repaired))
    report["after_errors"] = after_errors

    return repaired, report


if __name__ == "__main__":
    raw = """
{
  "action": "generate",
  "metadata": {
    "era": "1930年代",
    "sender_place_modern": "新加坡",
    "sender_place_old": "星洲",
    "receiver_place_modern": "汕头",
    "receiver_place_old": "汕头埠",
    "sender_role_modern": "儿子",
    "sender_role_old": "男",
    "receiver_role_modern": "母亲",
    "receiver_role_old": "慈亲大人",
    "relationship": "父子",
    "amount_modern": "二十元",
    "amount_old": "银二十元",
    "tags": ["报平安", "寄款", "家用"],
    "extra_tags": ["买米"]
  },
  "cover_fields": {
    "right_text": "星洲",
    "center_text": "汕头埠",
    "left_text": "男"
  },
  "body_fields": {
    "salutation": "慈亲大人万福金安",
    "body_text": "男自星洲寓居以来，一切均好。家中所需，男已汇寄银二十元，望母人收悉后",
    "closing": "",
    "date": "",
    "signature": ""
  },
  "rendering": {
    "cover_template_id": "八百长春",
    "body_template_id": "body_001",
    "writing_direction": "vertical",
    "body_pagination": "auto",
    "signature_on_last_page": true
  }
}
"""

    repaired, report = validate_and_repair(raw)

    print("REPORT:")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\nREPAIRED:")
    print(json.dumps(repaired, ensure_ascii=False, indent=2))