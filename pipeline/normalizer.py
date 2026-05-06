import json
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


def load_json(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到知识库文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class QiaopiNormalizer:
    def __init__(self):
        self.place_alias = load_json(KNOWLEDGE_DIR / "place_alias.json")
        self.role_alias = load_json(KNOWLEDGE_DIR / "role_alias.json")

    def lookup(self, table: Dict[str, List[str]], value: Optional[str]) -> Dict[str, Any]:
        if not value:
            return {
                "modern": None,
                "candidates": [],
                "selected": None,
                "source": "missing"
            }

        value = str(value).strip()

        # 1. 精确匹配
        if value in table:
            return {
                "modern": value,
                "candidates": table[value],
                "selected": table[value][0],
                "source": "knowledge_base"
            }

        # 2. 包含匹配，例如“广东汕头”匹配“汕头”
        for key, candidates in table.items():
            if key in value or value in key:
                return {
                    "modern": value,
                    "matched_key": key,
                    "candidates": candidates,
                    "selected": candidates[0],
                    "source": "knowledge_base_fuzzy"
                }

        # 3. 找不到就原样返回
        return {
            "modern": value,
            "candidates": [value],
            "selected": value,
            "source": "fallback_original"
        }

    def normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        sender_place = metadata.get("sender_place_modern")
        receiver_place = metadata.get("receiver_place_modern")
        sender_role = metadata.get("sender_role_modern")
        receiver_role = metadata.get("receiver_role_modern")

        sender_place_info = self.lookup(self.place_alias, sender_place)
        receiver_place_info = self.lookup(self.place_alias, receiver_place)
        sender_role_info = self.lookup(self.role_alias, sender_role)
        receiver_role_info = self.lookup(self.role_alias, receiver_role)

        normalized = dict(metadata)

        normalized["sender_place_old"] = sender_place_info["selected"]
        normalized["receiver_place_old"] = receiver_place_info["selected"]
        normalized["sender_role_old"] = sender_role_info["selected"]
        normalized["receiver_role_old"] = receiver_role_info["selected"]

        normalized["knowledge_candidates"] = {
            "sender_place": sender_place_info,
            "receiver_place": receiver_place_info,
            "sender_role": sender_role_info,
            "receiver_role": receiver_role_info
        }

        return normalized


if __name__ == "__main__":
    normalizer = QiaopiNormalizer()

    metadata = {
        "era": "1930年代",
        "sender_place_modern": "新加坡",
        "receiver_place_modern": "汕头",
        "sender_role_modern": "儿子",
        "receiver_role_modern": "母亲",
        "amount_modern": "二十元",
        "tags": ["报平安", "寄款", "家用"],
        "extra_tags": ["买米"]
    }

    print(json.dumps(
        normalizer.normalize_metadata(metadata),
        ensure_ascii=False,
        indent=2
    ))