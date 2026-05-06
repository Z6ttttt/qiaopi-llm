import json
from typing import Any, Dict

from models.qwen_client import QwenClient
from models.prompts import GEN_SYSTEM_PROMPT, GEN_USER_PROMPT_TEMPLATE


class QiaopiGenerator:
    def __init__(self, client: QwenClient):
        self.client = client

    def build_input(self, normalized_metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_type": "full_generation",
            "metadata": {
                "era": normalized_metadata.get("era"),
                "sender_place_modern": normalized_metadata.get("sender_place_modern"),
                "sender_place_old": normalized_metadata.get("sender_place_old"),
                "receiver_place_modern": normalized_metadata.get("receiver_place_modern"),
                "receiver_place_old": normalized_metadata.get("receiver_place_old"),
                "sender_role_modern": normalized_metadata.get("sender_role_modern"),
                "sender_role_old": normalized_metadata.get("sender_role_old"),
                "receiver_role_modern": normalized_metadata.get("receiver_role_modern"),
                "receiver_role_old": normalized_metadata.get("receiver_role_old"),
                "amount_modern": normalized_metadata.get("amount_modern"),
                "amount_old": normalized_metadata.get("amount_old", normalized_metadata.get("amount_modern")),
                "tags": normalized_metadata.get("tags", []),
                "extra_tags": normalized_metadata.get("extra_tags", [])
            },
            "knowledge_candidates": normalized_metadata.get("knowledge_candidates", {}),
            "generation_requirements": {
                "style": "上世纪侨批风格，浅显文言，民国家书口吻",
                "cover_rule": "封面三栏分别生成 right_text, center_text, left_text",
                "body_rule": "正文包含称呼、正文、结尾、日期、署名",
                "render_rule": "正文分页由代码处理，模型只输出完整 body_fields"
            }
        }

    def generate(self, normalized_metadata: Dict[str, Any], max_new_tokens: int = 256) -> str:
        input_obj = self.build_input(normalized_metadata)
        input_json = json.dumps(input_obj, ensure_ascii=False, indent=2)

        user_prompt = GEN_USER_PROMPT_TEMPLATE.format(input_json=input_json)

        return self.client.chat(
            system_prompt=GEN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.2
        )


if __name__ == "__main__":
    from pipeline.normalizer import QiaopiNormalizer

    client = QwenClient()
    normalizer = QiaopiNormalizer()
    generator = QiaopiGenerator(client)

    metadata = {
        "era": "1930年代",
        "sender_place_modern": "新加坡",
        "receiver_place_modern": "汕头",
        "sender_role_modern": "儿子",
        "receiver_role_modern": "母亲",
        "amount_modern": "二十元",
        "amount_old": "银二十元",
        "tags": ["报平安", "寄款", "家用"],
        "extra_tags": ["买米"]
    }

    normalized = normalizer.normalize_metadata(metadata)

    result = generator.generate(
        normalized_metadata=normalized,
        max_new_tokens=512
    )

    print(result)