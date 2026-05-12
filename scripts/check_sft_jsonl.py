import json
from pathlib import Path

paths = [
    Path("data/processed/train.jsonl"),
    Path("data/processed/val.jsonl"),
    Path("data/processed/test.jsonl"),
]

for path in paths:
    print(f"\nChecking {path}")
    total = 0
    bad = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total += 1
            try:
                obj = json.loads(line)

                messages = obj["messages"]
                assert len(messages) == 3, "messages length != 3"
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                assert messages[2]["role"] == "assistant"

                payload = json.loads(messages[2]["content"])
                task_type = obj.get("task_type")

                if task_type == "qiaopi_tagging":
                    assert "relationship" in payload
                    assert "tags" in payload
                    assert "tag_details" in payload
                    assert "extra_tags" in payload
                    assert "modern_explanation" in payload

                elif task_type == "user_to_qiaopi_body":
                    assert payload.get("action") == "generate"
                    assert "metadata" in payload
                    assert "body_fields" in payload
                    assert "tags" in payload["metadata"]
                    assert "body_text" in payload["body_fields"]

                elif task_type == "ask_clarification":
                    assert payload.get("action") == "ask_clarification"
                    assert "missing_fields" in payload
                    assert "question" in payload

            except Exception as e:
                bad.append((i, obj.get("case_id") if "obj" in locals() else None, str(e)))

    print(f"total: {total}")
    print(f"bad: {len(bad)}")
    if bad:
        print("First 20 bad lines:")
        for item in bad[:20]:
            print(item)