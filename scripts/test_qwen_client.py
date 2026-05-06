PYTHONPATH=. python scripts/test_qwen_client.py


from models.qwen_client import QwenClient
from models.prompts import EXTRACT_SYSTEM_PROMPT, EXTRACT_USER_PROMPT_TEMPLATE


def main():
    client = QwenClient()

    user_query = "1930年代，新加坡儿子写给汕头母亲，寄二十元，报平安，让家里买米。"

    prompt = EXTRACT_USER_PROMPT_TEMPLATE.format(user_query=user_query)

    result = client.chat(
        system_prompt=EXTRACT_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_new_tokens=128,
        temperature=0.2,
    )

    print(result)


if __name__ == "__main__":
    main()