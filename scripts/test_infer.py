import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data/luozetong/models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

messages = [
    {
        "role": "system",
        "content": "你是侨批生成系统。你必须输出严格 JSON，不要输出解释。"
    },
    {
        "role": "user",
        "content": """
请从用户输入中提取侨批生成需要的信息。

用户输入：
1930年代，新加坡儿子写给汕头母亲，寄二十元，报平安，让家里买米。

请输出 JSON，字段包括 action, metadata, missing_fields。
"""
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.2,
        top_p=0.9,
        do_sample=True
    )

generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
result = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(result)