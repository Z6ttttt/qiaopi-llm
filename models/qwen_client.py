import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenClient:
    def __init__(
        self,
        model_path: str = "/data/luozetong/models/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
    ):
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=1.05,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()