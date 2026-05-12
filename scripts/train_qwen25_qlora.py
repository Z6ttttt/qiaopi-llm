import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def add_task_rules_to_messages(obj: dict) -> list:
    """
    在训练阶段给不同任务追加规则说明。
    不改变 assistant 标签，只增强 user prompt。
    """
    messages = [dict(m) for m in obj["messages"]]
    task_type = obj.get("task_type", "")

    if len(messages) < 2 or messages[1].get("role") != "user":
        return messages

    if task_type == "user_to_qiaopi_body":
        rules = """
        
训练规则：
1. 如果用户输入严重不足，例如只说“帮我生成一个侨批”“写一封侨批”“帮我写一封家书”，应输出 action=ask_clarification，不要编造地点、人物、金额和正文。
2. 如果用户给出了寄出地、收批地、收批人身份或主题，应优先使用用户给出的信息，不要替换成其他地点、人物或金额。
3. 如果用户没有提供姓名，sender_name 使用“某某”，不要随机编造姓名。
4. 如果用户没有提供金额但允许默认补全，默认 amount_modern 使用“二十元”，amount_old 使用“银二十元”。
5. 如果正文涉及寄钱、家用、买米，应在 tags 中包含“寄款”和“家用”。
6. tags 只能从固定标签表中选择，extra_tags 只放具体细节。
7. 输出必须是严格 JSON，不要输出解释，不要使用 Markdown 代码块。
"""

        messages[1]["content"] = messages[1]["content"] + rules

    elif task_type == "qiaopi_tagging":
        rules = """
        
训练规则：
1. 必须输出 relationship、tags、tag_details、extra_tags、modern_explanation。
2. tags 必须是数组，只能从固定标签表中选择。
3. tag_details 必须为每个 tag 单独给出 evidence。
4. evidence 应尽量引用 body_text 中能支持该标签的原句或短句。
5. modern_explanation 应用现代白话概括正文含义。
6. 输出必须是严格 JSON，不要输出解释，不要使用 Markdown 代码块。
"""

        messages[1]["content"] = messages[1]["content"] + rules

    elif task_type == "ask_clarification":
        rules = """
        
训练规则：
1. 用户信息不足时必须输出 action=ask_clarification。
2. 不要编造寄出地、收批地、人物关系、金额或正文。
3. missing_fields 应列出缺失字段。
4. question 应引导用户补充寄出地、收批地、收批人身份和主要内容。
5. 输出必须是严格 JSON，不要输出解释。
"""

        messages[1]["content"] = messages[1]["content"] + rules

    return messages

class QiaopiSFTDataset(Dataset):
    """
    读取 data/processed/*.jsonl。
    每行格式：
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "case_id": "...",
      "task_type": "..."
    }

    训练时：
    - system + user 作为 prompt
    - assistant 作为 target
    - labels 中 prompt 部分全部设为 -100，只训练 assistant 输出
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.examples.append(obj)
                if max_samples is not None and len(self.examples) >= max_samples:
                    break

        if not self.examples:
            raise ValueError(f"No examples loaded from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        obj = self.examples[idx]
        messages = add_task_rules_to_messages(obj)

        if len(messages) != 3:
            raise ValueError(f"Expected 3 messages, got {len(messages)} at idx={idx}")

        prompt_messages = messages[:2]
        full_messages = messages[:3]

        # prompt: system + user + assistant generation header
        # prompt: system + user + assistant generation header
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # full: system + user + assistant answer
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        # labels: prompt 部分不算 loss，assistant 部分算 loss
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # 截断
        input_ids = full_ids[: self.max_length]
        labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)

        # 极端情况下如果截断后没有 assistant target，则退化处理
        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "case_id": obj.get("case_id", ""),
            "task_type": obj.get("task_type", ""),
        }


@dataclass
class DataCollatorForQiaopiSFT:
    tokenizer: object

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in features:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]

            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [pad_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data/luozetong/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train.jsonl",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/processed/val.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen25-7b-qiaopi-qlora",
    )

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = QiaopiSFTDataset(
        path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_train_samples,
    )

    eval_dataset = QiaopiSFTDataset(
        path=args.val_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_eval_samples,
    )

    data_collator = DataCollatorForQiaopiSFT(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,

        bf16=True,
        fp16=False,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",

        save_total_limit=3,
        load_best_model_at_end=False,

        report_to="none",
        remove_unused_columns=False,

        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n训练完成，LoRA adapter 已保存到：{final_dir}")


if __name__ == "__main__":
    main()