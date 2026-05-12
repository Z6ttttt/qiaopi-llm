# qiaopi-llm

侨批生成与渲染系统第一版实验项目。

本项目目标是构建一个小型“侨批生成系统”：用户输入现代自然语言需求，模型生成结构化侨批 JSON，再由规则系统和模板渲染代码生成封面图与正文图。

---

## 1. 项目简介

本项目不是直接训练图像生成模型，而是训练一个**侨批结构化文本生成模型**。

模型负责：

```text
理解用户输入
提取 metadata
判断 relationship
生成 tags / extra_tags
生成侨批风格 body_text
```

规则系统负责：

```text
信息不足时追问
默认字段补全
旧称和称谓修复
cover_fields 生成
渲染文本转繁体
封面和正文模板映射
```

当前第一版已经跑通完整流程：

```text
用户输入
→ Qwen2.5-7B-Instruct + LoRA adapter
→ 结构化 JSON
→ 规则后处理
→ 封面字段生成
→ 正文和封面模板渲染
→ result.json + cover.png + text.png
```

---

## 2. 当前第一版能力

支持：

```text
完整输入 → 生成侨批 JSON + 图片
部分缺失输入 → 默认补全
严重缺失输入 → ask_clarification 追问
封面字段自动生成
正文竖排渲染
封面竖排渲染
渲染阶段简体转繁体
```

示例输入：

```text
1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。
```

示例输出目录：

```text
outputs/render_demo_20260512_154628/
├── result.json
├── cover.png
└── text.png
```

---

## 3. 项目结构

```text
.
├── configs/                  # 配置文件
├── data/                     # 数据集、模板图、processed JSONL
├── fonts/                    # 渲染字体
├── knowledge/                # 地名/身份旧称知识库
├── models/                   # 早期模型封装
├── outputs/                  # 训练输出和渲染输出
├── pipeline/                 # 推理、后处理、渲染模块
├── scripts/                  # 数据转换、训练、测试、渲染脚本
├── app.py                    # 项目总入口预留
├── requirements.txt
└── structure.txt
```

核心文件：

```text
data/qiaopi_dataset_100_clean_v1.xlsx
scripts/build_sft_dataset.py
scripts/train_qwen25_qlora.py
pipeline/inference.py
pipeline/render_cover.py
pipeline/render_text.py
scripts/run_infer_and_render.py
```

---

## 4. 数据集

第一版数据源：

```text
data/qiaopi_dataset_100_clean_v1.xlsx
```

包含 4 个 sheet：

```text
Main
tags
relationship_rules
greeting_phrase
```

固定 20 个核心标签：

```text
报平安
寄款
思亲
家用
问候父母
问候家中
劝学
劝勤俭
说明近况
工作谋生
生病问候
婚嫁
丧事
添丁
建房修屋
债务
收成田园
节庆问候
托人带信
承诺再寄
```

---

## 5. 数据转换

从 Excel 生成 SFT JSONL：

```bash
python scripts/build_sft_dataset.py   --input data/qiaopi_dataset_100_clean_v1.xlsx   --output_dir data/processed   --no_rewrite
```

生成：

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/processed/all.jsonl
data/processed/summary.json
```

检查数据：

```bash
python scripts/check_sft_jsonl.py
```

---

## 6. 训练

当前使用：

```text
Qwen2.5-7B-Instruct + QLoRA
```

基座模型路径示例：

```text
/data/luozetong/models/Qwen2.5-7B-Instruct
```

训练命令：

```bash
python scripts/train_qwen25_qlora.py   --model_name_or_path /data/luozetong/models/Qwen2.5-7B-Instruct   --train_file data/processed/train.jsonl   --val_file data/processed/val.jsonl   --output_dir outputs/qwen25-7b-qiaopi-qlora-v2   --max_length 2048   --num_train_epochs 3   --learning_rate 1e-4   --per_device_train_batch_size 1   --gradient_accumulation_steps 8   --lora_rank 8   --lora_alpha 16   --lora_dropout 0.05   --logging_steps 5   --eval_steps 20   --save_steps 20
```

当前推荐 adapter：

```text
outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter
```

---

## 7. 推理测试

只测试结构化 JSON：

```bash
PYTHONPATH=. python scripts/test_pipeline_infer.py   --base_model /data/luozetong/models/Qwen2.5-7B-Instruct   --adapter outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter
```

---

## 8. 完整推理 + 渲染

生成 `result.json + cover.png + text.png`：

```bash
python scripts/run_infer_and_render.py   --query "1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。"
```

严重信息不足时：

```bash
python scripts/run_infer_and_render.py   --query "帮我生成一个侨批。"
```

返回：

```json
{
  "action": "ask_clarification",
  "missing_fields": ["sender_place", "receiver_place", "receiver_role", "theme"],
  "question": "请补充寄出地、收批地、收批人身份和主要内容。例如：新加坡寄给汕头母亲，寄二十元，报平安。"
}
```

此时不会生成图片。

---

## 9. 渲染模块

### 封面渲染

文件：

```text
pipeline/render_cover.py
```

输入：

```json
{
  "right_text": "汕头埠慈亲大人收",
  "center_text": "家慈亲大人安启",
  "left_text": "外付银二十元男阿明寄"
}
```

模板：

```text
data/cover_template.png
```

输出：

```text
cover.png
```

### 正文渲染

文件：

```text
pipeline/render_text.py
```

输入：

```json
{
  "salutation": "...",
  "body_text": "...",
  "closing": "...",
  "date": "...",
  "signature": "..."
}
```

模板：

```text
data/text_template.png
```

输出：

```text
text.png
```

### 字体说明

当前字体：

```text
fonts/MasaFont-Regular.ttf
```

该字体偏繁体/旧字形，部分简体字缺失。因此系统设计为：

```text
推理阶段：简体 JSON
保存阶段：result.json 保留简体
渲染阶段：自动转繁体
```

建议安装：

```bash
pip install opencc-python-reimplemented
```

---

## 10. 实验版本

### smoke

```text
outputs/qwen25-7b-qiaopi-smoke/
```

用于确认训练流程能跑通。

### v1

```text
outputs/qwen25-7b-qiaopi-qlora-v1/
```

第一轮正式训练，JSON 基本可用，但信息不足场景不稳定。

### v2

```text
outputs/qwen25-7b-qiaopi-qlora-v2/
```

第二轮训练，加入任务规则，当前推荐作为默认版本。

---

## 11. 后续改进计划

### 数据层

```text
扩展 ask_clarification 样本到 20–30 条
增加更多地区、人物关系、金额表达
增加更多真实侨批风格句式
减少训练集细节复用
```

### 模型层

```text
对比 checkpoint-80 和 final_adapter
尝试 lora_rank=16
尝试 cutoff_len=4096
增加更严格 schema 输出样本
```

### 推理层

```text
拆分 normalizer.py
完善 validator.py
增强 tags 后处理
支持多页正文分页
支持多封面模板
```

### 工程层

```text
接入 app.py
提供 CLI / Web UI
将路径配置写入 configs/model_config.yaml
整理 demo 图片
完善 .gitignore
```

---

## 12. Git 提交建议

建议提交：

```text
configs/
knowledge/
models/
pipeline/
scripts/
requirements.txt
README.md
structure.txt
```

不建议提交：

```text
outputs/
__pycache__/
*.safetensors
*.pt
*.bin
```

建议 `.gitignore`：

```gitignore
__pycache__/
*.pyc

outputs/
!outputs/.gitkeep

*.safetensors
*.pt
*.bin

data/processed/
data/*.xlsx

.env
.venv/
```

---

## 13. 一句话总结

本项目第一版已经完成：

```text
侨批 Excel 数据集
→ SFT 数据转换
→ Qwen2.5-7B QLoRA 微调
→ 结构化 JSON 生成
→ 规则后处理
→ 封面和正文模板渲染
```

下一阶段重点是提升数据多样性、减少幻觉细节、增强 validator/normalizer，并逐步接入正式应用入口。
