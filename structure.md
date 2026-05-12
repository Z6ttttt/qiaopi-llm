# qiaopi-llm 文件结构详细注释版

> 本文档基于当前第一版实验目录结构整理，用于说明各文件/目录在“侨批结构化生成 + LoRA 微调 + 模板渲染”流程中的作用。

## 1. 当前第一版主链路

```text
data/qiaopi_dataset_100_clean_v1.xlsx
→ scripts/build_sft_dataset.py
→ data/processed/train.jsonl / val.jsonl / test.jsonl
→ scripts/train_qwen25_qlora.py
→ outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter
→ pipeline/inference.py
→ pipeline/render_cover.py + pipeline/render_text.py
→ scripts/run_infer_and_render.py
→ outputs/render_demo_*/result.json + cover.png + text.png
```

当前第一版已经跑通：

```text
用户输入
→ 结构化侨批 JSON
→ 封面字段生成
→ 正文和封面模板渲染
```

---

## 2. 根目录

```text
.
├── app.py
├── configs/
├── data/
├── fonts/
├── knowledge/
├── models/
├── outputs/
├── pipeline/
├── scripts/
├── src/
├── requirements.txt
└── structure.txt
```

### `app.py`

项目总入口预留文件。

当前第一版主流程还没有完全接入 `app.py`，目前主要通过：

```bash
python scripts/run_infer_and_render.py --query "..."
```

完成完整推理和渲染。后续可以将 `scripts/run_infer_and_render.py` 中的逻辑迁移到 `app.py`，形成正式 CLI 或 Web 服务入口。

### `requirements.txt`

Python 依赖文件。建议固定：

```text
torch
transformers
peft
accelerate
bitsandbytes
pandas
openpyxl
Pillow
fonttools
opencc-python-reimplemented
```

### `structure.txt`

当前目录结构记录文件。建议每次完成一个实验版本后更新。

### `src/`

预留源码目录，当前暂未使用。后续如项目变大，可将正式包代码迁移到 `src/qiaopi_llm/`。

---

## 3. `configs/`

```text
configs/
├── model_config.yaml
├── schema.json
└── train_qwen25_7b_lora.yaml
```

### `model_config.yaml`

模型路径和推理参数配置预留文件。后续可统一管理：

```yaml
base_model: /data/luozetong/models/Qwen2.5-7B-Instruct
adapter: outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter
temperature: 0.0
max_new_tokens: 768
```

### `schema.json`

输出 JSON schema 预留文件。后续可配合 `pipeline/validator.py` 检查输出结构。

### `train_qwen25_7b_lora.yaml`

早期训练配置文件。当前实际训练主要由 `scripts/train_qwen25_qlora.py` 完成。后续可将训练参数迁移回 YAML，方便复现实验。

---

## 4. `data/`

```text
data/
├── cover_template.png
├── text_template.png
├── qiaopi_dataset_100_clean_v1.xlsx
└── processed/
```

### `qiaopi_dataset_100_clean_v1.xlsx`

第一版 100 条侨批数据集，是当前训练数据源。包含四个 sheet：

```text
Main
tags
relationship_rules
greeting_phrase
```

### `cover_template.png`

封面模板图，由 `pipeline/render_cover.py` 使用。

### `text_template.png`

正文模板图，由 `pipeline/render_text.py` 使用。

---

## 5. `data/processed/`

```text
data/processed/
├── all.jsonl
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── summary.json
└── splits/
```

### `train.jsonl`

训练集。由 Excel 转换而来，用于 QLoRA 微调。

### `val.jsonl`

验证集，用于训练过程中计算 `eval_loss`。

### `test.jsonl`

测试集，后续可用于自动评估。

### `all.jsonl`

所有 SFT 样本汇总。

### `summary.json`

数据转换统计文件，记录样本总数、任务数量、固定 tags 等。

### `splits/`

按 `case_id` 切分后的数据划分文件：

```text
train_case_ids.txt
val_case_ids.txt
test_case_ids.txt
```

作用是防止同一侨批案例的派生样本泄漏到不同集合。

---

## 6. `fonts/`

```text
fonts/
└── MasaFont-Regular.ttf
```

### `MasaFont-Regular.ttf`

当前封面和正文渲染字体。

注意：该字体偏繁体/旧字形，部分简体字不完整支持。当前系统采用：

```text
Qwen 推理阶段：简体 JSON
result.json：保留简体
图片渲染阶段：自动转繁体
```

这样既保证模型输出稳定，也避免图片缺字。

---

## 7. `knowledge/`

```text
knowledge/
├── place_alias.json
└── role_alias.json
```

### `place_alias.json`

地名旧称映射知识库，例如：

```text
新加坡 → 星洲
曼谷 → 暹京
汕头 → 汕头埠
潮州 → 潮州府
```

### `role_alias.json`

身份旧称映射知识库，例如：

```text
母亲 → 慈亲大人
父亲 → 严亲大人
儿子 → 男
```

后续可由 `pipeline/normalizer.py` 统一调用。

---

## 8. `models/`

```text
models/
├── __init__.py
├── prompts.py
└── qwen_client.py
```

### `qwen_client.py`

早期 Qwen 推理客户端封装。当前正式推理流程已经主要转向 `pipeline/inference.py`。后续可重构为统一模型加载模块。

### `prompts.py`

Prompt 模板文件。后续可集中管理 `user_to_qiaopi_body`、`qiaopi_tagging`、`ask_clarification` 的 prompt。

### `__pycache__/`

Python 缓存目录，不需要提交 GitHub。

---

## 9. `outputs/`

```text
outputs/
├── qwen25-7b-qiaopi-smoke/
├── qwen25-7b-qiaopi-qlora-v1/
├── qwen25-7b-qiaopi-qlora-v2/
└── render_demo_*/
```

### `qwen25-7b-qiaopi-smoke/`

smoke test 训练输出，只用于确认训练代码、数据格式、模型加载和 adapter 保存流程能跑通。

### `qwen25-7b-qiaopi-qlora-v1/`

第一轮正式 QLoRA 训练结果。JSON 输出基本可用，但信息不足场景不稳定。

### `qwen25-7b-qiaopi-qlora-v2/`

第二轮 QLoRA 训练结果。训练阶段加入了任务规则，当前推荐作为第一版默认 adapter。

核心路径：

```text
outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter
```

核心文件：

```text
adapter_model.safetensors
adapter_config.json
```

### `checkpoint-*`

训练中间保存点。如果只做推理，一般使用 `final_adapter/` 即可。

### `render_demo_*/`

完整推理 + 渲染输出目录。通常包含：

```text
result.json
cover.png
text.png
```

如果目录中只有 `result.json`，通常说明输入被判定为信息不足，返回了 `ask_clarification`，因此没有渲染图片。

---

## 10. `pipeline/`

```text
pipeline/
├── inference.py
├── render_cover.py
├── render_text.py
├── generator.py
├── normalizer.py
└── validator.py
```

### `inference.py`

当前最重要的正式推理 pipeline。

功能：

```text
1. 接收用户自然语言输入
2. 判断是否信息不足
3. 信息不足时直接返回 ask_clarification
4. 信息足够时调用 Qwen2.5 + LoRA adapter
5. 解析模型 JSON
6. 修复默认姓名、默认人物关系、tags
7. 自动生成 cover_fields
8. 自动补 rendering 字段
```

### `render_cover.py`

封面渲染模块。

输入：

```json
{
  "right_text": "汕头埠慈亲大人收",
  "center_text": "家慈亲大人安启",
  "left_text": "外付银二十元男阿明寄"
}
```

输出：

```text
cover.png
```

特点：

```text
竖排文字
随机笔迹扰动
墨迹效果
渲染前简体转繁体
```

### `render_text.py`

正文渲染模块。

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

输出：

```text
text.png
```

特点：

```text
自动计算正文栏位
右起竖排
左侧预留日期和署名
渲染前简体转繁体
```

### `generator.py`

早期生成模块，当前可视为待重构文件。后续可以将其改造成正式入口，内部调用 `inference.py`、`render_cover.py` 和 `render_text.py`。

### `normalizer.py`

旧称和字段标准化模块。后续可将 `inference.py` 中的部分后处理逻辑拆出到这里。

### `validator.py`

JSON 校验模块。后续应适配当前两类输出：

```text
generate
ask_clarification
```

---

## 11. `scripts/`

```text
scripts/
├── build_sft_dataset.py
├── check_sft_jsonl.py
├── train_qwen25_qlora.py
├── run_infer_and_render.py
├── test_pipeline_infer.py
├── test_lora_infer.py
├── test_lora_infer_v2.py
├── test_infer.py
└── test_qwen_client.py
```

### `build_sft_dataset.py`

数据转换脚本。

输入：

```text
data/qiaopi_dataset_100_clean_v1.xlsx
```

输出：

```text
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/processed/all.jsonl
data/processed/summary.json
```

生成任务：

```text
qiaopi_tagging
user_to_qiaopi_body
ask_clarification
```

### `check_sft_jsonl.py`

SFT 数据检查脚本。训练前建议必须运行。

检查内容：

```text
JSONL 每行是否可解析
messages 是否完整
assistant.content 是否为合法 JSON
关键字段是否存在
```

### `train_qwen25_qlora.py`

Qwen2.5-7B-Instruct + QLoRA 训练脚本。

功能：

```text
加载本地 Qwen2.5 基座
4bit 量化
注入 LoRA
读取 SFT JSONL
只对 assistant 输出计算 loss
保存 checkpoint 和 final_adapter
```

### `run_infer_and_render.py`

当前最重要的完整小样本实验脚本。

流程：

```text
用户输入
→ pipeline/inference.py
→ result.json
→ pipeline/render_cover.py
→ cover.png
→ pipeline/render_text.py
→ text.png
```

推荐运行：

```bash
python scripts/run_infer_and_render.py   --query "1930年代，阿明在新加坡写给汕头母亲，寄二十元，报平安，让家里买米。"
```

### `test_pipeline_infer.py`

测试 `pipeline/inference.py` 的结构化推理结果，不生成图片。

### `test_lora_infer.py`

第一版 LoRA 原始推理测试脚本。

### `test_lora_infer_v2.py`

增强版 LoRA 推理测试脚本，包含更强 prompt、schema warning 和更稳的 JSON 解析。

### `test_infer.py`

早期推理测试脚本，可保留作为历史文件。

### `test_qwen_client.py`

早期 Qwen 客户端测试脚本。

---

## 12. 当前核心文件清单

当前第一版实验最重要的文件：

```text
data/qiaopi_dataset_100_clean_v1.xlsx
scripts/build_sft_dataset.py
scripts/check_sft_jsonl.py
scripts/train_qwen25_qlora.py
outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter/
pipeline/inference.py
pipeline/render_cover.py
pipeline/render_text.py
scripts/run_infer_and_render.py
```

后续可重构文件：

```text
app.py
models/qwen_client.py
models/prompts.py
pipeline/generator.py
pipeline/normalizer.py
pipeline/validator.py
configs/*.yaml
```

不建议提交 GitHub 的文件：

```text
__pycache__/
outputs/qwen25-7b-qiaopi-*/checkpoint-*/
*.safetensors
*.pt
*.bin
```
