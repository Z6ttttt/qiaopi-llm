# qiaopi-llm

侨批生成与渲染系统：结构化文本生成、关系增强微调与封面/正文模板渲染实验项目。

本项目的核心思路不是直接训练图像生成模型，而是训练一个**侨批结构化文本生成模型**：用户输入现代自然语言需求，模型生成结构化 JSON，再由规则后处理和模板渲染模块生成封面图与正文图。

当前第二版已经完成从 v1 基础链路到 v3 关系增强模型的升级：

```text
用户输入
→ Qwen2.5-7B-Instruct + LoRA adapter
→ 结构化侨批 JSON
→ 规则后处理与字段校验
→ 渲染阶段简体转繁体
→ 封面模板渲染 + 正文模板渲染
→ result.json + cover.png + text.png
```

---

## 1. 当前版本状态

### v1：基础链路验证

第一版主要验证完整工程链路能否跑通：

```text
Excel 数据集
→ SFT JSONL 转换
→ Qwen2.5-7B QLoRA 微调
→ 结构化 JSON 生成
→ 封面/正文渲染
```

### v2：格式稳定性增强

第二轮 adapter 在训练 prompt 中加入任务规则，提升 JSON schema 稳定性和信息不足场景的格式控制。

默认历史路径：

```text
outputs/qwen25-7b-qiaopi-qlora-v2/final_adapter
```

### v3：关系增强版本

v3 使用关系增强数据集 `processed_v2_relation` 进行训练，重点提升多种人物关系下的称谓、署名、封面字段和标签稳定性。

当前推荐默认 adapter：

```text
outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter
```

当前 v3 关系测试结果：

```text
total = 11
passed = 11
failed = 0
```

覆盖关系包括：

```text
母子
父子
父母子
夫妻
兄弟
叔侄
祖孙
朋友
```

---

## 2. 项目能力

当前系统支持：

```text
完整输入 → 生成侨批 JSON + 封面图 + 正文图
部分缺失输入 → 默认补全姓名/金额等非关键字段
严重缺失输入 → ask_clarification 追问
多人物关系 → 自动匹配称谓、署名、封面字段
固定 tags → 自动补强生病问候、节庆问候、家用、寄款等标签
封面渲染 → 竖排文字 + 手写扰动 + 墨迹效果
正文渲染 → 竖排正文 + 日期署名栏位
渲染阶段 → 简体自动转繁体，适配 MasaFont 字体
```

示例输入：

```text
1930年代，阿明从新加坡寄二十元给汕头母亲，报平安并让家里买米。
```

示例输出目录：

```text
outputs/render_demo_*/
├── result.json
├── cover.png
└── text.png
```

---

## 3. 项目结构

当前目录结构中的关键部分：

```text
.
├── configs/                         # 配置文件
├── data/                            # 数据集、模板图、processed JSONL
│   ├── cover_template.png           # 封面模板图
│   ├── text_template.png            # 正文模板图
│   ├── processed/                   # v1 processed SFT 数据
│   └── processed_v2_relation/       # v2 关系增强清洗数据与 SFT JSONL
├── fonts/                           # 渲染字体
│   └── MasaFont-Regular.ttf
├── knowledge/                       # 地名/身份旧称知识库
├── models/                          # 早期模型封装与 prompt 模块
├── outputs/                         # 本地训练输出和渲染输出，不建议提交 GitHub
├── pipeline/                        # 正式推理、后处理、渲染模块
│   ├── inference.py                 # 主推理 pipeline
│   ├── render_cover.py              # 封面渲染
│   ├── render_text.py               # 正文渲染
│   ├── text_convert.py              # 渲染阶段简体转繁体
│   ├── normalizer.py
│   └── validator.py
├── scripts/                         # 数据转换、训练、测试、完整推理脚本
│   ├── build_sft_dataset.py
│   ├── check_sft_jsonl.py
│   ├── train_qwen25_qlora.py
│   ├── run_infer_and_render.py
│   ├── test_relation_v3.py
│   └── patch_inference_v3.py
├── app.py                           # 项目总入口预留
├── requirements.txt
└── README.md
```

当前仓库中还可能存在：

```text
outputs/qwen25-7b-qiaopi-qlora-v*/
outputs/render_demo_*/
outputs/v3_relation_eval/
__pycache__/
*.bak_v3_patch
```

这些是本地实验产物或缓存，通常不应上传到 GitHub。

---

## 4. 数据集

### 4.1 v1 基础数据

源文件：

```text
data/qiaopi_dataset_100_clean_v1.xlsx
```

包含：

```text
Main
tags
relationship_rules
greeting_phrase
```

### 4.2 v2 关系增强数据

当前第二版关系增强数据已经清洗并转换为 JSONL：

```text
data/processed_v2_relation/
├── all.jsonl
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── summary.json
├── cleaning_report.json
├── qiaopi_dataset_v2_relation_clean_cases.json
├── qiaopi_dataset_v2_relation_clean_cases.jsonl
└── README.md
```

关系增强数据覆盖：

```text
母子、父子、父母子、夫妻、兄弟、叔侄、祖孙、朋友
```

任务类型包括：

```text
qiaopi_tagging
user_to_qiaopi_body
modern_to_qiaopi_style
ask_clarification
```

### 4.3 固定 tags

当前固定 20 个核心标签：

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

## 5. 环境与依赖

建议 Python 环境：

```text
Python 3.10+
CUDA 环境可用
A6000 / 4090 / 3090 等显卡均可用于 QLoRA 实验
```

核心依赖：

```bash
pip install torch transformers peft accelerate bitsandbytes
pip install pandas openpyxl Pillow fonttools
pip install opencc-python-reimplemented
```

其中：

```text
transformers / peft / bitsandbytes：模型训练和 LoRA 加载
pandas / openpyxl：读取 Excel 数据集
Pillow：封面和正文图片渲染
fonttools：检查字体字库覆盖
opencc-python-reimplemented：渲染阶段简体转繁体
```

---

## 6. 数据转换

### 6.1 v1 Excel 转 SFT JSONL

```bash
python scripts/build_sft_dataset.py \
  --input data/qiaopi_dataset_100_clean_v1.xlsx \
  --output_dir data/processed \
  --no_rewrite
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

### 6.2 v2 关系增强数据

`data/processed_v2_relation/` 已经包含可直接训练的 JSONL，因此训练 v3 时不需要再跑 `build_sft_dataset.py`。

---

## 7. 训练

当前使用：

```text
Qwen2.5-7B-Instruct + QLoRA
```

基座模型路径示例：

```text
/data/luozetong/models/Qwen2.5-7B-Instruct
```

### 7.1 v3 关系增强训练命令

```bash
python scripts/train_qwen25_qlora.py \
  --model_name_or_path /data/luozetong/models/Qwen2.5-7B-Instruct \
  --train_file data/processed_v2_relation/train.jsonl \
  --val_file data/processed_v2_relation/val.jsonl \
  --output_dir outputs/qwen25-7b-qiaopi-qlora-v3 \
  --max_length 2048 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --logging_steps 5 \
  --eval_steps 20 \
  --save_steps 20
```

训练完成后推荐使用：

```text
outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter
```

---

## 8. 推理与渲染

### 8.1 完整推理 + 渲染

```bash
python scripts/run_infer_and_render.py \
  --base_model /data/luozetong/models/Qwen2.5-7B-Instruct \
  --adapter outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter \
  --query "1930年代，阿明从新加坡寄二十元给汕头母亲，报平安并让家里买米。"
```

输出目录类似：

```text
outputs/render_demo_*/
├── result.json
├── cover.png
└── text.png
```

### 8.2 信息不足时追问

```bash
python scripts/run_infer_and_render.py \
  --adapter outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter \
  --query "帮我生成一个侨批。"
```

预期返回：

```json
{
  "action": "ask_clarification",
  "missing_fields": ["sender_place", "receiver_place", "receiver_role", "theme"],
  "question": "请补充寄出地、收批地、收批人身份和主要内容。例如：新加坡寄给汕头母亲，寄二十元，报平安。"
}
```

这种情况下不会生成 `cover.png` 和 `text.png`。

---

## 9. v3 关系测试

运行：

```bash
python scripts/test_relation_v3.py \
  --base_model /data/luozetong/models/Qwen2.5-7B-Instruct \
  --adapter outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter
```

当前测试结果：

```json
{
  "total": 11,
  "passed": 11,
  "failed": 0,
  "adapter": "outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter"
}
```

如需同时渲染前 5 个测试样本：

```bash
python scripts/test_relation_v3.py \
  --base_model /data/luozetong/models/Qwen2.5-7B-Instruct \
  --adapter outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter \
  --render_first_n 5
```

输出：

```text
outputs/v3_relation_eval/
├── results.json
├── summary.json
├── render_mother/
├── render_father/
├── render_parents/
├── render_wife/
└── render_brother/
```

---

## 10. 渲染模块说明

### 10.1 封面渲染

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

输出：

```text
cover.png
```

### 10.2 正文渲染

文件：

```text
pipeline/render_text.py
```

输入：

```json
{
  "salutation": "慈亲大人膝下敬禀者",
  "body_text": "...",
  "closing": "专此奉闻，顺叩福安",
  "date": "民国三十四年春月",
  "signature": "男阿明"
}
```

输出：

```text
text.png
```

### 10.3 简体转繁体

文件：

```text
pipeline/text_convert.py
```

设计原则：

```text
训练阶段：使用简体
推理阶段：输出简体 JSON
result.json：保留简体，便于检查和程序处理
渲染阶段：写入图片前自动转繁体
```

当前字体：

```text
fonts/MasaFont-Regular.ttf
```

该字体偏繁体/旧字形，部分简体字缺失。因此渲染前需要通过 OpenCC 转繁体：

```bash
pip install opencc-python-reimplemented
```

---

## 11. 输出 JSON 格式

### 11.1 generate

```json
{
  "action": "generate",
  "completion_mode": "user_specified",
  "metadata": {
    "sender_name": "阿明",
    "sender_place_modern": "新加坡",
    "sender_place_old": "星洲",
    "sender_role_modern": "儿子",
    "sender_role_old": "男",
    "receiver_place_modern": "汕头",
    "receiver_place_old": "汕头埠",
    "receiver_role_modern": "母亲",
    "receiver_role_old": "慈亲大人",
    "relationship": "母子",
    "amount_modern": "二十元",
    "amount_old": "银二十元",
    "amount_value": 20,
    "tags": ["报平安", "寄款", "家用"],
    "extra_tags": ["买米"]
  },
  "body_fields": {
    "salutation": "慈亲大人膝下敬禀者",
    "body_text": "...",
    "closing": "专此奉闻，顺叩福安",
    "date": "民国三十四年春月",
    "signature": "男阿明"
  },
  "cover_fields": {
    "right_text": "汕头埠慈亲大人收",
    "center_text": "家慈亲大人安启",
    "left_text": "外付银二十元男阿明寄"
  },
  "rendering": {
    "cover_template": "cover_vertical_v1",
    "body_template": "body_vertical_v1",
    "writing_direction": "vertical",
    "pagination": "auto"
  }
}
```

### 11.2 ask_clarification

```json
{
  "action": "ask_clarification",
  "missing_fields": ["sender_place", "receiver_place", "receiver_role", "theme"],
  "question": "请补充寄出地、收批地、收批人身份和主要内容。例如：新加坡寄给汕头母亲，寄二十元，报平安。"
}
```

---

## 12. GitHub 上传建议

本项目包含训练输出和模型权重，直接上传整个目录会非常大。建议上传**代码和轻量配置**，不要提交大模型权重、checkpoint 和本地渲染结果。

### 12.1 建议提交

```text
configs/
knowledge/
models/
pipeline/
scripts/
app.py
requirements.txt
README.md
structure.md
```

可选提交：

```text
data/cover_template.png
data/text_template.png
data/processed_v2_relation/README.md
data/processed_v2_relation/summary.json
data/processed_v2_relation/cleaning_report.json
```

是否提交字体文件请先确认字体授权：

```text
fonts/MasaFont-Regular.ttf
```

### 12.2 不建议提交

```text
outputs/
__pycache__/
*.safetensors
*.pt
*.bin
*.pyc
*.bak_v3_patch
```

尤其是：

```text
outputs/qwen25-7b-qiaopi-qlora-v*/
outputs/qwen25-7b-qiaopi-smoke/
```

这些目录包含 LoRA 权重、optimizer、scheduler、checkpoint 等大文件，不适合直接传 GitHub。

---

## 13. 推荐 .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd

# Environments
.env
.venv/
venv/
conda-meta/

# Training outputs and model weights
outputs/
*.safetensors
*.bin
*.pt
*.pth
*.ckpt

# Local generated files
*.bak_v3_patch
*.bak_text_convert
*.log

# Optional large/intermediate data
data/processed/
data/processed_v2_relation/*.jsonl
data/processed_v2_relation/qiaopi_dataset_v2_relation_clean_cases.json
data/processed_v2_relation/qiaopi_dataset_v2_relation_clean_cases.jsonl
data/*.zip

# Keep lightweight data docs if needed
!data/processed_v2_relation/README.md
!data/processed_v2_relation/summary.json
!data/processed_v2_relation/cleaning_report.json

# OS / editor
.DS_Store
.vscode/
.idea/
```

---

## 14. GitHub 上传流程

### 14.1 初始化仓库

```bash
git init
```

### 14.2 添加远程仓库

```bash
git remote add origin https://github.com/Z6ttttt/qiaopi-llm.git
```

如果远程已经存在：

```bash
git remote set-url origin https://github.com/Z6ttttt/qiaopi-llm.git
```

### 14.3 检查待提交文件

```bash
git status
```

重点确认没有这些文件被加入：

```text
outputs/
*.safetensors
optimizer.pt
scheduler.pt
rng_state.pth
```

### 14.4 提交

```bash
git add .
git commit -m "Initial qiaopi-llm v3 relation-enhanced pipeline"
```

### 14.5 推送

```bash
git branch -M main
git push -u origin main
```

---

## 15. 后续改进计划

### 数据层

```text
继续扩展真实或半真实侨批语料
增加更多地域旧称
增加更多复杂家庭关系
增加更多信息不足追问样本
减少模型对训练集固定细节的复用
```

### 模型层

```text
尝试 lora_rank=16
尝试 max_length=4096
对比 v3 checkpoint-120 / checkpoint-129 / final_adapter
增加 schema 严格约束样本
```

### 推理层

```text
完善 pipeline/validator.py
拆分更多 normalizer 规则
增强金额、地名、关系的后处理
支持更严格的 JSON schema 校验
```

### 渲染层

```text
支持多页正文分页
支持更多封面模板
优化竖排字距和列距
增加字体 fallback
增加图像老化、纸张纹理、墨迹随机性
```

### 工程层

```text
接入 app.py
提供 CLI 或 Web UI
将模型路径和模板路径迁移到 configs/model_config.yaml
整理 demo 图片到 docs/demo/
编写自动测试脚本
```

---

## 16. 一句话总结

当前项目已经完成第二阶段实验：

```text
v1 基础侨批数据
→ v2 关系增强数据清洗
→ v3 QLoRA 关系增强训练
→ 11/11 关系测试通过
→ 简体 JSON 推理 + 繁体图像渲染
→ 封面和正文小样本生成成功
```

当前推荐默认方案：

```text
模型：outputs/qwen25-7b-qiaopi-qlora-v3/final_adapter
推理：pipeline/inference.py
渲染：pipeline/render_cover.py + pipeline/render_text.py
文字转换：pipeline/text_convert.py
测试：scripts/test_relation_v3.py
```
