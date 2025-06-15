import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Qwen2.5-7B 的本地路径
model_name = "/root/Qwen2.5/text-generation-webui/models/Qwen2.5-7B-Instruct"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 确保分词器有 pad_token，这对批处理（batching）和 DataCollatorForSeq2Seq 很重要。
# Qwen 分词器通常没有默认的 pad_token。
if tokenizer.pad_token is None:
    # 增加 [PAD] 作为新的特殊 token，并设置其为 pad_token
    # Qwen 通常使用 <|endoftext|> 作为 EOS token，这里我们不改变它
    tokenizer.add_special_tokens(
        {"pad_token": "<|extra_0|>"}
    )  # 或者其他未使用的 extra token，Qwen系列有预留
    print(
        f"Added new pad_token: '{tokenizer.pad_token}' with id: {tokenizer.pad_token_id}"
    )

# 加载基础模型
# 对于多GPU训练，Trainer 会自动处理模型的设备放置，所以不要手动 .to("cuda:0") 或 device_map="auto"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 使用半精度以节省内存
)

if tokenizer.pad_token is not None and model.config.vocab_size < len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} for new pad_token.")


dataset = load_dataset("json", data_files="./data/train.json")


def process_func(example):
    labels = []
    # 构建 Qwen 聊天格式的 prompt
    # 这里将系统指令、用户输入和期望输出拼接起来，形成一个完整的对话
    # 这是 LoRA 微调的标准做法，模型需要看到完整的输入-输出对
    system_instruction = (
        "你是一个信息抽取助手，你的任务是从社交媒体评论中抽取所有符合要求的仇恨四元组，格式为：评论对象 | 论点 | 目标群体 | 是否仇恨。\n"
        "四元组解释如下：\n"
        "评论对象：被评论或攻击的人或群体，如“你”、“老黑”；若没有明确对象则填 NULL。\n"
        "论点：对评论对象的描述或攻击内容，如“蠢驴”、“倒贴”。\n"
        "目标群体：包含“Region”、“Racism”、“Sexsim”、“LGBTQ”、“others”五类，若不涉及仇恨言论则填 non-hate。\n"
        "是否仇恨：若该评论对象-论点对构成仇恨言论则为 hate，否则为 non-hate。\n"
        "输出格式要求：每个四元组用 ' | ' 分隔，以 [END] 结束；若有多个四元组，之间用 [SEP] 分割。\n"
    )
    full_text = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\n请从下面这段社交媒体文本中抽取仇恨四元组：\n{example['content']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )

    # 对拼接后的文本进行分词
    tokenized_output = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",  # 填充到 max_length
        max_length=512,  # 最大序列长度，根据您的数据和GPU显存调整
        return_tensors="pt",  # 确保返回 PyTorch 张量
    )

    # 复制 input_ids 作为 labels
    labels = tokenized_output["input_ids"].clone()

    # 将 padding token id 的位置替换为 -100，以便在计算损失时忽略这些位置
    # 这里要确保 labels 也是 PyTorch tensor
    # 并且 labels 的每个元素都应该是 long 类型
    labels = labels.squeeze().tolist()  # 先转换为列表进行替换
    labels = [(ls if ls != tokenizer.pad_token_id else -100) for ls in labels]
    tokenized_output["labels"] = torch.tensor(labels, dtype=torch.long)  # 再转回张量

    # 返回字典格式，Trainer 需要 input_ids, attention_mask 和 labels
    return {
        "input_ids": tokenized_output["input_ids"].squeeze(),
        "attention_mask": tokenized_output["attention_mask"].squeeze(),
        "labels": tokenized_output["labels"].squeeze(),
    }


tokenized_dataset = dataset["train"].map(
    process_func,
    remove_columns=dataset["train"].column_names,  # 移除原始列，只保留 tokenized 后的列
    batched=False,  # 对于这类复杂拼接，通常逐个样本处理更安全
)

print(f"处理后的数据集大小: {len(tokenized_dataset)}")
# 打印一个样本看看
print("\n--- 预处理后的一个样本 ---")
for key, value in tokenized_dataset[0].items():
    if key == "input_ids" or key == "labels":
        print(f"{key}: {tokenizer.decode([id for id in value if id != -100])}")
    else:
        print(f"{key}: {value}")
        
print("--------------------------")
training_args = TrainingArguments(
    output_dir="/root/autodl-fs/models2",  # 保存模型检查点和日志的目录
    save_steps=500,  # 每 500 步保存一次检查点
    save_total_limit=2,  # 最多只保留一个检查点
    logging_steps=50,  # 每 50 步记录一次日志
    per_device_train_batch_size=2,  # 每个 GPU 的训练批次大小
    num_train_epochs=2,  # 训练的 epoch 数量
    learning_rate=2e-5,  # LoRA 建议的学习率通常比全量微调高一些
    warmup_steps=100,  # 学习率预热步数
    weight_decay=0.01,  # 权重衰减
    gradient_accumulation_steps=4,  # 梯度累积步数，用于增大有效批次大小 (有效批次大小 = per_device_train_batch_size * num_gpus * gradient_accumulation_steps)
    report_to="none",  # 不使用任何报告工具 (如 wandb)，或者可以设置为 "tensorboard" 等
    bf16=True,  # 使用半精度浮点数训练，节省显存，加速训练
    remove_unused_columns=False,  # Trainer 默认会移除未使用的列，但我们处理过的 dataset 已经没有了
)

# DataCollatorForSeq2Seq 负责将样本批量化并进行填充（padding）
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,  # 传入模型，DataCollatorForSeq2Seq 会使用模型的 config 来确定 pad_token_id
    padding=True,  # 动态填充到批次中最长序列的长度
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 开始训练
print("\n--- 开始训练模型 ---")
trainer.train()
print("--- 模型训练完成 ---")
