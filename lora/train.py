import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model  # 引入 PEFT 相关的库
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# --- 1. 设置路径和模型 ---
# Qwen2.5-7B 的本地路径
model_name = "./models/"

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
    torch_dtype=torch.float16,  # 使用半精度以节省内存
    # device_map="auto" # 建议不要在这里设置 device_map="auto"，让 Trainer 处理
)

# 如果添加了新的 pad_token，需要调整模型 embedding 的大小以适应新的词汇表
if tokenizer.pad_token is not None and model.config.vocab_size < len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} for new pad_token.")


# --- 2. 配置 LoRA ---
# LoRA 配置，根据 Qwen2.5-7B 的架构选择目标模块
# 常见的可训练模块有 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
# 对于Qwen2系列，通常是 q_proj, k_proj, v_proj, o_proj
lora_config = LoraConfig(
    r=8,  # LoRA 的秩，值越大，模型容量越大，但参数量和计算量也越大
    lora_alpha=16,  # LoRA 缩放因子，通常设置为 2 * r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 指定要应用 LoRA 的模块
    lora_dropout=0.05,  # Dropout 比例
    bias="none",  # 偏置项的训练方式，通常设置为 "none"
    task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
)

# 将 LoRA 配置应用到模型上
model = get_peft_model(model, lora_config)

# 打印模型中可训练参数的数量和比例
model.print_trainable_parameters()
# 期望输出类似：trainable params: 4,194,304 || all params: 7,748,012,288 || trainable%: 0.05413481239853926
# 这表明只有极少量的参数需要训练。

# --- 3. 数据加载和预处理 ---
# 加载你的 jsonl 数据
# 假设数据文件在 './data/train.json'
dataset = load_dataset("json", data_files="./data/train.json")


def process_func(example):
    MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    # 构建 Qwen 聊天格式的 prompt
    # 这里将系统指令、用户输入和期望输出拼接起来，形成一个完整的对话
    # 这是 LoRA 微调的标准做法，模型需要看到完整的输入-输出对
    system_instruction = (
        "你是一个信息抽取助手，你的任务是从社交媒体评论中抽取**所有**符合要求的仇恨四元组，格式为：评论对象 | 论点 | 目标群体 | 是否仇恨。\n"
        "四元组解释如下：\n"
        "评论对象（Target）：被评论或攻击的人或群体，如“你”、“老黑”；若没有明确对象则填 NULL。\n"
        "论点（Argument）：对评论对象的描述或攻击内容，如“蠢驴”、“倒贴”。\n"
        "目标群体（Targeted Group）：包含“地域”、“种族”、“性别”、“LGBTQ”、“其他”五类，若不涉及仇恨言论则填 non-hate。\n"
        "是否仇恨（Hateful）：若该评论对象-论点对构成仇恨言论则为 hate，否则为 non-hate。\n"
        "输出格式要求：每个四元组用 ' | ' 分隔，以 [END] 结束；若有多个四元组，之间用 [SEP] 分割。\n"
    )
    # 完整拼接对话
    full_text = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}\n{example['content']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    # 对拼接后的文本进行分词
    instruction = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
    )  # add_special_tokens 不在开头加 special_tokens

    # 处理分词后的输出，构建输入ID、注意力掩码和标签
    response = tokenizer(
        f"{example['output']}", return_tensors="pt", add_special_tokens=False
    )

    # 构建输入ID、注意力掩码和标签
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    # 注意力掩码需要考虑到 eos token，所以最后补充一个1
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )  # 因为eos token咱们也是要关注的所以 补充为1
    # 标签需要将输入部分的ID替换为-100，表示不计算损失
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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

# --- 4. 设置训练参数 ---
training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/qwen_lora_output",  # 保存模型检查点和日志的目录
    save_steps=1000,  # 每 500 步保存一次检查点
    save_total_limit=1,  # 最多只保留一个检查点
    logging_steps=50,  # 每 50 步记录一次日志
    per_device_train_batch_size=2,  # 每个 GPU 的训练批次大小
    num_train_epochs=5,  # 训练的 epoch 数量
    learning_rate=2e-4,  # LoRA 建议的学习率通常比全量微调高一些
    warmup_steps=100,  # 学习率预热步数
    weight_decay=0.01,  # 权重衰减
    gradient_accumulation_steps=4,  # 梯度累积步数，用于增大有效批次大小 (有效批次大小 = per_device_train_batch_size * num_gpus * gradient_accumulation_steps)
    report_to="none",  # 不使用任何报告工具 (如 wandb)，或者可以设置为 "tensorboard" 等
    fp16=True,  # 使用半精度浮点数训练，节省显存，加速训练
    max_grad_norm=1.0,  # 梯度裁剪，防止梯度爆炸
    remove_unused_columns=False,  # Trainer 默认会移除未使用的列，但我们处理过的 dataset 已经没有了
    # eval_strategy="steps", # 可选：设置评估策略
    # eval_steps=500, # 可选：评估步数
    # load_best_model_at_end=True, # 可选：训练结束时加载最佳模型
)

# --- 5. 数据对齐器 (Data Collator) ---
# DataCollatorForSeq2Seq 负责将样本批量化并进行填充（padding）
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,  # 传入模型，DataCollatorForSeq2Seq 会使用模型的 config 来确定 pad_token_id
    padding=True,  # 动态填充到批次中最长序列的长度
)

# --- 6. 初始化并开始训练 ---
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 开始训练
print("\n--- 开始训练 LoRA 模型 ---")
trainer.train()
print("--- LoRA 模型训练完成 ---")

# --- 7. 保存 LoRA 适配器 ---
# 保存的是 LoRA 适配器，而不是整个基础模型，文件非常小
output_adapter_dir = "/root/autodl-tmp/qwen_lora_adapter"
model.save_pretrained(output_adapter_dir)
tokenizer.save_pretrained(
    output_adapter_dir
)  # 同样保存分词器，因为可能添加了特殊 token
print(f"\nLoRA 适配器已保存到: {output_adapter_dir}")
