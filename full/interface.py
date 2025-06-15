import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

model_path = "/root/autodl-fs/models2/checkpoint-1000"
test_data_path = "./data/test.json"
output_file_path = "./data/output.txt"

# --- 1. 加载原始的基础模型和分词器 ---
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
base_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 如果原始分词器没有 pad_token，并且训练时添加了，这里也需要添加
if base_tokenizer.pad_token is None:
    base_tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})
    base_model.resize_token_embeddings(
        len(base_tokenizer)
    )  # 如果在训练时改变了词表大小，推理时基础模型也需要同步

# --- 2. 将模型移动到 GPU 并设置为评估模式 ---
base_model.to("cuda:0")  # 推理时手动指定设备
base_model.eval()  # 设置为评估模式

# --- 3. 定义系统指令 ---
system_instruction = (
    "你是一个信息抽取助手，你的任务是从社交媒体评论中抽取所有符合要求的仇恨四元组，格式为：评论对象 | 论点 | 目标群体 | 是否仇恨。\n"
    "四元组解释如下：\n"
    "评论对象：被评论或攻击的人或群体，如“你”、“老黑”；若没有明确对象则填 NULL。\n"
    "论点：对评论对象的描述或攻击内容，如“蠢驴”、“倒贴”。\n"
    "目标群体：包含“Region”、“Racism”、“Sexism”、“LGBTQ”、“others”五类，若不涉及仇恨言论则填 non-hate。\n"
    "是否仇恨：若该评论对象-论点对构成仇恨言论则为 hate，否则为 non-hate。\n"
    "输出格式要求：每个四元组用 ' | ' 分隔，以 [END] 结束；若有多个四元组，之间用 [SEP] 分割。\n"
)

# --- 4. 加载测试数据 ---
print(f"加载测试数据: {test_data_path}")
test_dataset = load_dataset("json", data_files=test_data_path)["train"]
print(f"测试集包含 {len(test_dataset)} 个样本。")

# --- 7. 对测试集进行推理并保存结果 ---
print(f"将模型输出保存到文件: {output_file_path}")
with open(output_file_path, "w", encoding="utf-8") as f_out:
    for i, example in enumerate(test_dataset):
        user_input_content = example.get(
            "content", ""
        )  # 获取 content 字段，如果不存在则为空字符串
        test_prompt = (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\n请从下面这段社交媒体文本中抽取仇恨四元组：\n{user_input_content}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        # 分词
        test_inputs = base_tokenizer(
            test_prompt, return_tensors="pt", truncation=True, max_length=512
        )
        test_input_ids = test_inputs.input_ids.to("cuda:0")
        test_attention_mask = test_inputs.attention_mask.to("cuda:0")

        # 生成
        with torch.no_grad():
            output_tokens = base_model.generate(
                input_ids=test_input_ids,
                attention_mask=test_attention_mask,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=base_tokenizer.eos_token_id,
                pad_token_id=base_tokenizer.pad_token_id
                if base_tokenizer.pad_token_id is not None
                else base_tokenizer.eos_token_id,
            )

        # 解码生成结果
        generated_text = base_tokenizer.decode(
            output_tokens[0], skip_special_tokens=False
        )

        # 提取助手的回复
        assistant_start_tag = "<|im_start|>assistant\n"
        extracted_output = ""
        if assistant_start_tag in generated_text:
            extracted_output = generated_text.split(assistant_start_tag, 1)[1]
            if "<|im_end|>" in extracted_output:
                extracted_output = extracted_output.split("<|im_end|>", 1)[0].strip()

        # 将提取出的结果写入文件，每行一个样本的输出
        f_out.write(extracted_output + "\n")
        f_out.flush()

        if (i + 1) % 10 == 0:  # 每处理10个样本打印一次进度
            print(f"已处理 {i + 1}/{len(test_dataset)} 个样本...")

print(f"\n所有测试样本的输出已保存到 {output_file_path}")
print("--- 推理完成 ---")
