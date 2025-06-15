import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# --- 8. 如何加载和使用 LoRA 适配器进行推理 ---
print("\n--- 演示如何加载 LoRA 适配器进行推理 ---")

# 首先加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/qwen_lora_output",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # device_map="auto" # 推理时可以设置 device_map="auto"
)
base_tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/qwen_lora_output", trust_remote_code=True
)

# 如果原始分词器没有 pad_token，并且训练时添加了，这里也需要添加
if base_tokenizer.pad_token is None:
    base_tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})
    # base_model.resize_token_embeddings(len(base_tokenizer)) # 如果在训练时改变了词表大小，推理时基础模型也需要同步

# 加载 LoRA 适配器
lora_adapter_model = PeftModel.from_pretrained(
    base_model, "/root/autodl-tmp/qwen_lora_adapter"
)
lora_adapter_model = (
    lora_adapter_model.merge_and_unload()
)  # 合并 LoRA 权重到基础模型，方便推理

# 将模型移动到 GPU (如果有多卡，Trainer 会自动处理)
lora_adapter_model.to("cuda:0")  # 推理时手动指定设备
lora_adapter_model.eval()  # 设置为评估模式

# 准备一个测试输入
test_input = "这个人说的话毫无道理。"
# 或者一个仇恨言论的例子
# test_input = "这些河南人都骗子，滚出去！"

system_instruction = (
    "你是一个信息抽取助手，你的任务是从社交媒体评论中抽取**所有**符合要求的仇恨四元组，格式为：评论对象 | 论点 | 目标群体 | 是否仇恨。\n"
    "四元组解释如下：\n"
    "评论对象（Target）：被评论或攻击的人或群体，如“你”、“老黑”；若没有明确对象则填 NULL。\n"
    "论点（Argument）：对评论对象的描述或攻击内容，如“蠢驴”、“倒贴”。\n"
    "目标群体（Targeted Group）：包含“地域”、“种族”、“性别”、“LGBTQ”、“其他”五类，若不涉及仇恨言论则填 non-hate。\n"
    "是否仇恨（Hateful）：若该评论对象-论点对构成仇恨言论则为 hate，否则为 non-hate。\n"
    "输出格式要求：每个四元组用 ' | ' 分隔，以 [END] 结束；若有多个四元组，之间用 [SEP] 分割。\n"
)

test_prompt = (
    f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
    f"<|im_start|>user\n请从下面这段社交媒体文本中抽取仇恨四元组：\n{test_input}\n<|im_end|>\n"
    f"<|im_start|>assistant\n"  # 注意这里，我们希望模型生成 assistant 的回复
)

# 分词
test_inputs = base_tokenizer(
    test_prompt, return_tensors="pt", truncation=True, max_length=512
)
test_input_ids = test_inputs.input_ids.to("cuda:0")
test_attention_mask = test_inputs.attention_mask.to("cuda:0")

# 生成
print(f"\n测试输入: {test_input}")
with torch.no_grad():
    output_tokens = lora_adapter_model.generate(
        input_ids=test_input_ids,
        attention_mask=test_attention_mask,
        max_new_tokens=200,  # 生成的最大新 token 数量
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=base_tokenizer.eos_token_id,  # Qwen 的结束符
        pad_token_id=base_tokenizer.pad_token_id
        if base_tokenizer.pad_token_id is not None
        else base_tokenizer.eos_token_id,  # 如果没有pad_token，就用eos_token_id
    )

# 解码生成结果
# 找到 assistant 标记的起始位置，只解码其后的内容
generated_text = base_tokenizer.decode(output_tokens[0], skip_special_tokens=False)
print(f"生成的完整文本:\n{generated_text}")

# 提取助教的回复
assistant_start_tag = "<|im_start|>assistant\n"
if assistant_start_tag in generated_text:
    extracted_output = generated_text.split(assistant_start_tag, 1)[1]
    # 移除 <|im_end|> 及其之后的内容
    if "<|im_end|>" in extracted_output:
        extracted_output = extracted_output.split("<|im_end|>", 1)[0].strip()
    print(f"\n抽取出的仇恨四元组:\n{extracted_output}")
else:
    print("\n未能在生成结果中找到助教的回复。")
