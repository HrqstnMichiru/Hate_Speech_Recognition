import json


def generate_instruction(output_str) -> str:
    """
    根据输出四元组字符串生成对应的 instruction。
    假设输出格式为：评论对象 | 论点 | 目标群体 | 是否仇恨 [END] [SEP] ...
    我们主要关注第一个四元组的“目标群体”和“是否仇恨”。
    """
    # 尝试解析第一个四元组
    if "[SEP]" in output_str:
        first_quadruplet = output_str.split(" [SEP] ")[0]
    else:
        first_quadruplet = output_str.split(" [END]")[0]
    print(first_quadruplet)
    parts = [p.strip() for p in first_quadruplet.split(" | ")]

    if len(parts) != 4:
        # 如果解析失败，返回一个通用 instruction，或者直接报错
        print(
            f"Warning: Could not parse quadruplet from output: {output_str}. Returning generic instruction."
        )
        return "请从下面这段社交媒体文本中抽取仇恨四元组。"

    # target = parts[0] # 评论对象
    # argument = parts[1] # 论点
    target_group = parts[2]  # 目标群体
    hateful_status = parts[3]  # 是否仇恨

    if hateful_status.lower() == "hate":
        if target_group.lower() == "region":
            return "请识别评论中包含的**地域仇恨言论**并抽取仇恨四元组。"
        elif target_group.lower() == "racism":
            return "请识别评论中包含的**种族仇恨言论**并抽取仇恨四元组。"
        elif target_group.lower() == "sexism":
            return "请识别评论中包含的**性别歧视仇恨言论**并抽取仇恨四元组。"
        elif target_group.lower() == "lgbtq":
            return "请识别评论中包含的**LGBTQ+仇恨言论**并抽取仇恨四元组。"
        elif target_group.lower() == "others":
            return "请识别评论中包含的**特定群体仇恨言论**并抽取仇恨四元组。"
        elif (
            target_group.lower() == "null"
        ):  # 如果是 hate 但目标群体为 NULL，可能是对个人的强烈攻击
            return "请识别评论中包含的**潜在仇恨或强烈攻击性言论**并抽取四元组。"
        else:  # 兜底，以防意外的 target_group
            return "请识别评论中包含的**仇恨言论**并抽取仇恨四元组。"
    elif hateful_status.lower() == "non-hate":
        if target_group.lower() == "non-hate":
            return "请分析以下评论，判断**是否包含仇恨言论**并抽取四元组。"
        else:  # 尽管是 non-hate 但可能依然有目标群体，比如中性描述
            return "请从以下评论中抽取四元组，如果不存在仇恨言论请标记为非仇恨。"
    else:  # 兜底，以防意外的 hateful_status
        return "请从下面这段社交媒体文本中抽取仇恨四元组。"


def transform_dataset(input_file_path, output_file_path) -> None:
    """
    读取原始 JSONL 数据集，为每个样本添加 'instruction' 字段。

    Args:
        input_file_path (str): 原始数据集文件路径 (e.g., './data/train.json').
        output_file_path (str): 转换后数据集的保存路径 (e.g., './data/train_transformed.json').
    """
    transformed_data = []

    # 确保输入文件存在
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    try:
        with open(input_file_path, "r", encoding="utf-8") as infile:
            # 使用 json.load() 一次性读取整个文件
            data = json.load(infile)

        if not isinstance(data, list):
            print("Error: The input JSON file is not a list of objects.")
            return
        for line_num, sample in enumerate(data):  # 此时 line_num 是列表索引
            if "output" not in sample:
                print(
                    f"Warning: Sample at index {line_num} is missing 'output' field. Skipping or using generic instruction."
                )
                sample["instruction"] = "请从下面这段社交媒体文本中抽取仇恨四元组。"
                transformed_data.append(sample)
                continue

            sample["instruction"] = generate_instruction(sample["output"])
            transformed_data.append(sample)

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON on line {line_num + 1}: {line.strip()} - {e}")
    except Exception as e:
        print(
            f"An unexpected error occurred on line {line_num + 1}: {line.strip()} - {e}"
        )

    # 将转换后的数据写入新的 JSON 文件，格式为 JSON 数组
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file_path, "w", encoding="utf-8") as outfile:
        # 使用 json.dump() 将整个列表作为 JSON 数组写入
        json.dump(
            transformed_data, outfile, ensure_ascii=False, indent=4
        )  # indent=4 可以使输出更易读

    print(f"\n数据集转换完成！已处理 {len(transformed_data)} 条样本。")
    print(f"转换后的数据已保存到: {output_file_path}")


# --- 配置和运行 ---
if __name__ == "__main__":
    import os

    # 定义原始数据文件路径
    original_data_path = "./data/train.json"
    # 定义转换后数据保存路径
    transformed_data_path = "./data/train_transformed.json"

    # 创建示例数据文件（如果不存在）
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists(original_data_path):
        print(f"创建示例数据文件: {original_data_path}")
        example_data = [
            {
                "instruction": "",
                "input": "他就是一个白痴，这种人都应该被关起来。",
                "output": "他 | 白痴 | non-hate | non-hate [END]",
            },
            {
                "instruction": "",
                "input": "这些南方人就知道吃狗肉，素质真低。",
                "output": "南方人 | 吃狗肉 | 地域 | hate [END] [SEP] 南方人 | 素质真低 | 地域 | hate [END]",
            },
            {
                "instruction": "",
                "input": "死gay，看你那娘炮样。",
                "output": "死gay | 娘炮样 | LGBTQ | hate [END]",
            },
            {
                "instruction": "",
                "input": "所有男人都是垃圾，都该死。",
                "output": "所有男人 | 垃圾 | 性别 | hate [END]",
            },
            {
                "instruction": "",
                "input": "这个人说的话毫无道理。",
                "output": "这个人 | 毫无道理 | non-hate | non-hate [END]",
            },
            {
                "instruction": "",
                "input": "艾滋病患者就应该被隔离，不应该出现在公共场合。",
                "output": "艾滋病患者 | 应该被隔离 | 其他 | hate [END] [SEP] 艾滋病患者 | 不应该出现在公共场合 | 其他 | hate [END]",
            },
        ]
        with open(original_data_path, "w", encoding="utf-8") as f:
            for item in example_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("示例数据文件创建成功。")

    # 运行数据转换
    transform_dataset(original_data_path, transformed_data_path)

    # 打印转换后的数据的前几行以供检查
    print("\n--- 转换后数据的前几行示例 ---")
    if os.path.exists(transformed_data_path):
        with open(transformed_data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:  # 只打印前5行
                    break
                print(line.strip())
    print("----------------------------")

    print(f"\n您现在可以在您的LoRA微调脚本中使用 '{transformed_data_path}' 文件了。")
