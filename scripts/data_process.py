# 给json文件添加instruction和system列，格式为json数组
import json


def add_instruction_and_system(input_file, output_file):
    """
    给json文件添加instruction和system列
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :return: None
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    for item in data:
        item["instruction"] = "请从下面这段社交媒体文本中抽取仇恨四元组：\n"
        item["system"] = (
            "你是一个信息抽取助手，你的任务是从社交媒体评论中抽取所有符合要求的仇恨四元组，格式为：评论对象 | 论点 | 目标群体 | 是否仇恨。\n四元组解释如下：\n评论对象：被评论或攻击的人或群体，如“你”、“老黑”；若没有明确对象则填 NULL。\n论点：对评论对象的描述或攻击内容，如“蠢驴”、“倒贴”。\n目标群体：包含“Region”、“Racism”、“Sexism”、“LGBTQ”、“others”五类，若不涉及仇恨言论则填 non-hate。\n是否仇恨：若该评论对象-论点对构成仇恨言论则为 hate，否则为 non-hate。\n输出格式要求：每个四元组用 ' | ' 分隔，以 [END] 结束；若有多个四元组，之间用 [SEP] 分割。"
        )
        del item["id"]
        item["input"] = item["content"]
        del item["content"]
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
    
    print("已成功添加instruction和system列")
    
if __name__ == "__main__":
    # 测试文件路径
    input_file_path = "./data/test.json"
    output_file_path = "./data/test_output.json"
    add_instruction_and_system(input_file_path, output_file_path)