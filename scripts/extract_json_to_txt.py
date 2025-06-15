# 从jsonl文件中提取predict字段并保存为txt文件
import json

def extract_json_to_txt(input_file_path, output_file_path):
    """
    从JSONL文件中提取predict字段并保存为TXT文件。
    
    :param input_file_path: 输入的JSONL文件路径 
    :param output_file_path: 输出的TXT文件路径
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                if 'predict' in data:
                    outfile.write(data['predict'] + '\n')
            except json.JSONDecodeError as e:
                print(f"JSON解码错误: {e}，在行: {line}")
                continue
            except KeyError as e:
                print(f"键错误: {e}，在行: {line}")
                continue
            except Exception as e:
                print(f"发生意外错误: {e}，在行: {line}")
                continue
    print(f"提取完成，已保存到: {output_file_path}")
    
if __name__ == "__main__":
    input_file = "data/generated_predictions.jsonl"  # 输入的JSONL文件路径
    output_file = 'data/result.txt'   # 输出的TXT文件路径
    extract_json_to_txt(input_file, output_file)