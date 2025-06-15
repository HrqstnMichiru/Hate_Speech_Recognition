# 写个脚本检查txt文件中的每一行是否符合特定格式
# 每一行中间不应该出现" [END]"，而应该以" [END]"结尾
# 每一行包含多个四元组，四元组之间用" [SEP] "分隔，四元组内部用" | "分隔
# 如果某行出现了non-hate，那么只会有一个四元组
# 所有的注释都用中文
import os


def check_file_format(file_path, output_log_path):
    """
    检查文件格式是否符合要求
    :param file_path: 文件路径
    :return: None
    """
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return

    with open(file_path, "r", encoding="utf-8") as file, open(output_log_path, "w", encoding="utf-8") as log_file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            # 检查行是否以 " [END]" 结尾
            if not line.endswith(" [END]"):
                print(f"第 {line_number} 行格式错误：应该以 ' [END]' 结尾")
                log_file.write(f"第 {line_number} 行格式错误：应该以 ' [END]' 结尾\n")
                continue

            # 检查行中是否包含 " [END]"，但不在结尾处
            if " [END]" in line[:-6]:
                print(f"第 {line_number} 行格式错误：' [END]' 不应该出现在行中间")
                log_file.write(f"第 {line_number} 行格式错误：' [END]' 不应该出现在行中间\n")
                continue

            # 检查四元组内部的分隔符
            tuples = line.split(" [SEP] ")
            for tuple_index, tuple_str in enumerate(tuples):
                if("non-hate" in tuple_str and len(tuples) > 1):
                    print(f"第 {line_number} 行包含 'non-hate'，但有多个四元组")
                    log_file.write(f"第 {line_number} 行包含 'non-hate'，但有多个四元组\n")
                    continue
                parts = tuple_str.split("|")
                if len(parts) != 4:
                    print(
                        f"第 {line_number} 行的第 {tuple_index + 1} 个四元组格式错误：应该包含四个部分，实际有 {len(parts)} 个部分"
                    )
                    log_file.write(
                        f"第 {line_number} 行的第 {tuple_index + 1} 个四元组格式错误：应该包含四个部分，实际有 {len(parts)} 个部分\n"
                    )
                    continue

    print("文件格式检查完成")
    
if __name__ == "__main__":
    # 测试文件路径
    test_file_path = "./data/result.txt"
    output_log_path = "./data/format.txt"
    check_file_format(test_file_path, output_log_path)
