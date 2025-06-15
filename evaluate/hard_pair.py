import json


def calculate_metrics(predictions):
    """
    计算模型预测的精确率、召回率和F1分数

    Args:
        predictions (list): 包含预测结果的字典列表，每个字典应包含 'output' (真实四元组) 和 'response' (预测四元组)。

    Returns:
        tuple: 包含精确率、召回率和F1分数的元组。
    """

    true_positives = 0  # 真阳性: 正确预测的四元组数量
    predicted_positives = 0  # 预测阳性: 模型预测的四元组总数
    actual_positives = 0  # 实际阳性: 真实标签中的四元组总数

    for item in predictions:
        # 分割目标（真实）四元组和预测四元组
        # 移除 '[END]' 标记并去除首尾空格，同时过滤掉空字符串
        target_quadruples = [
            t.replace("[END]", "").strip()
            for t in item["output"].split("[SEP]")
            if t.strip()
        ]
        prediction_quadruples = [
            p.replace("[END]", "").strip()
            for p in item["response"].split("[SEP]")
            if p.strip()
        ]

        actual_positives += len(target_quadruples)  # 更新实际阳性总数
        predicted_positives += len(prediction_quadruples)  # 更新预测阳性总数

        # 比较每个预测四元组与目标四元组
        # 为了避免重复匹配，每次匹配成功后从 target_quadruples 中移除已匹配项
        temp_target_quadruples = list(target_quadruples)  # 创建一个临时列表进行匹配操作
        for pred in prediction_quadruples:
            for t in temp_target_quadruples:
                if pred == t:  # 如果预测四元组与目标四元组完全匹配
                    true_positives += 1
                    temp_target_quadruples.remove(t)  # 确保每个目标四元组只被匹配一次
                    break  # 找到匹配后，跳出内部循环，处理下一个预测四元组

    # 计算精确率 (Precision)
    # 精确率 = 真阳性 / (真阳性 + 假阳性) = 真阳性 / 预测阳性总数
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0

    # 计算召回率 (Recall)
    # 召回率 = 真阳性 / (真阳性 + 假阴性) = 真阳性 / 实际阳性总数
    recall = true_positives / actual_positives if actual_positives > 0 else 0

    # 计算F1分数
    # F1分数是精确率和召回率的调和平均值
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1_score


def load_data(file_path):
    """
    从JSON文件中加载数据

    Args:
        file_path (str): JSON文件的路径。

    Returns:
        list: 从JSON文件加载的数据。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main(input_json_path):
    """
    主函数，用于加载数据并计算评估指标

    Args:
        input_json_path (str): 输入JSON文件的路径。
    """
    print(f"正在加载数据：{input_json_path}")
    data = load_data(input_json_path)
    print(f"数据加载完成，共 {len(data)} 条记录。")

    print("正在计算评估指标...")
    precision, recall, f1_score = calculate_metrics(data)

    print("\n--- 评估结果 ---")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数 (F1 Score): {f1_score:.4f}")
    print("----------------")


if __name__ == "__main__":
    input_json_path = "../data/predictions.json"
    main(input_json_path)
