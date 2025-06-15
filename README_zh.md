## 中文细粒度仇恨言论四元组抽取

本仓库包含我们在中文社交媒体文本细粒度仇恨言论识别任务上的代码实现、 数据预处理流程和实验配置。 该任务旨在从原始文本中抽取以下结构化四元组： 

```BASH
(Target, Argument, Targeted Group, Hateful)
```

#### 任务简介

给定一条社交媒体评论， 目标是识别： 
* Target： 被评论或攻击的对象
* Argument： 评论者的主要观点或理由
* Targeted Group： 评论中被歧视的群体（如有）
* Hateful： 该评论是否构成仇恨言论（二分类标签）

#### 模型与方法

我们对阿里通义千问团队发布的两款大语言模型进行了微调： 
* Qwen2.5-7B-Instruct
* Qwen3-8B-Base

采用参数高效微调方法 LoRA（Low-Rank Adaptation）， 并借助 LLaMA-Factory 工具进行训练。 我们同时对比了全参数微调和不同训练步数的设置效果。 

#### 实验结果

最佳模型为使用 Qwen3-8B-Base + LoRA 微调（训练步数 2500 步）， 在官方评估指标上取得了 平均 F1 分数 0.3216， 显著优于无微调基线及早期提交结果， 验证了大语言模型与参数高效微调在中文细粒度信息抽取任务中的潜力。 

#### 项目结构

* data/: 已预处理的 Alpaca 格式训练与测试数据  
* scripts/: 数据预处理脚本
* results/:训练可视化以及测试集预测结果
* full/:全参微调脚本
* lora/:LoRA微调脚本
* evaluate/:计算软匹配和硬匹配F1分数的脚本
* README.md: 本文件

#### LoRA 微调参数下载
你可以通过以下链接下载我们训练好的 LoRA Adapter 权重文件：[LoRA 微调权重文件](https://pan.baidu.com/s/1Q9ZHNNm9pikmnUE34OJFNQ?pwd=0721)

#### 引用与联系方式
如果你觉得这个项目对你有帮助，欢迎引用本仓库或提交 Issues 与我们交流！
项目维护者：Hrqstn Michiru
欢迎 PR