## Chinese Fine-Grained Hate Speech Quadruple Extraction

This repository contains the code, data preprocessing pipeline, and experiment configurations for our work on fine-grained hate speech quadruple extraction from Chinese social media text. The task aims to extract structured quadruples in the form of:

```BASH
(Target, Argument, Targeted Group, Hateful)
```

#### Chinese Documentation

For a full explanation of the task, methods, and results in Chinese, please refer to our Chinese README:[中文说明文档](https://github.com/HrqstnMichiru/Hate_Speech_Recognition/README_zh.md)

#### Task Overview

Given a social media comment, the goal is to identify:
* Target: the object being commented on or attacked
* Argument: the main opinion or justification
* Targeted Group: the group being discriminated against (if any)
* Hateful: a binary label indicating whether the comment constitutes hate speech

#### Model & Method

We fine-tune two large language models released by Alibaba Qwen team:
* Qwen2.5-7B-Instruct
* Qwen3-8B-Base

We adopt LoRA (Low-Rank Adaptation) to efficiently fine-tune these models using LLaMA-Factory. Full-parameter fine-tuning and different training step settings are also compared.

#### Results

Our best-performing model, Qwen3-8B-Base with 2500-step LoRA fine-tuning, achieved an average F1 score of 0.3215, outperforming baseline and earlier submissions by a large margin.

#### Contents

* data/: preprocessed training and test data in Alpaca-style format
* scripts/: preprocess scripts
* results/:visualizations and predictions
* full/:full fine-tuning
* lora/:LoRA fine-tuning
* evaluate/:scripts of computing soft-f1 and hard-f1
* README.md: this file

#### Download LoRA Weights

You can download our LoRA fine-tuned adapter weights via the following link:[LoRA Adapter(Baidu Netdisk)](https://pan.baidu.com/s/1Q9ZHNNm9pikmnUE34OJFNQ?pwd=0721)----Password:0721

#### Citation & Contact

If you find this repository useful, feel free to cite or open an issue.
Maintained by [your name / alias] – pull requests are welcome!
