<div align="center">

ZH | [EN](./README-en.md)

<h1>DISC-LawLLM</h1>
  
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/ShengbinYue/DISC-LawLLM)
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](./LICENSE)

[Demo]() | [技术报告](https://arxiv.org/abs/2309.11325)

</div>

DISC-LawLLM 是一个旨在为用户提供专业、智能、全面的**法律服务**的法律领域大模型，由[复旦大学数据智能与社会计算实验室 (Fudan-DISC)](http://fudan-disc.com) 开发并开源。

我们将在该项目中开源如下资源：
* [DISC-Law-SFT 数据集](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)（不包括法律问答部分）
* [DISC-LawLLM 模型权重](https://huggingface.co/ShengbinYue/DISC-LawLLM)
* DISC-Law-Eval Benchmark

您可以通过访问这个[链接]()来在线体验我们的 DISC-LawLLM。

## 新闻

**[2023/09/25]** DISC-LawLLM v1.0 已正式发布，开源 [DISC-LawLLM-13B 模型](https://huggingface.co/ShengbinYue/DISC-LawLLM) 和 [DISC-Law-SFT 数据集](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)。

## 目录

- [概述](#概述)
- [推理和部署](#推理和部署)
- [模型微调](#模型微调)
- [DISC-Law-Eval Benchmark](#disc-law-eval-benchmark)
- [致谢](#致谢)
- [声明](#声明)
- [引用](#引用)
- [协议](#协议)

## 概述

![Image](./images/model_zh.png)

<p></p>

DISC-LawLLM 是一个具有法律推理和知识检索能力的智能法律系统，它面向不同群体，能在不同应用场景下提供帮助，主要有以下几个特点：

* **法律文本处理能力：** 针对法律语言的理解与生成能力，包括信息抽取、文本摘要等，我们基于现有的NLP司法任务公开数据和真实世界的法律相关文本进行了微调数据的构建。
* **法律推理思维能力：** 针对智慧司法领域任务的需求，我们使用法律三段论这一法理推理理论设计了指令数据，有效地提高了模型的法理推理能力。
* **司法领域知识遵循能力：** 我们为智能法律处理系统配备了检索增强的模块，增强了系统对于背景知识的的检索和遵循能力。

除此之外，我们的研究过程还包括了如下贡献：

* **高质量的训练数据集和普遍有效的训练范式**
* **完备的法律模型测评框架和测评数据集**

### 模型效果演示

* 法律问答

![img1](./images/example_consult.gif)

* 带检索模块的法律考试选择题解答

![img2](./images/example_exam_ref.gif)

* 带检索模块的法律咨询服务

![img3](./images/example_consult_ref.gif)

* 文书撰写（离婚财产分割协议）

![img4](./images/example_document.gif)

* 法律专业人员工具（司法事件检测）

![img5](./images/example_tool.gif)

### DISC-Law-SFT 数据集

不同场景下的法律智能应用通常需要结合法律文本理解和生成的多种基本能力。为此，我们构建了一个高质量的监督微调数据集 DISC-Law-SFT，包括法律信息提取、判决预测、文档摘要和法律问题解答，确保覆盖不同司法应用场景。DISC-Law-SFT包括两个子集，即DISC-Law-SFT-Pair 和 DISC-Law-SFT-Triplet。前者旨在为 LLM 引入法律推理能力，后者则有助于提高模型利用外部知识的能力，具体的构建细节请参照我们的[技术报告](https://arxiv.org/abs/2309.11325)。数据集的分布如下所示：

<img src="" alt="" width=""/>

<table>
  <tr>
    <th>数据集</th>
    <th>对应任务/来源</th>
    <th>样本量</th>
    <th>对应情境</th>
  </tr>
  <tr>
    <td rowspan="10">DISC-LawLLM-SFT-Pair</td>
    <td>司法要素提取</td>
    <td>32K</td>
    <td rowspan="7">法律专业人员助手</td>
  </tr>
  <tr>
    <td>司法事件检测</td>
    <td>27K</td>
  </tr>
  <tr>
    <td>案件分类</td>
    <td>20K</td>
  </tr>
  <tr>
    <td>判决预测</td>
    <td>11K</td>
  </tr>
  <tr>
    <td>类案匹配</td>
    <td>8K</td>
  </tr>
  <tr>
    <td>司法摘要</td>
    <td>9K</td>
  </tr>
  <tr>
    <td>舆情摘要</td>
    <td>6K</td>
  </tr>
  <tr>
    <td>法律问答</td>
    <td>93K</td>
    <td>法律咨询服务</td>
  </tr>
  <tr>
    <td>司法阅读理解</td>
    <td>38K</td>
    <td rowspan="2">法律考试助手</td>
  </tr>
  <tr>
    <td>法律考试</td>
    <td>12K</td>
  </tr>
  <tr>
    <td rowspan="2">DISC-LawLLM-SFT-Triple</td>
    <td>判决预测</td>
    <td>16K</td>
    <td>法律专业人员助手</td>
  </tr>
  <tr>
    <td>法律问答</td>
    <td>23K</td>
    <td>法律咨询服务</td>
  </tr>
  <tr>
    <td rowspan="2">General</td>
    <td>Alpaca-GPT4</td>
    <td>48K</td>
    <td rowspan="2">通用场景</td>
  </tr>
  <tr>
    <td>Firefly</td>
    <td>60K</td>
  </tr>
  <tr>
    <td>总计</td>
    <td colspan="3">403K</td>
  </tr>
</table>

我们总共发布了近30万条训练数据，其中包括 DISC-Law-SFT-Pair 和DISC-Law-SFT-Triplet。您可以访问这个[链接](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)下载数据集。

### 检索增强模块

我们在 DISC-LawLLM 的基础上增加了一个基于开源检索框架 [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) 的检索模块。我们的知识库目前包括法条库和法考题库。

* 法条库包含 800 多部国家地方法律、条例和规定，其中包括《宪法》、《刑法》、《行政诉讼法》、《保险法》、《劳动法》、《著作权法》、《民法典》、《专利法》、《专属经济区和大陆架法》、《中国人民解放军选举全国人民代表大会和县级以上地方各级人民代表大会代表的办法》、《反分裂国家法》、《出境入境边防检查条例》、《国务院关于鼓励台湾同胞投资的规定》、《境内外国人宗教活动管理规定》等。
* 法考题库包含2.4万道法律相关的考试题目。

在未来，我们会增加更加丰富的知识库。我们还将进一步深入探索检索增强的 DISC-LawLLM，包括但不限于检索器与 LLM 的联合训练机制，各位有兴趣可以与我们一起交流。

## 推理和部署

开源版本的 DISC-LawLLM 是基于 [Baichuan-13B-Base](https://github.com/baichuan-inc/Baichuan-13B) 进行微调训练得到的。您可以直接从 [Hugging Face](https://huggingface.co/ShengbinYue/DISC-LawLLM) 上下载我们的模型权重，或者根据下面的代码样例自动获取。推理前请安装依赖：

```
pip install -r requirements.txt
```

### Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_path = "ShengbinYue/DISC-LawLLM"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.generation_config = GenerationConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, use_fast=False, trust_remote_code=True,
)

messages = [
    {"role": "user", "content": "生产销售假冒伪劣商品罪如何判刑？"},
]
response = model.chat(tokenizer, messages)
```

### 命令行工具

```
python cli_demo.py
```

### 网页 Demo

依靠 streamlit 工具运行以下命令，会在本地启动一个 web 服务，把控制台给出的地址输入浏览器即可访问：

```
streamlit run web_demo.py --server.port 8888
```

### 部署

目前版本的 DISC-LawLLM 是以 Baichuan-13B 作为基座的，您可以参照 [Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B) 的介绍来进行 int8 或 int4 量化推理部署以及CPU部署。

## 模型微调

开发者可以对 DISC-LawLLM 进行微调使用。在此可以参照与 DISC-LawLLM 兼容的微调工具 [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 或是我们的 [DISC-MedLLM](https://github.com/FudanDISC/DISC-MedLLM) 医疗大模型。我们以 [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 为例给出**全量**和  **LoRA** 两种微调实例。

首先，下载 [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 并按其要求[安装依赖](https://github.com/hiyouga/LLaMA-Efficient-Tuning#getting-started)。注意训练数据按照项目中的要求进行处理。下面我们给出两种微调场景下的脚本样例。

### 全量微调

我们在 8 * Nvidia A800 80 GB + deepspeed 的环境下进行了全量微调测试。训练启动脚本示例如下：

```
deepspeed --num_gpus=8 src/train_bash.py \
    --stage sft \
    --model_name_or_path S heng bin \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --template baichuan \
    --finetuning_type full \
    --output_dir path_to_your_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \ 
    --per_device_eval_batch_size 4 \ 
    --gradient_accumulation_steps 8 \ 
    --preprocessing_num_workers 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --dev_ratio 0.01 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16 \
    --deepspeed deepspeed.json
```

`deep_speed.json` 配置示例如下：

```json
{
    "train_micro_batch_size_per_gpu": "auto",
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "initial_scale_power": 16, 
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },  
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": false,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients" : true
    }
}
```

### LoRA 微调

我们在 4 * Nvidia A800 80G 显卡上进行了 LoRA 微调测试。训练启动脚本示例如下：

```
torchrun --nproc_per_node 4 src/train_bash.py \
    --stage sft \
    --model_name_or_path ShengbinYue/DISC-LawLLM \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --template baichuan \
    --finetuning_type lora \
    --lora_rank 8 \ 
    --lora_target W_pack \
    --output_dir path_to_your_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \ 
    --per_device_eval_batch_size 4 \ 
    --gradient_accumulation_steps 8 \ 
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --dev_ratio 0.01 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16
```

## DISC-Law-Eval-Benchmark

受司法考试构成的启发，我们开发了一个公平的评估框架 —— DISC-Law-Eval Benchmark，从客观和主观两个角度对法律大语言模型的性能进行评估，以考察模型在中国法律领域的性能。您可以可以点击此[链接](https://github.com/FudanDISC/DISC-LawLLM/evl)使用我们的 DISC-Law-Eval-Benchmark。

### 客观评测

为了客观、定量地评估智能法律系统的法律知识和推理能力，客观的评价数据集由一系列中国法律标准化考试和知识竞赛的单项和多项选择题组成，并根据内容复杂性和演绎难度，将问题分为困难、中等和简单三个层次。它可以提供一个更具挑战性和可靠的方法来衡量模型是否可以利用其知识来推理正确的答案。我们通过计算精度来表明性能。具体构成如下：

<table>
  <tr>
    <th>科目</th>
    <th>难度等级</th>
    <th>单选题数量</th>
    <th>多选题数量</th>
    <th>总数</th>
  </tr>
  <tr>
    <td>NJE：国家统一法律职业资格考试</td>
    <td rowspan="3">困难</td>
    <td>537</td>
    <td>463</td>
    <td>1000</td>
  </tr>
  <tr>
    <td>PAE：专利代理人考试</td>
    <td>118</td>
    <td>276</td>
    <td>394</td>
  </tr>
  <tr>
    <td>CPA：注册会计师资格考试</td>
    <td>197</td>
    <td>120</td>
    <td>317</td>
  </tr>
  <tr>
    <td>UNGEE：法学专硕全国统考试题</td>
    <td>中等</td>
    <td>320</td>
    <td>87</td>
    <td>407</td>
  </tr>
  <tr>
    <td>LBK：法律基础知识题库</td>
    <td rowspan="2">简单</td>
    <td>275</td>
    <td>-</td>
    <td>275</td>
  </tr>
  <tr>
    <td>PFE：事业编、公务员考试法律试题</td>
    <td>170</td>
    <td>-</td>
    <td>170</td>
  </tr>
</table>

### 主观评测

在主观评测部分，我们采用问答题形式进行评估，模拟主观考试问题的过程。我们从法律咨询、在线论坛、与司法相关的出版物和法律文件中手工构建了一个高质量的测试集。我们用 GPT-3.5 Turbo 作为裁判模型来评估模型的输出，并基于标准答案用准确性、完整性和清晰度这三个标准提供 1-5 的评分。

主观题数据集从来源于法律咨询、网上发帖、司法相关出版物和法律文书中手动构建的一个高质量的测试集，其中包括 300 个示例，涵盖了法律知识问答、法律咨询和判决预测等场景。

### 评测结果

客观题评测结果(%)如下（采用 few-shot 的方式）。

|        模型        |  NJE 单选   |  NJE 多选   |  PAE 单选   |  PAE 多选   |  CPA 单选   |  CPA 多选   | UNGEE 单选  | UNGEE 多选  |  PFE 单选   |  LBK 单选   |   平均   |
|:----------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|     ChatGLM      |   31.66   |   1.08    |   27.97   |   2.90    |   37.06   |   13.33   |   39.69   |   20.69   |   37.65   |   42.91   |   24.66   |
|  Baichuan-Chat   |   31.47   |   10.15   |   29.66   |   8.70    |   35.53   |   19.17   |   50.00   |   27.59   |   53.12   |   53.45   |   30.78   |
| Chinese-Alpaca-2 |   25.70   |   10.15   |   30.51   |   11.59   |   32.99   |   19.17   |   40.94   |   21.84   |   44.12   |   43.27   |   26.73   |
|  GPT-3.5-turbo   |   36.50   |   10.58   |   37.29   |   17.03   | **42.13** | **21.67** | **51.25** | **28.74** |   53.53   |   54.18   |   34.10   |
|     LexiLaw      |   20.11   |   7.56    |   23.73   |   10.14   |   24.87   |   19.17   |   31.56   |   16.09   |   31.76   |   40.36   |   21.50   |
|      LawGPT      |   22.91   |   6.26    |   31.36   |   7.61    |   25.38   |   16.67   |   30.31   |   13.79   |   34.71   |   29.09   |   20.60   |
|   Lawyer LLaMa   |   35.75   |   5.62    |   32.20   |   6.52    |   29.95   |   13.33   |   32.50   |   14.94   |   39.41   |   39.64   |   25.05   |
|     ChatLaw      |   27.56   |   7.99    |   31.36   |   9.42    |   35.53   |   11.67   |   35.62   |   17.24   |   42.35   |   41.09   |   25.20   |
|   DISC-LawLLM    | **42.09** | **19.87** | **40.68** | **18.48** |   39.59   |   19.17   |   50.94   |   25.29   | **57.06** | **54.91** | **37.10** |

主观题评测结果如下，每个分数为 1-5。

|        模型        | 准确性  | 完整性  | 清晰性  |  平均  |
|:----------------:|:----:|:----:|:----:|:----:|
|     ChatGLM      | 2.64 | 2.75 | 3.23 | 2.87 |
|  Baichuan-Chat   | 3.22 | **3.34** | 3.18 | 3.25 |
| Chinese-Alpaca-2 | 3.13 | 3.23 | 3.17 | 3.17 |
|     LexiLaw      | 3.06 | 2.62 | 3.00 | 2.90 |
|      LawGPT      | 3.02 | 2.58 | 2.96 | 2.86 |
|   Lawyer LLaMa   | 3.13 | 2.83 | 3.35 | 3.10 |
|     ChatLaw      | 3.31 | 2.90 | 3.35 | 3.19 |
|   DISC-LawLLM    | **3.46** | 3.12 | **3.59** | **3.39** |

## 致谢

本项目基于如下开源项目展开，在此对相关项目和开发人员表示诚挚的感谢：

- [**Baichuan-13B**](https://github.com/baichuan-inc/Baichuan-13B)
- [**Langchain-Chatchat**](https://github.com/chatchat-space/Langchain-Chatchat)
- [**LLaMA Efficient Tuning**](https://github.com/hiyouga/LLaMA-Efficient-Tuning)
- [**FireFly**](https://github.com/yangjianxin1/Firefly)

同样感谢其他限于篇幅未能列举的为本项目提供了重要帮助的工作。

## 声明

DISC-LawLLM 有着目前大语言模型尚无法克服的问题和缺陷，尽管它能够在许多任务和情境上提供法律服务，但模型应当仅供用户参考使用，并不能替代专业律师和法律专家，我们希望 DISC-LawLLM 的用户以批判性的眼光去评估模型。我们不对因使用 DISC-LawLLM 所引发的任何问题、风险或不良后果承担责任。

## 引用

如果我们的项目对您的研究和工作有帮助，请如下引用我们的项目：

```
@misc{yue2023disclawllm,
    title={DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services}, 
    author={Shengbin Yue and Wei Chen and Siyuan Wang and Bingxuan Li and Chenchen Shen and Shujun Liu and Yuxuan Zhou and Yao Xiao and Song Yun and Wei Lin and Xuanjing Huang and Zhongyu Wei},
    year={2023},
    eprint={2309.11325},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## 协议

DISC-LawLLM 可在 Apache 许可证下使用。请查看 [LICENSE 文件](./LICENSE) 获取更多信息。


## Star History

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FudanDISC/DISC-LawLLM&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FudanDISC/DISC-LawLLM&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=FudanDISC/DISC-LawLLM&type=Date" />
</picture>
