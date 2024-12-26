# 基于 Attention 英译中应用

[English](README.md) | [中文](README.zh-CN.md)

## 项目简介
这是一个基于 Flask 构建的英文到中文翻译应用, 旨在探索 `Attention`在 NLP 上的应用。前端使用 HTML 和 Tailwind CSS 设计，提供简洁美观的用户界面，让用户能够输入英文文本并获取相应的中文翻译结果。后端使用 PyTorch 构建的 Seq2Seq 模型进行翻译，并且引入了 `Attention` 机制，[模型](model.pth)已训练并保存，前后端在同一服务器上运行，无需跨域配置。

## 功能
- 简洁美观的用户界面：使用 Tailwind CSS 设计，提供良好的用户体验。
- 输入英文文本：用户可在输入框中输入英文句子。
- 实时翻译：点击“翻译”按钮后，应用会调用后端模型进行翻译，并显示中文结果。
- 错误处理：若翻译请求失败，将显示相应的错误消息。


## 项目结构

```
Project/
├── app.py
├── config.py
├── demo.py
├── model.pth
├── Tatoeba.cmn-en.cmn
├── Tatoeba.cmn-en.en
├── templates/
│   └── index.html
├── vocab.pkl
└── requirements.txt
```
- app.py：Flask 后端应用，处理翻译请求。
- config.py：配置文件，定义模型训练和加载的参数。
- demo.py：包含数据处理、模型定义、训练和翻译功能。
- model.pth：训练好的 PyTorch 模型文件。
- vocab.pkl：词汇表文件，包含源语言和目标语言的词汇表。
- Tatoeba.cmn-en.cmn 和 Tatoeba.cmn-en.en：英文和中文的语料库文件，用于模型训练。
- templates/index.html：前端页面文件。
- requirements.txt：项目依赖列表。

## 运行项目
1.Clone 项目
```
git clone 
```
2.安装依赖

确保您已安装 Python 3.6 及以上版本。然后，在项目根目录下运行以下命令安装所需依赖：
```bash
pip install -r requirements.txt
```
您可以进入训练模型这一步，也可以直接跳过，转到 `运行后端` 这一步。

## 训练模型
模型已经训练完毕，您可以直接使用，见根目录[model.pth](model.pth)。您也可以训练模型，请按照以下步骤进行：

1.准备语料库

确保 [Tatoeba.cmn-en.en](Tatoeba.cmn-en.en) 和 [Tatoeba.cmn-en.cmn](Tatoeba.cmn-en.cmn) 文件位于项目根目录，且每行对应一个英文和中文句子对。您也可以使用自己的语料库。

2.运行训练脚本

在项目根目录下运行以下命令开始训练模型：
```bash
python demo.py --max_pairs 5000
```

这将使用前 5000 个句子对进行训练。训练完成后，model.pth 和 vocab.pkl 文件将被保存到项目根目录。

注意：训练过程可能需要较长时间，具体取决于硬件配置和数据量。

## 运行后端服务
确保 model.pth 和 vocab.pkl 文件位于项目根目录。然后，在项目根目录下运行以下命令启动 Flask 应用：
```
python app.py
```
后端服务将启动在 http://localhost:5000。

## 使用前端页面
1.访问页面

打开浏览器，访问 http://localhost:5000，您将看到翻译页面。

2.进行翻译

- 在输入框中输入英文文本。
- 点击“翻译”按钮。
- 翻译结果将在下方显示。

    **注意**：由于硬件的限制，作者所使用的语料库较小，并不能涵盖日常用语，并且由于时间有限并没有花太多的时间进行模型训练，所以使用仓库中的模型和词汇表进行翻译的效果有时不尽人意，敬请谅解！

## 项目详细说明

1. 配置参数 (config.py)

    config.py 定义了模型训练和加载所需的参数。

    | 参数名称          | 类型     | 默认值                | 描述                                 |
    |-------------------|----------|-----------------------|--------------------------------------|
    | `--batch_size`    | `int`    | `64`                  | 每批处理的样本数量                   |
    | `--enc_emb_dim`   | `int`    | `300`                 | 编码器嵌入向量的维度                 |
    | `--dec_emb_dim`   | `int`    | `300`                 | 解码器嵌入向量的维度                 |
    | `--hid_dim`       | `int`    | `768`                 | 隐藏层的神经元数量                   |
    | `--num_epochs`    | `int`    | `20`                  | 训练的总轮数                         |
    | `--learning_rate` | `float`  | `0.0005`              | 优化器的学习率                       |
    | `--model_path`    | `str`    | `'model.pth'`         | 模型文件的保存路径                   |
    | `--vocab_path`    | `str`    | `'vocab.pkl'`         | 词汇表文件的保存路径                 |
    | `--src_corpus`    | `str`    | `'Tatoeba.cmn-en.en'` | 源语言语料库文件路径                 |
    | `--tgt_corpus`    | `str`    | `'Tatoeba.cmn-en.cmn'` | 目标语言语料库文件路径               |
    | `--max_pairs`     | `int` 或 `None` | `None`           | 最大句子对数量，用于限制训练数据规模 |

2. 模型定义和训练 (demo.py)

    demo.py 包含数据处理、模型定义、训练和翻译功能。

    主要功能：

    - 构建词汇表：根据语料库生成源语言和目标语言的词汇表。
    - 数据预处理：将句子转换为对应的词汇索引，并进行填充。
    - 模型定义：定义 Encoder、Decoder 和 Seq2Seq 模型。
    - 训练模型：使用训练数据对模型进行训练，并保存训练好的模型和词汇表。
    - 翻译功能：加载训练好的模型和词汇表，对输入英文文本进行翻译。

3. 后端应用 (app.py)

    app.py 使用 Flask 构建后端服务，处理翻译请求并返回结果。

    主要功能：

    - 加载模型和词汇表
    - 定义路由
    - 主页
    - 翻译接口：

4. 前端页面 (templates/index.html)

    前端页面使用 Tailwind CSS 设计，提供用户友好的界面。
    主要功能：
    - 输入框：用户可在此输入英文文本。
    - 翻译按钮：带有呼吸感动画效果，点击后发送翻译请求。
    - 翻译结果显示：展示翻译后的中文文本。
    - 错误消息显示：当翻译请求失败时显示相应的错误信息。

## 注意事项
- 模型和词汇表路径
确保 model.pth 和 vocab.pkl 文件位于项目根目录，或者根据实际情况调整 app.py 和 demo.py 中的 --model_path 和 --vocab_path 参数。

- 语料库格式
Tatoeba.cmn-en.en 和 Tatoeba.cmn-en.cmn 文件应为每行一个句子的英文和中文对，句子数量应一致。

- 运行环境
推荐在支持 CUDA 的 GPU 环境下运行以加快训练和翻译速度；否则，模型将在 CPU 上运行，可能会较慢。

## 使用示例
1.启动后端服务

```
python app.py
```

2.访问翻译页面

打开浏览器，访问 http://localhost:5000。

3.进行翻译

- 在输入框中输入英文文本，例如："I will try it"。
- 点击“翻译”按钮。
- 翻译结果将在下方显示，如：我會再試。

    ![img](img/example.png)

## 致谢

- 感谢所有为本项目提供支持和灵感的资源。
- 感谢[Tatoeba](https://opus.nlpl.eu/Tatoeba-v2023-04-12.php)

## 参考

 J. Tiedemann, 2012, <a href="http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf"><i>Parallel Data, Tools and Interfaces in OPUS.</i></a> In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)<br/>


## License

MIT License