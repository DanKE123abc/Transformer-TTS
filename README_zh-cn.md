# Transformer-TTS
* 这是[Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)论文的PyTorch实现
* 该模型的训练速度比著名的序列到序列模型（如Tacotron）快约3到4倍，且合成语音的质量几乎相同。通过实验确认，每步训练大约需要0.5秒。
* 我没有使用WaveNet声码器，而是使用Tacotron的CBHG模型学习后处理网络，并使用Griffin-Lim算法将频谱图转换为原始波形。

<img src="png/model.png">

## 环境要求
  * 安装Python 3
  * 安装PyTorch == 0.4.0
  * 安装依赖项：
    ```
   	pip install -r requirements.txt
   	```

## 数据
* 我使用了LJSpeech数据集，该数据集由文本脚本和WAV文件对组成。完整数据集（13,100对）可以在[这里](https://keithito.com/LJ-Speech-Dataset/)下载。我参考了https://github.com/keithito/tacotron 和 https://github.com/Kyubyong/dc_tts 的预处理代码。

## 预训练模型
* 你可以在[这里](https://drive.google.com/drive/folders/1r1tdgsdtipLossqD9ZfDmxSZb8nMO8Nf)下载预训练模型（AR模型160K步 / 后处理网络100K步）
* 将预训练模型放在checkpoint/目录下。

## 注意力图
* 对角线对齐在大约15k步后出现。下面的注意力图是在160k步时的结果。这些图表示所有层的多头注意力。在这个实验中，三个注意力层使用了h=4。因此，对于编码器、解码器和编码器-解码器，分别绘制了12个注意力图。除了解码器外，只有少数多头显示对角线对齐。

### 编码器自注意力
<img src="png/attention_encoder.gif" height="200">

### 解码器自注意力
<img src="png/attention_decoder.gif" height="200">

### 编码器-解码器注意力
<img src="png/attention.gif" height="200">

## 学习曲线和Alpha值
* 我使用了与[Tacotron](https://github.com/Kyubyong/tacotron)相同的Noam风格预热和衰减

<img src="png/training_loss.png">

* 缩放位置编码的alpha值与论文不同。在论文中，编码器的alpha值增加到4，而在本实验中，它在开始时略有增加，然后持续下降。解码器的alpha值自开始以来一直稳定下降。

<img src="png/alphas.png">

## 实验笔记
1. **学习率是训练的重要参数。** 使用0.001的初始学习率和指数衰减不起作用。
2. **梯度裁剪也是训练的重要参数。** 我使用范数值1裁剪梯度。
3. 使用停止标记损失时，模型无法训练。
4. **在注意力机制中连接输入和上下文向量非常重要。**

## 生成的样本
* 你可以在下面查看一些生成的样本。所有样本都是在160k步时生成的，所以我认为模型还没有完全收敛。这个模型在长句子上的性能似乎较低。

    * [样本1](https://soundcloud.com/ksrbpbmcxrzu/160k-0)
    * [样本2](https://soundcloud.com/ksrbpbmcxrzu/160k_sample_1)
    * [样本3](https://soundcloud.com/ksrbpbmcxrzu/160k_sample_2)

* 第一个图是预测的梅尔频谱图，第二个是真实值。
<img src="png/mel_pred.png" width="800">
<img src="png/mel_original.png" width="800">

## 文件说明
  * `hyperparams.py` 包含所有需要的超参数。
  * `prepare_data.py` 将WAV文件预处理为梅尔频谱图、线性频谱图并保存它们以加快训练时间。文本的预处理代码在text/目录中。
  * `preprocess.py` 包含加载数据时的所有预处理代码。
  * `module.py` 包含所有方法，包括注意力、前置网络、后处理网络等。
  * `network.py` 包含网络，包括编码器、解码器和后处理网络。
  * `train_transformer.py` 用于训练自回归注意力网络。（文本 --> 梅尔频谱图）
  * `train_postnet.py` 用于训练后处理网络。（梅尔频谱图 --> 线性频谱图）
  * `synthesis.py` 用于生成TTS样本。

## 训练网络
  * 步骤1. 在任何你想要的目录下载并解压LJSpeech数据。
  * 步骤2. 调整`hyperparams.py`中的超参数，特别是'data_path'，这是你解压文件的目录，以及其他必要的参数。
  * 步骤3. 运行`prepare_data.py`。
  * 步骤4. 运行`train_transformer.py`。
  * 步骤5. 运行`train_postnet.py`。

## 生成TTS WAV文件
  * 步骤1. 运行`synthesis.py`。确保恢复步骤正确。

## 参考
  * Keith ito: https://github.com/keithito/tacotron
  * Kyubyong Park: https://github.com/Kyubyong/dc_tts
  * jadore801120: https://github.com/jadore801120/attention-is-all-you-need-pytorch/

## 评论
  * 欢迎对代码提出任何评论。

