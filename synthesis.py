import torch as t  # 导入PyTorch并重命名为t
from utils import spectrogram2wav  # 导入谱图转波形函数
from scipy.io.wavfile import write  # 导入波形文件写入函数
import hyperparams as hp  # 导入超参数
from text import text_to_sequence  # 导入文本到序列的转换函数
import numpy as np  # 导入numpy用于数值计算
from network import ModelPostNet, Model  # 导入模型类
from collections import OrderedDict  # 导入有序字典
from tqdm import tqdm  # 导入进度条
import argparse  # 导入命令行参数解析器

def load_checkpoint(step, model_name="transformer"):
    """加载模型检查点
    Args:
        step: 训练步数
        model_name: 模型名称，默认为"transformer"
    Returns:
        模型状态字典
    """
    # 加载检查点文件
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()  # 创建新的有序字典
    # 处理状态字典中的键名（移除"module."前缀）
    for k, value in state_dict['model'].items():
        key = k[7:]  # 移除"module."前缀
        new_state_dict[key] = value

    return new_state_dict  # 返回处理后的状态字典

def synthesis(text, args):
    """合成语音
    Args:
        text: 输入文本
        args: 命令行参数
    """
    # 初始化模型
    m = Model()  # 创建Transformer模型实例
    m_post = ModelPostNet()  # 创建PostNet模型实例

    # 加载模型检查点
    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))  # 加载Transformer模型权重
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))  # 加载PostNet模型权重

    # 处理输入文本
    text = np.asarray(text_to_sequence(text, [hp.cleaners]))  # 将文本转换为数字序列
    text = t.LongTensor(text).unsqueeze(0)  # 转换为PyTorch张量并添加批次维度
    text = text.cuda()  # 将文本张量移至GPU
    # 初始化梅尔谱输入（起始帧）
    mel_input = t.zeros([1,1, 80]).cuda()  # 创建全零张量作为初始梅尔谱输入
    # 创建文本位置编码
    pos_text = t.arange(1, text.size(1)+1).unsqueeze(0)  # 生成文本位置编码
    pos_text = pos_text.cuda()  # 将位置编码移至GPU

    # 将模型移至GPU并设置为评估模式
    m=m.cuda()  # 将Transformer模型移至GPU
    m_post = m_post.cuda()  # 将PostNet模型移至GPU
    m.train(False)  # 设置Transformer模型为评估模式
    m_post.train(False)  # 设置PostNet模型为评估模式
    
    # 自回归生成梅尔谱
    pbar = tqdm(range(args.max_len))  # 创建进度条
    with t.no_grad():  # 禁用梯度计算
        for i in pbar:  # 逐帧生成梅尔谱
            # 创建梅尔谱位置编码
            pos_mel = t.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()  # 生成梅尔谱位置编码
            # 前向传播
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)  # 模型前向传播
            # 将预测的梅尔谱帧添加到输入中
            mel_input = t.cat([mel_input, mel_pred[:,-1:,:]], dim=1)  # 将预测的最后一帧添加到输入中

        # 使用PostNet生成线性谱
        mag_pred = m_post.forward(postnet_pred)  # 将梅尔谱转换为线性谱
        
    # 将线性谱转换为波形
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())  # 将线性谱转换为波形
    # 保存波形文件
    write(hp.sample_path + "/test.wav", hp.sr, wav)  # 将波形保存为WAV文件
    
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--restore_step1', type=int, help='Transformer模型的检查点步数', default=172000)  # Transformer模型检查点步数
    parser.add_argument('--restore_step2', type=int, help='PostNet模型的检查点步数', default=100000)  # PostNet模型检查点步数
    parser.add_argument('--max_len', type=int, help='生成的最大帧数', default=400)  # 生成的最大帧数

    args = parser.parse_args()  # 解析命令行参数
    synthesis("Transformer model is so fast!",args)  # 合成示例文本
