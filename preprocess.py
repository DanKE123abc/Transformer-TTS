import hyperparams as hp  # 导入超参数配置
import pandas as pd  # 导入pandas用于数据处理
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch数据集和数据加载器
import os  # 导入操作系统模块
import librosa  # 导入librosa音频处理库
import numpy as np  # 导入numpy用于数值计算
from text import text_to_sequence  # 导入文本到序列的转换函数
import collections  # 导入集合模块
from scipy import signal  # 导入信号处理模块
import torch as t  # 导入PyTorch并重命名为t
import math  # 导入数学模块


class LJDatasets(Dataset):
    """LJSpeech数据集类，用于Transformer模型训练。
    LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        初始化数据集
        Args:
            csv_file (string): 包含标注的csv文件路径。
            root_dir (string): 包含所有wav文件的目录。

        """
        # 读取csv文件，使用'|'作为分隔符，不使用标题行
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir  # 保存根目录路径

    def load_wav(self, filename):
        """加载音频文件
        Args:
            filename (string): 音频文件路径
        Returns:
            音频数据和采样率
        """
        return librosa.load(filename, sr=hp.sample_rate)  # 使用指定采样率加载音频

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.landmarks_frame)  # 返回csv文件中的行数

    def __getitem__(self, idx):
        """获取指定索引的样本
        Args:
            idx (int): 样本索引
        Returns:
            包含文本、梅尔谱、文本长度、梅尔输入、位置编码等的字典
        """
        # 构建wav文件路径
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'
        # 获取对应的文本
        text = self.landmarks_frame.ix[idx, 1]

        # 将文本转换为序列（数字表示）
        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        # 加载预处理好的梅尔谱
        mel = np.load(wav_name[:-4] + '.pt.npy')
        # 创建梅尔谱输入（将梅尔谱向右移动一帧，并在开头添加零帧）
        mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        # 计算文本长度
        text_length = len(text)
        # 创建文本位置编码（从1开始）
        pos_text = np.arange(1, text_length + 1)
        # 创建梅尔谱位置编码（从1开始）
        pos_mel = np.arange(1, mel.shape[0] + 1)

        # 创建包含所有必要数据的样本字典
        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text}

        return sample
    
class PostDatasets(Dataset):
    """LJSpeech数据集类，用于PostNet网络训练。
    LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        初始化数据集
        Args:
            csv_file (string): 包含标注的csv文件路径。
            root_dir (string): 包含所有wav文件的目录。

        """
        # 读取csv文件，使用'|'作为分隔符，不使用标题行
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir  # 保存根目录路径

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.landmarks_frame)  # 返回csv文件中的行数

    def __getitem__(self, idx):
        """获取指定索引的样本
        Args:
            idx (int): 样本索引
        Returns:
            包含梅尔谱和线性谱的字典
        """
        # 构建wav文件路径
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'
        # 加载预处理好的梅尔谱
        mel = np.load(wav_name[:-4] + '.pt.npy')
        # 加载预处理好的线性谱
        mag = np.load(wav_name[:-4] + '.mag.npy')
        # 创建包含梅尔谱和线性谱的样本字典
        sample = {'mel':mel, 'mag':mag}

        return sample
    
def collate_fn_transformer(batch):
    """Transformer模型的批次整合函数
    将批次中的样本整合为模型输入格式，并进行填充和排序
    Args:
        batch: 批次数据，包含多个样本
    Returns:
        整合后的批次数据，包括文本、梅尔谱、梅尔输入、文本位置编码、梅尔位置编码和文本长度
    """
    # 将每个数据字段放入具有批次大小外层维度的张量中
    if isinstance(batch[0], collections.Mapping):

        # 从批次中提取各个字段
        text = [d['text'] for d in batch]  # 文本序列
        mel = [d['mel'] for d in batch]  # 梅尔谱
        mel_input = [d['mel_input'] for d in batch]  # 梅尔谱输入（向右移动一帧）
        text_length = [d['text_length'] for d in batch]  # 文本长度
        pos_mel = [d['pos_mel'] for d in batch]  # 梅尔谱位置编码
        pos_text= [d['pos_text'] for d in batch]  # 文本位置编码
        
        # 按文本长度降序排序所有数据
        # 这样做是为了提高计算效率，使得较长的序列放在批次的前面
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)  # 文本长度降序排序
        
        # 使用批次中最大长度对序列进行填充
        text = _prepare_data(text).astype(np.int32)  # 填充文本序列
        mel = _pad_mel(mel)  # 填充梅尔谱
        mel_input = _pad_mel(mel_input)  # 填充梅尔谱输入
        pos_mel = _prepare_data(pos_mel).astype(np.int32)  # 填充梅尔谱位置编码
        pos_text = _prepare_data(pos_text).astype(np.int32)  # 填充文本位置编码

        # 将numpy数组转换为PyTorch张量并返回
        return t.LongTensor(text), t.FloatTensor(mel), t.FloatTensor(mel_input), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)

    # 如果批次数据类型不正确，抛出类型错误
    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))
    
def collate_fn_postnet(batch):
    """PostNet模型的批次整合函数
    将批次中的样本整合为PostNet模型输入格式，并进行填充
    Args:
        batch: 批次数据，包含多个样本
    Returns:
        整合后的批次数据，包括梅尔谱和线性谱
    """
    # 将每个数据字段放入具有批次大小外层维度的张量中
    if isinstance(batch[0], collections.Mapping):

        # 从批次中提取各个字段
        mel = [d['mel'] for d in batch]  # 梅尔谱
        mag = [d['mag'] for d in batch]  # 线性谱（幅度谱）
        
        # 使用批次中最大长度对序列进行填充
        mel = _pad_mel(mel)  # 填充梅尔谱
        mag = _pad_mel(mag)  # 填充线性谱

        # 将numpy数组转换为PyTorch张量并返回
        return t.FloatTensor(mel), t.FloatTensor(mag)

    # 如果批次数据类型不正确，抛出类型错误
    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def _pad_data(x, length):
    """填充一维数据到指定长度
    Args:
        x: 需要填充的数组
        length: 目标长度
    Returns:
        填充后的数组
    """
    _pad = 0  # 填充值
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)  # 在末尾填充0

def _prepare_data(inputs):
    """准备批次数据，将列表中的所有数组填充到相同长度并堆叠
    Args:
        inputs: 包含多个数组的列表
    Returns:
        填充并堆叠后的数组
    """
    max_len = max((len(x) for x in inputs))  # 计算最大长度
    return np.stack([_pad_data(x, max_len) for x in inputs])  # 填充每个数组并堆叠

def _pad_per_step(inputs):
    """按步长填充输入数据
    Args:
        inputs: 输入数组
    Returns:
        填充后的数组
    """
    timesteps = inputs.shape[-1]  # 获取时间步数
    # 填充到outputs_per_step的整数倍
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    """计算模型的参数数量
    Args:
        model: PyTorch模型
    Returns:
        模型参数总数
    """
    params = 0
    for p in model.parameters():  # 遍历模型的所有参数
        tmp = 1
        for x in p.size():  # 计算每个参数的元素数量
            tmp *= x
        params += tmp  # 累加参数数量
    return params

def get_dataset():
    """获取Transformer模型的训练数据集
    Returns:
        LJDatasets实例
    """
    return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def get_post_dataset():
    """获取PostNet模型的训练数据集
    Returns:
        PostDatasets实例
    """
    return PostDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def _pad_mel(inputs):
    """填充梅尔谱数据到批次中的最大长度
    Args:
        inputs: 包含多个梅尔谱数组的列表
    Returns:
        填充并堆叠后的梅尔谱数组
    """
    _pad = 0  # 填充值
    def _pad_one(x, max_len):
        """填充单个梅尔谱"""
        mel_len = x.shape[0]  # 获取梅尔谱长度
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)  # 在时间维度填充
    max_len = max((x.shape[0] for x in inputs))  # 计算最大长度
    return np.stack([_pad_one(x, max_len) for x in inputs])  # 填充每个梅尔谱并堆叠

