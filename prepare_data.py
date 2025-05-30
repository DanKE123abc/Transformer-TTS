import numpy as np  # 导入numpy用于数值计算
import pandas as pd  # 导入pandas用于数据处理
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch数据集和数据加载器
import os  # 导入操作系统模块
from utils import get_spectrograms  # 从utils模块导入获取谱图的函数
import hyperparams as hp  # 导入超参数
import librosa  # 导入librosa音频处理库

class PrepareDataset(Dataset):
    """LJSpeech数据集类，用于准备训练数据。
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
            包含梅尔谱和线性谱的字典
        """
        # 构建wav文件路径
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'
        # 获取梅尔谱和线性谱
        mel, mag = get_spectrograms(wav_name)
        
        # 保存梅尔谱和线性谱到文件
        np.save(wav_name[:-4] + '.pt', mel)   # 保存梅尔谱为.pt文件
        np.save(wav_name[:-4] + '.mag', mag)  # 保存线性谱为.mag文件

        # 创建包含梅尔谱和线性谱的样本字典
        sample = {'mel':mel, 'mag': mag}

        return sample
    
if __name__ == '__main__':
    # 创建数据集实例
    dataset = PrepareDataset(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))
    # 创建数据加载器，使用8个工作进程并行处理
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm  # 导入进度条
    pbar = tqdm(dataloader)  # 创建进度条
    # 遍历数据集，提取并保存所有样本的谱图
    for d in pbar:
        pass  # 实际处理在__getitem__中已完成
