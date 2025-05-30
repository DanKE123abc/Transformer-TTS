import numpy as np  # 导入数值计算库
import librosa  # 导入音频处理库
import os, copy  # 导入操作系统和复制模块
from scipy import signal  # 导入信号处理模块
import hyperparams as hp  # 导入超参数
import torch as t  # 导入PyTorch并重命名为t

def get_spectrograms(fpath):
    '''解析波形文件并返回归一化的梅尔谱图和线性谱图。
    Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.（声音文件的完整路径）
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.（形状为(T, n_mels)的梅尔谱图）
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.（形状为(T, 1+n_fft/2)的线性谱图）
    '''
    # Loading sound file（加载声音文件）
    y, sr = librosa.load(fpath, sr=hp.sr)  # 加载音频，使用指定的采样率

    # Trimming（修剪静音部分）
    y, _ = librosa.effects.trim(y)  # 去除音频开头和结尾的静音部分

    # Preemphasis（预加重处理）
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])  # 应用预加重滤波器增强高频

    # stft（短时傅里叶变换）
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,  # FFT窗口大小
                          hop_length=hp.hop_length,  # 帧移
                          win_length=hp.win_length)  # 窗口长度

    # magnitude spectrogram（幅度谱）
    mag = np.abs(linear)  # (1+n_fft//2, T)，计算复数的绝对值得到幅度

    # mel spectrogram（梅尔谱）
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)，创建梅尔滤波器组
    mel = np.dot(mel_basis, mag)  # (n_mels, t)，应用梅尔滤波器组

    # to decibel（转换为分贝值）
    mel = 20 * np.log10(np.maximum(1e-5, mel))  # 转换为分贝值，避免log(0)
    mag = 20 * np.log10(np.maximum(1e-5, mag))  # 转换为分贝值，避免log(0)

    # normalize（归一化）
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)  # 归一化到[0,1]范围
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)  # 归一化到[0,1]范围

    # Transpose（转置）
    mel = mel.T.astype(np.float32)  # (T, n_mels)，转置并转换为float32类型
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)，转置并转换为float32类型

    return mel, mag  # 返回梅尔谱和线性谱

def spectrogram2wav(mag):
    '''从线性幅度谱图生成波形文件
    # Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)（形状为(T, 1+n_fft//2)的numpy数组）
    Returns:
      wav: A 1-D numpy array.（一维numpy数组，表示波形）
    '''
    # transpose（转置）
    mag = mag.T  # 转置谱图

    # de-noramlize（反归一化）
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db  # 将归一化的谱图转换回分贝值

    # to amplitude（转换为幅度）
    mag = np.power(10.0, mag * 0.05)  # 将分贝值转换为幅度值

    # wav reconstruction（波形重建）
    wav = griffin_lim(mag**hp.power)  # 使用Griffin-Lim算法重建波形

    # de-preemphasis（去预加重）
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)  # 应用去预加重滤波器

    # trim（修剪）
    wav, _ = librosa.effects.trim(wav)  # 去除音频开头和结尾的静音部分

    return wav.astype(np.float32)  # 返回float32类型的波形

def griffin_lim(spectrogram):
    '''应用Griffin-Lim算法重建相位信息
    Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)  # 深拷贝谱图
    for i in range(hp.n_iter):  # 迭代指定次数
        X_t = invert_spectrogram(X_best)  # 应用逆短时傅里叶变换
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # 应用短时傅里叶变换
        phase = est / np.maximum(1e-8, np.abs(est))  # 提取相位信息
        X_best = spectrogram * phase  # 使用原始幅度和估计相位更新谱图
    X_t = invert_spectrogram(X_best)  # 最终的逆短时傅里叶变换
    y = np.real(X_t)  # 取实部作为波形

    return y  # 返回重建的波形

def invert_spectrogram(spectrogram):
    '''应用逆傅里叶变换
    Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]（形状为[1+n_fft//2, t]的谱图）
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")  # 使用汉宁窗应用逆短时傅里叶变换

def get_positional_table(d_pos_vec, n_position=1024):
    '''获取位置编码表
    Args:
      d_pos_vec: 位置向量的维度
      n_position: 位置数量，默认为1024
    Returns:
      位置编码表，形状为[n_position, d_pos_vec]的张量
    '''
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]  # 计算位置编码的原始值
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])  # 位置0使用全零向量

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # 偶数维度使用正弦函数
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # 奇数维度使用余弦函数
    return t.from_numpy(position_enc).type(t.FloatTensor)  # 转换为PyTorch张量

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' 正弦位置编码表（Sinusoid position encoding table）'''

    def cal_angle(position, hid_idx):
        '''计算角度值'''
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)  # 计算位置编码的角度

    def get_posi_angle_vec(position):
        '''获取位置的角度向量'''
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]  # 为每个隐藏维度计算角度

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])  # 为每个位置创建角度向量

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维度使用正弦函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维度使用余弦函数

    if padding_idx is not None:
        # 为填充索引创建零向量
        sinusoid_table[padding_idx] = 0.  # 填充位置使用全零向量

    return t.FloatTensor(sinusoid_table)  # 转换为PyTorch张量

def guided_attention(N, T, g=0.2):
    '''引导注意力矩阵。参考论文第3页。
    Guided attention. Refer to page 3 on the paper.
    Args:
      N: 编码器序列长度
      T: 解码器序列长度
      g: 高斯分布的标准差，默认为0.2
    Returns:
      形状为(N, T)的引导注意力权重矩阵
    '''
    W = np.zeros((N, T), dtype=np.float32)  # 初始化权重矩阵
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            # 计算注意力权重，使用高斯分布鼓励对角线对齐
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(T) - n_pos / float(N)) ** 2 / (2 * g * g))
    return W  # 返回引导注意力权重矩阵
