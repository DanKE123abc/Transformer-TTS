# Audio（音频参数）
num_mels = 80  # 梅尔谱特征的数量
# num_freq = 1024  # 频率数量（未使用）
n_fft = 2048  # FFT窗口大小
sr = 22050  # 采样率
# frame_length_ms = 50.  # 帧长度（毫秒，未使用）
# frame_shift_ms = 12.5  # 帧移（毫秒，未使用）
preemphasis = 0.97  # 预加重系数
frame_shift = 0.0125  # 帧移（秒）
frame_length = 0.05  # 帧长度（秒）
hop_length = int(sr*frame_shift)  # 帧移（样本数）
win_length = int(sr*frame_length)  # 窗口长度（样本数）
n_mels = 80  # 生成的梅尔滤波器组数量
power = 1.2  # 预测幅度的放大指数
min_level_db = -100  # 最小分贝级别
ref_level_db = 20  # 参考分贝级别
hidden_size = 256  # 隐藏层大小
embedding_size = 512  # 嵌入层大小
max_db = 100  # 最大分贝值
ref_db = 20  # 参考分贝值
    
# 训练和生成参数
n_iter = 60  # Griffin-Lim算法的迭代次数
# power = 1.5  # 另一个幂值（未使用）
outputs_per_step = 1  # 每步输出的帧数

# 训练参数
epochs = 10000  # 训练轮数
lr = 0.001  # 学习率
save_step = 2000  # 保存检查点的步数间隔
image_step = 500  # 生成图像的步数间隔
batch_size = 32  # 批量大小

# 文本处理
cleaners='english_cleaners'  # 文本清洗器类型

# 路径设置
data_path = './data/LJSpeech-1.1'  # 数据路径
checkpoint_path = './checkpoint'  # 检查点保存路径
sample_path = './samples'  # 样本保存路径