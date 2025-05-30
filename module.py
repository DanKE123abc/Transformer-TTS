import torch.nn as nn  # 导入PyTorch神经网络模块
import torch as t  # 导入PyTorch，使用别名t
import torch.nn.functional as F  # 导入PyTorch函数式接口
import math  # 导入数学库
import hyperparams as hp  # 导入超参数
from text.symbols import symbols  # 导入文本符号
import numpy as np  # 导入NumPy，使用别名np
import copy  # 导入复制模块
from collections import OrderedDict  # 导入有序字典

def clones(module, N):
    # 克隆模块N次，返回一个ModuleList
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Linear(nn.Module):
    """
    Linear Module
    线性模块
    这是对PyTorch的nn.Linear的封装，提供了更好的权重初始化
    在Transformer-TTS中广泛使用于各种线性变换，如注意力机制中的查询/键/值投影、前馈网络等
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param bias: 布尔值。如果为True，则包含偏置项
        :param w_init: 字符串。权重初始化方法，默认为'linear'，使用xavier初始化
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)  # 创建线性层

        # 使用xavier均匀初始化权重
        # xavier初始化可以使得每层输出的方差大致相等，有助于解决深度网络中的梯度消失/爆炸问题
        # gain参数根据激活函数类型调整初始化范围
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        """
        线性层的前向传播
        
        :param x: 输入张量，形状为 [..., in_dim]
        :return: 线性变换后的输出，形状为 [..., out_dim]
        """
        # 前向传播，返回线性变换结果
        # 计算 y = xW^T + b，其中W是权重矩阵，b是偏置向量
        return self.linear_layer(x)


class Conv(nn.Module):
    """
    Convolution Module
    卷积模块
    这是对PyTorch的nn.Conv1d的封装，提供了更好的权重初始化
    在Transformer-TTS中用于处理序列数据，特别是在CBHG模块和PostConvNet中
    一维卷积在时序数据处理中可以捕获局部特征和上下文信息
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小，决定了感受野的大小
        :param stride: 步长大小，控制卷积核移动的步长
        :param padding: 填充大小，用于控制输出序列长度
        :param dilation: 扩张率，用于增加感受野而不增加参数量
        :param bias: 布尔值。如果为True，则包含偏置项
        :param w_init: 字符串。权重初始化方法，默认为'linear'，使用xavier初始化
        """
        super(Conv, self).__init__()

        # 创建一维卷积层
        # 一维卷积适用于序列数据，如语音或文本
        # 输入形状为 [batch_size, in_channels, seq_len]
        # 输出形状为 [batch_size, out_channels, new_seq_len]
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        # 使用xavier均匀初始化权重
        # 这有助于在深度网络中保持梯度的稳定性
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        """
        卷积层的前向传播
        
        :param x: 输入张量，形状为 [batch_size, in_channels, seq_len]
        :return: 卷积后的输出，形状为 [batch_size, out_channels, new_seq_len]
        """
        # 前向传播，应用卷积操作
        # 卷积操作会在序列维度上滑动卷积核，提取局部特征
        x = self.conv(x)
        return x


class EncoderPrenet(nn.Module):
    """
    Pre-network for Encoder consists of convolution networks.
    编码器的前置网络，由卷积网络组成。
    这个网络负责将文本符号的嵌入向量转换为更高级的特征表示
    通过三层卷积网络和一个投影层，提取文本的局部和全局特征
    在Transformer-TTS中，作为编码器的输入预处理网络
    """
    def __init__(self, embedding_size, num_hidden):
        """
        初始化编码器前置网络
        
        :param embedding_size: 嵌入向量的维度
        :param num_hidden: 隐藏层的维度
        """
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size  # 嵌入大小
        # 创建嵌入层，将文本符号索引转换为密集向量表示
        # padding_idx=0表示填充索引，对应的嵌入向量将被初始化为零并在训练中保持不变
        self.embed = nn.Embedding(len(symbols), embedding_size, padding_idx=0)  

        # 第一个卷积层：将嵌入向量转换为隐藏表示
        self.conv1 = Conv(in_channels=embedding_size,
                          out_channels=num_hidden,
                          kernel_size=5,  # 5x1的卷积核，捕获局部上下文
                          padding=int(np.floor(5 / 2)),  # 保持输出大小与输入相同
                          w_init='relu')  # 使用适合ReLU的权重初始化
        # 第二个卷积层：进一步提取特征
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        # 第三个卷积层：最终特征提取
        self.conv3 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        # 批归一化层：用于稳定训练过程，加速收敛
        self.batch_norm1 = nn.BatchNorm1d(num_hidden)  # 第一层的批归一化
        self.batch_norm2 = nn.BatchNorm1d(num_hidden)  # 第二层的批归一化
        self.batch_norm3 = nn.BatchNorm1d(num_hidden)  # 第三层的批归一化

        # Dropout层：随机丢弃神经元，防止过拟合
        self.dropout1 = nn.Dropout(p=0.2)  # 20%的丢弃率
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        # 投影层：最终的线性变换，保持维度不变
        # 用于将卷积特征映射到适合Transformer编码器的表示空间
        self.projection = Linear(num_hidden, num_hidden)

    def forward(self, input_):
        """
        编码器前置网络的前向传播
        
        :param input_: 输入的文本索引序列，形状为 [batch_size, seq_len]
        :return: 处理后的特征序列，形状为 [batch_size, seq_len, num_hidden]
        """
        # 步骤1：文本嵌入 - 将文本索引转换为嵌入向量
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_size]
        input_ = self.embed(input_)  
        
        # 步骤2：维度转置 - 为卷积操作准备输入格式
        # [batch_size, seq_len, embedding_size] -> [batch_size, embedding_size, seq_len]
        input_ = input_.transpose(1, 2)  
        
        # 步骤3：三层卷积网络处理
        # 第一个卷积块：卷积->批归一化->ReLU->Dropout
        # [batch_size, embedding_size, seq_len] -> [batch_size, num_hidden, seq_len]
        input_ = self.dropout1(t.relu(self.batch_norm1(self.conv1(input_))))  
        
        # 第二个卷积块：进一步提取特征
        # [batch_size, num_hidden, seq_len] -> [batch_size, num_hidden, seq_len]
        input_ = self.dropout2(t.relu(self.batch_norm2(self.conv2(input_))))  
        
        # 第三个卷积块：最终特征提取
        # [batch_size, num_hidden, seq_len] -> [batch_size, num_hidden, seq_len]
        input_ = self.dropout3(t.relu(self.batch_norm3(self.conv3(input_))))  
        
        # 步骤4：维度转置回原格式，为后续处理准备
        # [batch_size, num_hidden, seq_len] -> [batch_size, seq_len, num_hidden]
        input_ = input_.transpose(1, 2)  
        
        # 步骤5：通过投影层，进行最终的特征变换
        # [batch_size, seq_len, num_hidden] -> [batch_size, seq_len, num_hidden]
        input_ = self.projection(input_)  

        return input_  # 返回处理后的特征序列


class FFN(nn.Module):
    """
    Positionwise Feed-Forward Network
    按位置的前馈神经网络
    这是Transformer架构中的标准组件，用于在注意力层之后进行非线性变换
    采用两层卷积网络实现，第一层扩展维度，第二层压缩回原始维度
    每个位置独立处理，不共享位置间的信息（这与注意力机制形成互补）
    """
    
    def __init__(self, num_hidden):
        """
        初始化前馈神经网络
        
        :param num_hidden: dimension of hidden 
        :param num_hidden: 隐藏层维度，输入和输出的维度
        """
        super(FFN, self).__init__()
        # 第一个卷积层，扩展维度到4倍
        # 使用1x1卷积（等价于对每个位置应用相同的全连接层）
        self.w_1 = Conv(num_hidden, num_hidden * 4, kernel_size=1, w_init='relu')
        # 第二个卷积层，将维度恢复
        # 同样使用1x1卷积，将扩展的维度压缩回原始大小
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)  # Dropout层，防止过拟合，丢弃率为10%
        self.layer_norm = nn.LayerNorm(num_hidden)  # 层归一化，用于稳定训练

    def forward(self, input_):
        """
        前馈神经网络的前向传播
        
        :param input_: 输入张量，形状为 [batch_size, seq_len, num_hidden]
        :return: 处理后的特征，形状与输入相同 [batch_size, seq_len, num_hidden]
        """
        # FFN Network（前馈神经网络）
        # 步骤1：维度转置 - 为卷积操作准备输入格式
        # [batch_size, seq_len, num_hidden] -> [batch_size, num_hidden, seq_len]
        x = input_.transpose(1, 2)  
        
        # 步骤2：两层卷积网络处理
        # 第一层扩展维度：[batch_size, num_hidden, seq_len] -> [batch_size, num_hidden*4, seq_len]
        # 第二层压缩维度：[batch_size, num_hidden*4, seq_len] -> [batch_size, num_hidden, seq_len]
        # 中间使用ReLU激活函数引入非线性
        x = self.w_2(t.relu(self.w_1(x)))  
        
        # 步骤3：维度转置回原格式
        # [batch_size, num_hidden, seq_len] -> [batch_size, seq_len, num_hidden]
        x = x.transpose(1, 2)  

        # 步骤4：残差连接 - 将输入直接加到输出上
        # 这有助于解决深度网络中的梯度消失问题，并允许信息直接流动
        # [batch_size, seq_len, num_hidden] + [batch_size, seq_len, num_hidden]
        x = x + input_  

        # 注意：这里的dropout在代码中被注释掉了，实际未使用
        # x = self.dropout(x) 

        # 步骤5：层归一化 - 标准化每个位置的特征
        # 这有助于稳定训练过程，加速收敛
        x = self.layer_norm(x)  
        
        return x  # 返回处理后的特征

        return x


class PostConvNet(nn.Module):
    """
    Post Convolutional Network (mel --> mel)
    后卷积网络（梅尔频谱图 --> 梅尔频谱图）
    """
    def __init__(self, num_hidden):
        """
        
        :param num_hidden: dimension of hidden 
        :param num_hidden: 隐藏层维度
        """
        super(PostConvNet, self).__init__()
        # 第一个卷积层
        self.conv1 = Conv(in_channels=hp.num_mels * hp.outputs_per_step,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=4,  # 填充为4，用于因果卷积
                          w_init='tanh')
        # 克隆3个相同的卷积层
        self.conv_list = clones(Conv(in_channels=num_hidden,
                                     out_channels=num_hidden,
                                     kernel_size=5,
                                     padding=4,
                                     w_init='tanh'), 3)
        # 最后一个卷积层，将维度恢复
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=hp.num_mels * hp.outputs_per_step,
                          kernel_size=5,
                          padding=4)

        # 批归一化层
        self.batch_norm_list = clones(nn.BatchNorm1d(num_hidden), 3)  # 克隆3个批归一化层
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)  # 第一个卷积层的批归一化

        # Dropout层
        self.dropout1 = nn.Dropout(p=0.1)  # 第一个卷积层的Dropout
        self.dropout_list = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(3)])  # 中间卷积层的Dropout

    def forward(self, input_, mask=None):
        # Causal Convolution (for auto-regressive)（因果卷积，用于自回归）
        # 应用第一个卷积层，然后批归一化，tanh激活函数，最后Dropout
        # [:, :, :-4]用于实现因果卷积，只使用过去的信息
        input_ = self.dropout1(t.tanh(self.pre_batchnorm(self.conv1(input_)[:, :, :-4])))
        # 应用中间的卷积层
        for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
            input_ = dropout(t.tanh(batch_norm(conv(input_)[:, :, :-4])))
        # 应用最后一个卷积层
        input_ = self.conv2(input_)[:, :, :-4]
        return input_


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    多头注意力机制（点积注意力）
    实现注意力机制的核心计算，包括缩放点积注意力和上下文向量生成
    """
    def __init__(self, num_hidden_k):
        """
        初始化多头注意力模块
        :param num_hidden_k: dimension of hidden（隐藏层维度，用于缩放点积）
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k  # 隐藏层维度，用于缩放点积
        self.attn_dropout = nn.Dropout(p=0.1)  # 注意力的Dropout层，防止过拟合

    def forward(self, key, value, query, mask=None, query_mask=None):
        """
        多头注意力前向传播函数
        :param key: 键张量 [batch_size, seq_k, d_k]
        :param value: 值张量 [batch_size, seq_k, d_v]
        :param query: 查询张量 [batch_size, seq_q, d_q]
        :param mask: 掩码张量，用于屏蔽键序列中的填充位置 [batch_size, seq_q, seq_k]
        :param query_mask: 查询掩码，用于屏蔽查询序列中的填充位置 [batch_size, seq_q, seq_k]
        :return: 上下文向量和注意力权重
        """
        # Get attention score（计算注意力分数）
        # 使用批量矩阵乘法(bmm)计算查询和键的点积，得到原始注意力分数
        attn = t.bmm(query, key.transpose(1, 2))  # [batch_size, seq_q, seq_k]
        
        # 缩放点积，防止梯度消失
        # 当维度很大时，点积的方差也会很大，导致softmax后的梯度很小
        # 除以sqrt(d_k)可以稳定梯度
        attn = attn / math.sqrt(self.num_hidden_k)  # [batch_size, seq_q, seq_k]

        # Masking to ignore padding (key side)（使用掩码忽略键侧的填充）
        if mask is not None:
            # 将掩码位置的值设为非常小的负数，使softmax后接近0
            # 这通常用于屏蔽填充位置或未来位置（在解码器中）
            attn = attn.masked_fill(mask, -2 ** 32 + 1)  # [batch_size, seq_q, seq_k]
            attn = t.softmax(attn, dim=-1)  # 应用softmax得到注意力权重，在键序列维度上归一化
        else:
            attn = t.softmax(attn, dim=-1)  # 如果没有掩码，直接应用softmax

        # Masking to ignore padding (query side)（使用掩码忽略查询侧的填充）
        if query_mask is not None:
            # 应用查询掩码，将查询序列中填充位置的注意力权重置为0
            attn = attn * query_mask  # [batch_size, seq_q, seq_k]

        # Dropout（在注释中，实际未使用）
        # attn = self.attn_dropout(attn)  # 对注意力权重应用dropout，增加模型鲁棒性
        
        # Get Context Vector（获取上下文向量）
        # 使用注意力权重对值进行加权求和，得到上下文向量
        # 这是注意力机制的核心：根据相关性加权聚合信息
        result = t.bmm(attn, value)  # [batch_size, seq_q, d_v]

        return result, attn  # 返回上下文向量和注意力权重矩阵


class Attention(nn.Module):
    """
    Attention Network（注意力网络）
    实现多头注意力机制，用于编码器自注意力、解码器自注意力和编码器-解码器注意力
    """
    def __init__(self, num_hidden, h=4):
        """
        初始化注意力网络
        :param num_hidden: dimension of hidden（隐藏层维度）
        :param h: num of heads（注意力头数量）
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden  # 隐藏层总维度
        self.num_hidden_per_attn = num_hidden // h  # 每个注意力头的维度
        self.h = h  # 注意力头的数量

        # 线性变换层，用于生成键、值和查询
        self.key = Linear(num_hidden, num_hidden, bias=False)  # 键的线性变换
        self.value = Linear(num_hidden, num_hidden, bias=False)  # 值的线性变换
        self.query = Linear(num_hidden, num_hidden, bias=False)  # 查询的线性变换

        # 多头注意力机制
        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        # 残差连接的Dropout
        self.residual_dropout = nn.Dropout(p=0.1)

        # 最终的线性变换，注意输入维度是num_hidden*2，因为会连接输入和上下文向量
        self.final_linear = Linear(num_hidden * 2, num_hidden)  # 输出投影层

        # 层归一化，用于稳定训练
        self.layer_norm_1 = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):
        """
        注意力网络前向传播函数
        :param memory: 键和值的来源，可以是编码器输出（编码器-解码器注意力）或当前层输入（自注意力）
        :param decoder_input: 查询的来源，通常是当前层的输入
        :param mask: 掩码矩阵，用于屏蔽不应该被注意的位置（如填充位置或未来位置）
        :param query_mask: 查询掩码，用于屏蔽查询序列中的填充位置
        :return: 注意力输出和注意力权重
        """

        batch_size = memory.size(0)  # 获取批量大小
        seq_k = memory.size(1)  # 获取键序列长度（源序列长度）
        seq_q = decoder_input.size(1)  # 获取查询序列长度（目标序列长度）
        
        # Repeat masks h times（将掩码重复h次，用于多头注意力）
        if query_mask is not None:
            # 扩展查询掩码维度并重复，使其形状与注意力计算兼容
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)  # [B, seq_q, seq_k]
            query_mask = query_mask.repeat(self.h, 1, 1)  # [h*B, seq_q, seq_k]，为每个注意力头重复掩码
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)  # [h*B, seq_q, seq_k]，为每个注意力头重复掩码

        # Make multihead（创建多头注意力的输入）
        # 将输入转换为多头形式，每个头有自己的键、值和查询
        # 线性变换后分割为多个头
        key = self.key(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)  # [B, seq_k, h, d_k]
        value = self.value(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)  # [B, seq_k, h, d_v]
        query = self.query(decoder_input).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)  # [B, seq_q, h, d_q]

        # 重新排列维度，便于并行计算多头注意力
        # 将头的维度移到最前面，并合并头和批次维度
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)  # [h*B, seq_k, d_k]
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)  # [h*B, seq_k, d_v]
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)  # [h*B, seq_q, d_q]

        # Get context vector（获取上下文向量）
        # 应用多头注意力计算，得到上下文向量和注意力权重
        result, attns = self.multihead(key, value, query, mask=mask, query_mask=query_mask)  # [h*B, seq_q, d_v], [h*B, seq_q, seq_k]

        # Concatenate all multihead context vector（连接所有多头注意力的上下文向量）
        # 首先将结果重塑回原始的多头形式
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)  # [h, B, seq_q, d_v]
        # 然后将头的维度移回原位，并合并所有头的输出
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)  # [B, seq_q, h*d_v]
        
        # Concatenate context vector with input (most important)（连接上下文向量和输入，这是最重要的步骤）
        # 这种连接方式类似于残差网络中的跳跃连接，有助于信息流动
        result = t.cat([decoder_input, result], dim=-1)  # [B, seq_q, d_model*2]
        
        # Final linear（最终的线性变换）
        # 将连接后的向量投影回原始维度
        result = self.final_linear(result)  # [B, seq_q, d_model]

        # Residual dropout & connection（残差连接）
        # 添加残差连接，将原始输入加到注意力输出上，帮助梯度流动
        result = result + decoder_input  # [B, seq_q, d_model]

        # result = self.residual_dropout(result)  # 在注释中，实际未使用

        # Layer normalization（层归一化）
        # 应用层归一化，标准化输出，稳定训练过程
        result = self.layer_norm_1(result)  # [B, seq_q, d_model]
        
        return result, attns  # 返回注意力输出和注意力权重矩阵
    

class Prenet(nn.Module):
    """
    Prenet before passing through the network
    通过网络前的前置网络
    前置网络是一个简单的前馈神经网络，用于特征转换和降维
    在Transformer-TTS中，用于处理编码器和解码器的输入
    特点是在训练和推理时都使用Dropout，这有助于增加模型的鲁棒性
    """
    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        
        :param input_size: 输入维度
        :param hidden_size: 隐藏单元维度
        :param output_size: 输出维度
        :param p: Dropout概率，默认为0.5，较高的Dropout率有助于防止过拟合
        """
        super(Prenet, self).__init__()
        self.input_size = input_size  # 输入大小
        self.output_size = output_size  # 输出大小
        self.hidden_size = hidden_size  # 隐藏层大小
        # 创建一个有序的网络层序列
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', Linear(self.input_size, self.hidden_size)),  # 第一个全连接层，将输入映射到隐藏维度
             ('relu1', nn.ReLU()),  # ReLU激活函数，引入非线性
             ('dropout1', nn.Dropout(p)),  # Dropout层，随机丢弃一部分神经元，防止过拟合
             ('fc2', Linear(self.hidden_size, self.output_size)),  # 第二个全连接层，将隐藏表示映射到输出维度
             ('relu2', nn.ReLU()),  # ReLU激活函数
             ('dropout2', nn.Dropout(p)),  # Dropout层，注意在推理时也会保持激活状态
        ]))

    def forward(self, input_):
        """
        前置网络的前向传播
        
        :param input_: 输入张量，形状为 [batch_size, seq_len, input_size]
        :return: 经过前置网络处理后的输出，形状为 [batch_size, seq_len, output_size]
        """
        # 前向传播，通过网络层序列
        # 注意：即使在推理阶段，Dropout也会保持激活状态
        # 这是Prenet的一个特殊设计，有助于增加模型的鲁棒性和泛化能力
        out = self.layer(input_)

        return out
    
class CBHG(nn.Module):
    """
    CBHG Module
    CBHG模块（卷积层+高速公路网络+双向GRU）
    CBHG是一个强大的模块，由卷积组(Conv1D Bank)、瓶颈层(Bottleneck layers)、高速公路网络(Highway)和双向GRU组成
    用于提取序列的特征表示，在语音合成中用于将梅尔谱转换为线性谱
    """
    def __init__(self, hidden_size, K=16, projection_size = 256, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        """
        :param hidden_size: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        
        :param hidden_size: 隐藏单元维度
        :param K: 卷积组的数量
        :param projection_size: 投影单元维度
        :param num_gru_layers: GRU单元的层数
        :param max_pool_kernel_size: 最大池化核大小
        :param is_post: 是否为后处理
        """
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size  # 隐藏层大小
        self.projection_size = projection_size  # 投影大小
        
        # 创建卷积组列表
        self.convbank_list = nn.ModuleList()
        # 添加第一个卷积层，核大小为1
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size,
                                                out_channels=hidden_size,
                                                kernel_size=1,
                                                padding=int(np.floor(1/2))))

        # 添加剩余的卷积层，核大小从2到K
        for i in range(2, K+1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=i,
                                                padding=int(np.floor(i/2))))

        # 创建批归一化层列表
        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K+1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        # 卷积组的输出维度
        convbank_outdim = hidden_size * K
        
        # 投影卷积层1
        self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size,
                                             kernel_size=3,
                                             padding=int(np.floor(3 / 2)))
        # 投影卷积层2
        self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3 / 2)))
        # 投影层的批归一化
        self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)

        # 最大池化层
        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        # 高速公路网络
        self.highway = Highwaynet(self.projection_size)
        # 双向GRU
        self.gru = nn.GRU(self.projection_size, self.hidden_size // 2, num_layers=num_gru_layers,
                          batch_first=True,
                          bidirectional=True)  # 双向GRU，输出维度为hidden_size


    def _conv_fit_dim(self, x, kernel_size=3):
        # 调整卷积输出的维度，确保与输入维度匹配
        # 如果核大小是偶数，则去掉最后一个元素，否则保持不变
        if kernel_size % 2 == 0:
            return x[:,:,:-1]  # 去掉最后一个元素
        else:
            return x  # 保持不变

    def forward(self, input_):
        # 确保输入是连续的内存布局
        input_ = input_.contiguous()
        batch_size = input_.size(0)  # 获取批量大小
        total_length = input_.size(-1)  # 获取序列长度

        convbank_list = list()  # 存储卷积组的输出
        convbank_input = input_  # 初始输入

        # Convolution bank filters（卷积组滤波器）
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            # 应用卷积、维度调整、批归一化和ReLU激活
            convbank_input = t.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input)  # 将结果添加到列表中

        # Concatenate all features（连接所有特征）
        conv_cat = t.cat(convbank_list, dim=1)  # 在通道维度上连接

        # Max pooling（最大池化）
        conv_cat = self.max_pool(conv_cat)[:,:,:-1]  # 应用最大池化并去掉最后一个元素

        # Projection（投影）
        # 应用第一个投影卷积、维度调整、批归一化和ReLU激活
        conv_projection = t.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        # 应用第二个投影卷积、维度调整、批归一化，并添加残差连接
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_

        # Highway networks（高速公路网络）
        highway = self.highway.forward(conv_projection.transpose(1,2))  # 转置维度并应用高速公路网络
        

        # Bidirectional GRU（双向GRU）
        # 展平参数以提高效率
        self.gru.flatten_parameters()
        out, _ = self.gru(highway)  # 应用GRU，忽略隐藏状态

        return out  # 返回GRU的输出


class Highwaynet(nn.Module):
    """
    Highway network（高速公路网络）
    高速公路网络是一种允许信息无阻碍流动的网络结构
    通过门控机制控制信息流动，类似于LSTM/GRU中的门控机制
    在CBHG模块中用于特征转换和信息流控制
    """
    def __init__(self, num_units, num_layers=4):
        """
        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        
        :param num_units: 隐藏单元的维度
        :param num_layers: 高速公路层的数量
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units  # 隐藏单元数量
        self.num_layers = num_layers  # 网络层数，默认为4
        self.gates = nn.ModuleList()  # 门控层列表，用于计算变换门（transform gate）
        self.linears = nn.ModuleList()  # 线性变换层列表，用于特征变换
        for _ in range(self.num_layers):
            self.linears.append(Linear(num_units, num_units))  # 添加线性变换层，维持输入输出维度相同
            self.gates.append(Linear(num_units, num_units))  # 添加门控层，用于计算变换门的值

    def forward(self, input_):
        """
        高速公路网络的前向传播
        
        :param input_: 输入张量，形状为 [batch_size, seq_len, num_units]
        :return: 经过高速公路网络处理后的输出，形状与输入相同
        """
        # 前向传播
        out = input_  # 初始输出等于输入

        # highway gated function（高速公路门控函数）
        for fc1, fc2 in zip(self.linears, self.gates):
            # fc1是线性变换层，fc2是门控层
            h = t.relu(fc1.forward(out))  # 应用ReLU激活的线性变换，生成候选特征
            t_ = t.sigmoid(fc2.forward(out))  # 应用Sigmoid激活的门控变换，生成变换门（取值范围0-1）

            c = 1. - t_  # 计算携带门（carry gate），携带门 = 1 - 变换门
            out = h * t_ + out * c  # 高速公路网络的核心公式：输出 = 变换门*变换输入 + 携带门*原始输入
            # 当变换门接近1时，更多依赖变换后的特征；当变换门接近0时，更多保留原始输入

        return out  # 返回最终输出，经过多层高速公路网络处理
