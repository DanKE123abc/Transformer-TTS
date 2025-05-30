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
        :param w_init: 字符串。使用xavier初始化权重
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)  # 创建线性层

        # 使用xavier均匀初始化权重
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        # 前向传播，返回线性变换结果
        return self.linear_layer(x)


class Conv(nn.Module):
    """
    Convolution Module
    卷积模块
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
        :param kernel_size: 卷积核大小
        :param stride: 步长大小
        :param padding: 填充大小
        :param dilation: 扩张率
        :param bias: 布尔值。如果为True，则包含偏置项
        :param w_init: 字符串。使用xavier初始化权重
        """
        super(Conv, self).__init__()

        # 创建一维卷积层
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        # 使用xavier均匀初始化权重
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        # 前向传播，应用卷积操作
        x = self.conv(x)
        return x


class EncoderPrenet(nn.Module):
    """
    Pre-network for Encoder consists of convolution networks.
    编码器的前置网络，由卷积网络组成。
    """
    def __init__(self, embedding_size, num_hidden):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size  # 嵌入大小
        self.embed = nn.Embedding(len(symbols), embedding_size, padding_idx=0)  # 创建嵌入层，padding_idx=0表示填充索引

        # 第一个卷积层
        self.conv1 = Conv(in_channels=embedding_size,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),  # 保持输出大小与输入相同
                          w_init='relu')
        # 第二个卷积层
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        # 第三个卷积层
        self.conv3 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        # 批归一化层
        self.batch_norm1 = nn.BatchNorm1d(num_hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_hidden)
        self.batch_norm3 = nn.BatchNorm1d(num_hidden)

        # Dropout层，防止过拟合
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.projection = Linear(num_hidden, num_hidden)  # 投影层，保持维度不变

    def forward(self, input_):
        input_ = self.embed(input_)  # 将输入转换为嵌入向量
        input_ = input_.transpose(1, 2)  # 转置维度，便于卷积操作
        input_ = self.dropout1(t.relu(self.batch_norm1(self.conv1(input_))))  # 第一个卷积块：卷积->批归一化->ReLU->Dropout
        input_ = self.dropout2(t.relu(self.batch_norm2(self.conv2(input_))))  # 第二个卷积块
        input_ = self.dropout3(t.relu(self.batch_norm3(self.conv3(input_))))  # 第三个卷积块
        input_ = input_.transpose(1, 2)  # 转置回原来的维度
        input_ = self.projection(input_)  # 通过投影层

        return input_


class FFN(nn.Module):
    """
    Positionwise Feed-Forward Network
    按位置的前馈神经网络
    """
    
    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden 
        :param num_hidden: 隐藏层维度
        """
        super(FFN, self).__init__()
        # 第一个卷积层，扩展维度到4倍
        self.w_1 = Conv(num_hidden, num_hidden * 4, kernel_size=1, w_init='relu')
        # 第二个卷积层，将维度恢复
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)  # Dropout层，防止过拟合
        self.layer_norm = nn.LayerNorm(num_hidden)  # 层归一化

    def forward(self, input_):
        # FFN Network（前馈神经网络）
        x = input_.transpose(1, 2)  # 转置维度，便于卷积操作
        x = self.w_2(t.relu(self.w_1(x)))  # 两层卷积网络，中间使用ReLU激活函数
        x = x.transpose(1, 2)  # 转置回原来的维度


        # residual connection（残差连接）
        x = x + input_  # 添加残差连接

        # dropout（在注释中，实际未使用）
        # x = self.dropout(x) 

        # layer normalization（层归一化）
        x = self.layer_norm(x)  # 应用层归一化

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
    """
    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden 
        :param num_hidden_k: 隐藏层维度
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k  # 隐藏层维度
        self.attn_dropout = nn.Dropout(p=0.1)  # 注意力的Dropout层

    def forward(self, key, value, query, mask=None, query_mask=None):
        # Get attention score（计算注意力分数）
        attn = t.bmm(query, key.transpose(1, 2))  # 批量矩阵乘法，计算查询和键的点积
        attn = attn / math.sqrt(self.num_hidden_k)  # 缩放点积，防止梯度消失

        # Masking to ignore padding (key side)（使用掩码忽略键侧的填充）
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1)  # 将掩码位置的值设为非常小的负数
            attn = t.softmax(attn, dim=-1)  # 应用softmax得到注意力权重
        else:
            attn = t.softmax(attn, dim=-1)  # 如果没有掩码，直接应用softmax

        # Masking to ignore padding (query side)（使用掩码忽略查询侧的填充）
        if query_mask is not None:
            attn = attn * query_mask  # 应用查询掩码

        # Dropout（在注释中，实际未使用）
        # attn = self.attn_dropout(attn)
        
        # Get Context Vector（获取上下文向量）
        result = t.bmm(attn, value)  # 使用注意力权重对值进行加权求和

        return result, attn  # 返回上下文向量和注意力权重


class Attention(nn.Module):
    """
    Attention Network
    注意力网络
    """
    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        
        :param num_hidden: 隐藏层维度
        :param h: 注意力头的数量
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden  # 隐藏层维度
        self.num_hidden_per_attn = num_hidden // h  # 每个注意力头的隐藏层维度
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
        self.final_linear = Linear(num_hidden * 2, num_hidden)

        # 层归一化
        self.layer_norm_1 = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):

        batch_size = memory.size(0)  # 获取批量大小
        seq_k = memory.size(1)  # 获取键序列长度
        seq_q = decoder_input.size(1)  # 获取查询序列长度
        
        # Repeat masks h times（将掩码重复h次，用于多头注意力）
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)  # 扩展查询掩码维度并重复
            query_mask = query_mask.repeat(self.h, 1, 1)  # 为每个注意力头重复掩码
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)  # 为每个注意力头重复掩码

        # Make multihead（创建多头注意力的输入）
        # 将输入转换为多头形式，每个头有自己的键、值和查询
        key = self.key(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)  # 生成键
        value = self.value(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)  # 生成值
        query = self.query(decoder_input).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)  # 生成查询

        # 重新排列维度，便于并行计算多头注意力
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector（获取上下文向量）
        result, attns = self.multihead(key, value, query, mask=mask, query_mask=query_mask)  # 应用多头注意力

        # Concatenate all multihead context vector（连接所有多头注意力的上下文向量）
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)  # 重塑为原始形状
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)  # 重新排列维度
        
        # Concatenate context vector with input (most important)（连接上下文向量和输入，这是最重要的步骤）
        result = t.cat([decoder_input, result], dim=-1)  # 在最后一个维度上连接
        
        # Final linear（最终的线性变换）
        result = self.final_linear(result)  # 应用线性变换

        # Residual dropout & connection（残差连接）
        result = result + decoder_input  # 添加残差连接

        # result = self.residual_dropout(result)  # 在注释中，实际未使用

        # Layer normalization（层归一化）
        result = self.layer_norm_1(result)  # 应用层归一化

        return result, attns  # 返回结果和注意力权重
    

class Prenet(nn.Module):
    """
    Prenet before passing through the network
    通过网络前的前置网络
    """
    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        
        :param input_size: 输入维度
        :param hidden_size: 隐藏单元维度
        :param output_size: 输出维度
        :param p: Dropout概率
        """
        super(Prenet, self).__init__()
        self.input_size = input_size  # 输入大小
        self.output_size = output_size  # 输出大小
        self.hidden_size = hidden_size  # 隐藏层大小
        # 创建一个有序的网络层序列
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', Linear(self.input_size, self.hidden_size)),  # 第一个全连接层
             ('relu1', nn.ReLU()),  # ReLU激活函数
             ('dropout1', nn.Dropout(p)),  # Dropout层
             ('fc2', Linear(self.hidden_size, self.output_size)),  # 第二个全连接层
             ('relu2', nn.ReLU()),  # ReLU激活函数
             ('dropout2', nn.Dropout(p)),  # Dropout层
        ]))

    def forward(self, input_):
        # 前向传播，通过网络层序列
        out = self.layer(input_)

        return out
    
class CBHG(nn.Module):
    """
    CBHG Module
    CBHG模块（卷积层+高速公路网络+双向GRU）
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
    """
    def __init__(self, num_units, num_layers=4):
        """
        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units  # 隐藏单元数量
        self.num_layers = num_layers  # 网络层数，默认为4
        self.gates = nn.ModuleList()  # 门控层列表
        self.linears = nn.ModuleList()  # 线性变换层列表
        for _ in range(self.num_layers):
            self.linears.append(Linear(num_units, num_units))  # 添加线性变换层
            self.gates.append(Linear(num_units, num_units))  # 添加门控层

    def forward(self, input_):
        # 前向传播
        out = input_  # 初始输出等于输入

        # highway gated function（高速公路门控函数）
        for fc1, fc2 in zip(self.linears, self.gates):
            # fc1是线性变换层，fc2是门控层
            h = t.relu(fc1.forward(out))  # 应用ReLU激活的线性变换
            t_ = t.sigmoid(fc2.forward(out))  # 应用Sigmoid激活的门控变换

            c = 1. - t_  # 计算携带门（carry gate）
            out = h * t_ + out * c  # 高速公路网络的核心公式：输出 = 变换门*变换输入 + 携带门*原始输入

        return out  # 返回最终输出
