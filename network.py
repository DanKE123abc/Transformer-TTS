from module import *  # 导入所有模块组件
from utils import get_positional_table, get_sinusoid_encoding_table  # 导入位置编码相关函数
import hyperparams as hp  # 导入超参数
import copy  # 导入复制模块

class Encoder(nn.Module):
    """
    Encoder Network（编码器网络）
    文本编码器，将输入的字符序列转换为隐藏表示
    """
    def __init__(self, embedding_size, num_hidden):
        """
        初始化编码器
        :param embedding_size: dimension of embedding（嵌入维度）
        :param num_hidden: dimension of hidden（隐藏层维度）
        """
        super(Encoder, self).__init__()
        self.alpha = nn.Parameter(t.ones(1))  # 位置编码的缩放参数，可学习
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)  # 预训练的正弦位置编码，固定不更新
        self.pos_dropout = nn.Dropout(p=0.1)  # 位置编码的dropout层，概率为0.1
        self.encoder_prenet = EncoderPrenet(embedding_size, num_hidden)  # 编码器的前置网络，处理字符嵌入
        self.layers = clones(Attention(num_hidden), 3)  # 克隆3个注意力层，形成多层自注意力
        self.ffns = clones(FFN(num_hidden), 3)  # 克隆3个前馈网络层，用于特征转换

    def forward(self, x, pos):
        """
        编码器前向传播函数
        :param x: 输入的字符序列 [batch_size, seq_len]
        :param pos: 位置索引 [batch_size, seq_len]
        :return: 编码器输出、字符掩码和注意力权重列表
        """

        # Get character mask（获取字符掩码）
        if self.training:  # 训练模式下
            c_mask = pos.ne(0).type(t.float)  # 创建字符掩码，非零位置为1，零位置为0（用于区分实际内容和填充）
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)  # 创建注意力掩码，零位置为1，非零位置为0（用于注意力计算）

        else:  # 推理模式下
            c_mask, mask = None, None  # 不使用掩码

        # Encoder pre-network（编码器前置网络处理）
        x = self.encoder_prenet(x)  # 通过编码器前置网络处理输入，将字符转换为嵌入表示

        # Get positional embedding, apply alpha and add（获取位置编码，应用alpha缩放并添加）
        pos = self.pos_emb(pos)  # 获取位置编码，为序列添加位置信息
        x = pos * self.alpha + x  # 位置编码乘以alpha参数后与输入相加，alpha是可学习的缩放因子

        # Positional dropout（位置编码dropout）
        x = self.pos_dropout(x)  # 应用dropout，防止过拟合

        # Attention encoder-encoder（编码器自注意力）
        attns = list()  # 存储注意力权重列表，用于可视化和分析
        for layer, ffn in zip(self.layers, self.ffns):  # 遍历注意力层和前馈网络层
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)  # 自注意力计算，x作为查询、键和值
            x = ffn(x)  # 前馈网络处理，进一步转换特征
            attns.append(attn)  # 保存注意力权重，用于可视化

        return x, c_mask, attns  # 返回编码器输出、字符掩码和注意力权重列表


class MelDecoder(nn.Module):
    """
    Decoder Network（解码器网络）
    梅尔谱解码器，将编码器的隐藏表示转换为梅尔谱序列
    """
    def __init__(self, num_hidden):
        """
        初始化解码器
        :param num_hidden: dimension of hidden（隐藏层维度）
        """
        super(MelDecoder, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)  # 预训练的正弦位置编码，固定不更新
        self.pos_dropout = nn.Dropout(p=0.1)  # 位置编码的dropout层，概率为0.1
        self.alpha = nn.Parameter(t.ones(1))  # 位置编码的缩放参数，可学习
        self.decoder_prenet = Prenet(hp.num_mels, num_hidden * 2, num_hidden, p=0.2)  # 解码器的前置网络，处理梅尔谱输入
        self.norm = Linear(num_hidden, num_hidden)  # 归一化线性层，用于中心化位置

        self.selfattn_layers = clones(Attention(num_hidden), 3)  # 克隆3个自注意力层，用于解码器内部注意力
        self.dotattn_layers = clones(Attention(num_hidden), 3)  # 克隆3个点积注意力层（编码器-解码器注意力），用于关注编码器输出
        self.ffns = clones(FFN(num_hidden), 3)  # 克隆3个前馈网络层，用于特征转换
        self.mel_linear = Linear(num_hidden, hp.num_mels * hp.outputs_per_step)  # 梅尔谱输出线性层，生成梅尔谱
        self.stop_linear = Linear(num_hidden, 1, w_init='sigmoid')  # 停止标记预测线性层，使用sigmoid初始化，预测序列结束

        self.postconvnet = PostConvNet(num_hidden)  # 后卷积网络，用于改善梅尔谱质量

    def forward(self, memory, decoder_input, c_mask, pos):
        """
        解码器前向传播函数
        :param memory: 编码器输出的记忆 [batch_size, text_len, hidden_size]
        :param decoder_input: 解码器输入（梅尔谱）[batch_size, mel_len, num_mels]
        :param c_mask: 字符掩码 [batch_size, text_len]
        :param pos: 位置索引 [batch_size, mel_len]
        :return: 原始梅尔谱输出、后处理梅尔谱输出、注意力权重列表和停止标记
        """
        
        batch_size = memory.size(0)  # 获取批量大小
        decoder_len = decoder_input.size(1)  # 获取解码器序列长度（梅尔谱帧数）

        # get decoder mask with triangular matrix（使用三角矩阵获取解码器掩码）
        if self.training:  # 训练模式下
            m_mask = pos.ne(0).type(t.float)  # 创建梅尔谱掩码，非零位置为1，零位置为0（区分实际内容和填充）
            mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)  # 创建填充掩码，扩展维度并重复
            if next(self.parameters()).is_cuda:  # 如果模型在GPU上
                # 创建因果掩码（上三角矩阵），确保当前位置只能看到之前的位置（自回归特性）
                mask = mask + t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:  # 如果模型在CPU上
                mask = mask + t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)  # 转换为布尔掩码
            # 创建编码器-解码器注意力的掩码，用于屏蔽填充位置
            zero_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)  # 扩展字符掩码维度
            zero_mask = zero_mask.transpose(1, 2)  # 转置掩码以匹配注意力维度
        else:  # 推理模式下
            if next(self.parameters()).is_cuda:  # 如果模型在GPU上
                # 只需要因果掩码（上三角矩阵），确保自回归生成
                mask = t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:  # 如果模型在CPU上
                mask = t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)  # 转换为布尔掩码
            m_mask, zero_mask = None, None  # 推理模式下不需要其他掩码

        # Decoder pre-network（解码器前置网络）
        decoder_input = self.decoder_prenet(decoder_input)  # 通过解码器前置网络处理输入，降维并提取特征

        # Centered position（中心化位置）
        decoder_input = self.norm(decoder_input)  # 应用归一化，稳定训练

        # Get positional embedding, apply alpha and add（获取位置编码，应用alpha缩放并添加）
        pos = self.pos_emb(pos)  # 获取位置编码，为序列添加位置信息
        decoder_input = pos * self.alpha + decoder_input  # 位置编码乘以alpha参数后与输入相加，alpha是可学习的缩放因子

        # Positional dropout（位置编码dropout）
        decoder_input = self.pos_dropout(decoder_input)  # 应用dropout，防止过拟合

        # Attention decoder-decoder, encoder-decoder（解码器自注意力和编码器-解码器注意力）
        attn_dot_list = list()  # 存储编码器-解码器注意力权重列表，用于可视化和分析
        attn_dec_list = list()  # 存储解码器自注意力权重列表，用于可视化和分析

        for selfattn, dotattn, ffn in zip(self.selfattn_layers, self.dotattn_layers, self.ffns):  # 遍历注意力层和前馈网络层
            # 解码器自注意力计算，捕捉梅尔谱内部依赖关系
            decoder_input, attn_dec = selfattn(decoder_input, decoder_input, mask=mask, query_mask=m_mask)
            # 编码器-解码器注意力计算，将文本信息融入梅尔谱生成
            decoder_input, attn_dot = dotattn(memory, decoder_input, mask=zero_mask, query_mask=m_mask)
            # 前馈网络处理，进一步转换特征
            decoder_input = ffn(decoder_input)
            # 保存注意力权重，用于可视化和分析
            attn_dot_list.append(attn_dot)  # 保存编码器-解码器注意力权重
            attn_dec_list.append(attn_dec)  # 保存解码器自注意力权重

        # Mel linear projection（梅尔谱线性投影）
        mel_out = self.mel_linear(decoder_input)  # 生成梅尔谱输出，将隐藏表示转换为梅尔谱
        
        # Post Mel Network（后梅尔网络处理）
        postnet_input = mel_out.transpose(1, 2)  # 转置维度以适应卷积网络 [B, T, D] -> [B, D, T]
        out = self.postconvnet(postnet_input)  # 通过后卷积网络处理，改善梅尔谱质量
        out = postnet_input + out  # 残差连接，保留原始信息
        out = out.transpose(1, 2)  # 转置回原始维度 [B, D, T] -> [B, T, D]

        # Stop tokens（停止标记）
        stop_tokens = self.stop_linear(decoder_input)  # 预测停止标记，判断序列是否结束

        return mel_out, out, attn_dot_list, stop_tokens, attn_dec_list  # 返回梅尔谱输出、后处理输出、注意力权重和停止标记


class Model(nn.Module):
    """
    Transformer Network（Transformer网络）
    完整的Transformer TTS模型，包含编码器和解码器
    """
    def __init__(self):
        """
        初始化Transformer模型
        创建编码器和解码器组件
        """
        super(Model, self).__init__()
        self.encoder = Encoder(hp.embedding_size, hp.hidden_size)  # 初始化编码器
        self.decoder = MelDecoder(hp.hidden_size)  # 初始化解码器

    def forward(self, characters, mel_input, pos_text, pos_mel):
        """
        模型前向传播函数
        :param characters: 字符输入，文本的字符级表示 [batch_size, text_len]
        :param mel_input: 梅尔谱输入，用于教师强制训练 [batch_size, mel_len, num_mels]
        :param pos_text: 文本位置索引 [batch_size, text_len]
        :param pos_mel: 梅尔谱位置索引 [batch_size, mel_len]
        :return: 梅尔谱输出、后处理输出、注意力权重和停止标记
        """
        
        # 编码器前向传播，处理输入文本
        memory, c_mask, attns_enc = self.encoder.forward(characters, pos=pos_text)
        # 解码器前向传播，生成梅尔谱
        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder.forward(memory, mel_input, c_mask,
                                                                                             pos=pos_mel)

        # 返回所有输出
        # mel_output: 原始梅尔谱输出
        # postnet_output: 经过后处理网络的梅尔谱输出
        # attn_probs: 编码器-解码器注意力权重
        # stop_preds: 停止标记预测
        # attns_enc: 编码器自注意力权重
        # attns_dec: 解码器自注意力权重
        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec


class ModelPostNet(nn.Module):
    """
    CBHG Network（CBHG网络）
    将梅尔谱转换为线性谱的后置网络，使用CBHG架构增强频谱细节
    """
    def __init__(self):
        """
        初始化后置网络模型
        创建预投影层、CBHG模块和后投影层
        """
        super(ModelPostNet, self).__init__()
        self.pre_projection = Conv(hp.n_mels, hp.hidden_size)  # 预投影卷积层，将梅尔谱维度转换为隐藏维度
        self.cbhg = CBHG(hp.hidden_size)  # CBHG模块，包含卷积组、高速公路网络和双向GRU
        self.post_projection = Conv(hp.hidden_size, (hp.n_fft // 2) + 1)  # 后投影卷积层，将隐藏维度转换为线性谱维度

    def forward(self, mel):
        """
        后置网络前向传播函数
        :param mel: 梅尔谱输入 [batch_size, mel_len, num_mels]
        :return: 预测的线性谱 [batch_size, mel_len, num_freq]
        """
        
        mel = mel.transpose(1, 2)  # 转置维度以适应卷积网络 [B, n_mels, T] -> [B, T, n_mels]
        mel = self.pre_projection(mel)  # 应用预投影卷积 [B, T, n_mels] -> [B, T, hidden_size]
        mel = self.cbhg(mel).transpose(1, 2)  # 应用CBHG模块并转置维度 [B, T, hidden_size] -> [B, hidden_size, T]
        mag_pred = self.post_projection(mel).transpose(1, 2)  # 应用后投影卷积并转置维度 [B, hidden_size, T] -> [B, n_fft//2+1, T] -> [B, T, n_fft//2+1]

        return mag_pred  # 返回预测的线性谱，用于波形重建