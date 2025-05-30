from preprocess import get_dataset, DataLoader, collate_fn_transformer  # 导入数据集和数据加载器
from network import *  # 导入网络模型
from tensorboardX import SummaryWriter  # 导入TensorBoard可视化工具
import torchvision.utils as vutils  # 导入视觉工具
import os  # 导入操作系统模块
from tqdm import tqdm  # 导入进度条

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    """调整学习率，实现预热和衰减策略
    Args:
        optimizer: 优化器
        step_num: 当前步数
        warmup_step: 预热步数，默认为4000
    """
    # 计算学习率，先预热后衰减
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    # 更新优化器中的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def main():
    """主训练函数"""
    # 获取数据集
    dataset = get_dataset()  # 获取LJSpeech数据集
    global_step = 0  # 初始化全局步数
    
    # 初始化模型（使用DataParallel进行多GPU训练）
    m = nn.DataParallel(Model().cuda())  # 创建模型并移至GPU

    # 设置训练模式
    m.train()  # 设置模型为训练模式
    # 初始化优化器
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)  # 使用Adam优化器

    # 初始化停止标记的正样本权重
    pos_weight = t.FloatTensor([5.]).cuda()  # 设置正样本权重为5
    # 初始化TensorBoard写入器
    writer = SummaryWriter()  # 创建TensorBoard日志写入器
    
    # 开始训练循环
    for epoch in range(hp.epochs):  # 遍历所有训练轮次
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)  # 创建数据加载器
        pbar = tqdm(dataloader)  # 创建进度条
        for i, data in enumerate(pbar):  # 遍历批次数据
            pbar.set_description("Processing at epoch %d"%epoch)  # 设置进度条描述
            global_step += 1  # 更新全局步数
            # 学习率调整（前400000步）
            if global_step < 400000:  # 在前400000步调整学习率
                adjust_learning_rate(optimizer, global_step)  # 调整学习率
                
            # 解包数据
            character, mel, mel_input, pos_text, pos_mel, _ = data  # 获取文本、梅尔谱、梅尔输入、位置编码等数据
            
            # 计算停止标记的目标值（填充位置为1，非填充位置为0）
            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)  # 计算停止标记
            
            # 将数据移至GPU
            character = character.cuda()  # 将文本数据移至GPU
            mel = mel.cuda()  # 将梅尔谱数据移至GPU
            mel_input = mel_input.cuda()  # 将梅尔输入数据移至GPU
            pos_text = pos_text.cuda()  # 将文本位置编码移至GPU
            pos_mel = pos_mel.cuda()  # 将梅尔位置编码移至GPU
            
            # 前向传播
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)  # 模型前向传播

            # 计算损失
            mel_loss = nn.L1Loss()(mel_pred, mel)  # 计算梅尔谱预测损失（L1损失）
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)  # 计算后处理梅尔谱预测损失（L1损失）
            
            # 总损失
            loss = mel_loss + post_mel_loss  # 计算总损失
            
            # 记录训练损失到TensorBoard
            writer.add_scalars('training_loss',{  # 记录训练损失
                    'mel_loss':mel_loss,  # 梅尔谱损失
                    'post_mel_loss':post_mel_loss,  # 后处理梅尔谱损失

                }, global_step)
                
            # 记录注意力机制的alpha参数到TensorBoard
            writer.add_scalars('alphas',{  # 记录注意力机制的alpha参数
                    'encoder_alpha':m.module.encoder.alpha.data,  # 编码器alpha参数
                    'decoder_alpha':m.module.decoder.alpha.data,  # 解码器alpha参数
                }, global_step)
            
            # 定期保存注意力图
            if global_step % hp.image_step == 1:  # 每隔image_step步保存一次注意力图
                # 保存编码器-解码器注意力图
                for i, prob in enumerate(attn_probs):  # 遍历注意力概率
                    
                    num_h = prob.size(0)  # 获取头数
                    for j in range(4):  # 为每个头保存注意力图
                
                        x = vutils.make_grid(prob[j*16] * 255)  # 创建注意力图网格
                        writer.add_image('Attention_%d_0'%global_step, x, i*4+j)  # 添加注意力图到TensorBoard
                
                # 保存编码器自注意力图
                for i, prob in enumerate(attns_enc):  # 遍历编码器自注意力概率
                    num_h = prob.size(0)  # 获取头数
                    
                    for j in range(4):  # 为每个头保存注意力图
                
                        x = vutils.make_grid(prob[j*16] * 255)  # 创建注意力图网格
                        writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)  # 添加注意力图到TensorBoard
            
                # 保存解码器自注意力图
                for i, prob in enumerate(attns_dec):  # 遍历解码器自注意力概率

                    num_h = prob.size(0)  # 获取头数
                    for j in range(4):  # 为每个头保存注意力图
                
                        x = vutils.make_grid(prob[j*16] * 255)  # 创建注意力图网格
                        writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)  # 添加注意力图到TensorBoard
                
            # 反向传播和优化
            optimizer.zero_grad()  # 清除梯度
            # 计算梯度
            loss.backward()  # 反向传播
            
            # 梯度裁剪（防止梯度爆炸）
            nn.utils.clip_grad_norm_(m.parameters(), 1.)  # 将梯度范数裁剪到1
            
            # 更新权重
            optimizer.step()  # 更新模型参数

            # 定期保存模型检查点
            if global_step % hp.save_step == 0:  # 每隔save_step步保存一次模型
                # 保存模型和优化器状态
                t.save({'model':m.state_dict(),  # 保存模型状态字典
                                 'optimizer':optimizer.state_dict()},  # 保存优化器状态字典
                                os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))  # 保存到指定路径


# 程序入口点
if __name__ == '__main__':
    main()  # 调用主函数开始训练