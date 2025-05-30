from preprocess import get_post_dataset, DataLoader, collate_fn_postnet  # 导入后置网络数据集和数据加载器
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
    """后置网络训练的主函数"""
    # 获取后置网络数据集
    dataset = get_post_dataset()  # 获取用于训练后置网络的数据集
    global_step = 0  # 初始化全局步数
    
    # 初始化后置网络模型（使用DataParallel进行多GPU训练）
    m = nn.DataParallel(ModelPostNet().cuda())  # 创建后置网络模型并移至GPU

    # 设置训练模式
    m.train()  # 设置模型为训练模式
    # 初始化优化器
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)  # 使用Adam优化器

    # 初始化TensorBoard写入器
    writer = SummaryWriter()  # 创建TensorBoard日志写入器

    # 开始训练循环
    for epoch in range(hp.epochs):  # 遍历所有训练轮次
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_postnet, drop_last=True, num_workers=8)  # 创建数据加载器
        pbar = tqdm(dataloader)  # 创建进度条
        for i, data in enumerate(pbar):  # 遍历批次数据
            pbar.set_description("Processing at epoch %d"%epoch)  # 设置进度条描述
            global_step += 1  # 更新全局步数
            # 学习率调整（前400000步）
            if global_step < 400000:  # 在前400000步调整学习率
                adjust_learning_rate(optimizer, global_step)  # 调整学习率
                
            # 解包数据
            mel, mag = data  # 获取梅尔谱和线性谱数据
        
            # 将数据移至GPU
            mel = mel.cuda()  # 将梅尔谱数据移至GPU
            mag = mag.cuda()  # 将线性谱数据移至GPU
            
            # 前向传播
            mag_pred = m.forward(mel)  # 通过后置网络预测线性谱

            # 计算损失（L1损失）
            loss = nn.L1Loss()(mag_pred, mag)  # 计算预测线性谱与真实线性谱之间的L1损失
            
            # 记录训练损失到TensorBoard
            writer.add_scalars('training_loss',{  # 记录训练损失
                    'loss':loss,  # 线性谱预测损失

                }, global_step)
                    
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
                                os.path.join(hp.checkpoint_path,'checkpoint_postnet_%d.pth.tar' % global_step))  # 保存到指定路径


# 程序入口点
if __name__ == '__main__':
    main()  # 调用主函数开始训练后置网络