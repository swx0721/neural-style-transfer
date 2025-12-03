# fast_style_transfer_train.py (MindSpore 快速风格迁移训练脚本 - 最终完整修正版)

import mindspore as ms
# 🌟 修正 1：添加 load_param_into_net 导入
from mindspore import nn, ops, context, Tensor, save_checkpoint, load_checkpoint, load_param_into_net, Parameter 
from mindspore.nn import Adam
from mindspore import dtype as mstype
import numpy as np
import time
import yaml
import argparse
import sys
import os

# 确保 src/ 在路径中以便导入 StyleGenerator, VGG19, GramMatrix
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# 导入所有必要的组件
from src.train_model import StyleGenerator, VGG19, GramMatrix, NSTLoss
from src.anime_dataloader import create_dataloader 
from src.process_image import load_image, tensor_convert 

# VGG19 权重文件路径定义 (用于 TotalLossNet 中的 VGG19 实例化)
VGG19_CHECKPOINT_PATH = './vgg19-5104d1ea-910v2.ckpt' 

# --- Global MindSpore Context Setup ---
# 运行在 CPU，使用 PYNATIVE_MODE 避免 VGG19 编译问题
# ⚠️ 注意：CPU 训练非常慢，建议切换到 GPU 环境
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


class TotalLossNet(nn.Cell):
    """
    MindSpore Cell，用于计算训练 StyleGenerator 所需的 Content Loss 和 Style Loss。
    """
    def __init__(self, generator, avg_style_ckpt_path, config):
        super(TotalLossNet, self).__init__()
        self.generator = generator
        
        # 1. 实例化 VGG19 并加载权重
        self.vgg19 = VGG19(requires_grad=False)
        
        # 加载 VGG19 预训练权重
        if not os.path.exists(VGG19_CHECKPOINT_PATH):
            raise FileNotFoundError(f"Error: VGG19 checkpoint not found at {VGG19_CHECKPOINT_PATH}")
            
        print(f"[VGG19] Loading weights from: {VGG19_CHECKPOINT_PATH}")
        param_dict = load_checkpoint(VGG19_CHECKPOINT_PATH)
        
        # 🌟 关键修正 2：处理 VGG19 Checkpoint 键名中的 'vgg19.' 前缀
        new_param_dict = {}
        for name, param in param_dict.items():
            # 移除 MindSpore VGG19 Checkpoint 中常见的 'vgg19.' 前缀
            if name.startswith('vgg19.'):
                new_name = name[len('vgg19.'):]
            else:
                new_name = name
            new_param_dict[new_name] = param
            
        load_param_into_net(self.vgg19, new_param_dict)
        self.vgg19.set_train(False)
        print("[VGG19] VGG19 Weights loaded successfully.")
        
        # 2. 从 Checkpoint 加载平均 Gram 矩阵 (固定风格目标)
        print(f"[LossNet] Loading average Gram features from: {avg_style_ckpt_path}")
        if not os.path.exists(avg_style_ckpt_path):
            raise FileNotFoundError(f"Error: Average style checkpoint not found at {avg_style_ckpt_path}")
            
        param_dict = load_checkpoint(avg_style_ckpt_path)
        
        style_features = {}
        for param_name, param in param_dict.items():
            layer_name = param_name.replace('avg_gram_', '')
            style_features[layer_name] = param.data
            
        if not style_features:
            raise ValueError(f"Error: No style features loaded from {avg_style_ckpt_path}")
            
        print("[LossNet] Average Gram features loaded successfully.")
        
        # 3. 实例化损失网络
        
        # 🌟 关键修正 3：创建一个占位符 Content Feature 字典，用于通过 NSTLoss.__init__ 检查
        # 确保 Content Target 键 'relu4_1' 存在，避免 KeyError
        dummy_content_features = {
            'relu4_1': ms.Tensor(0.0, mstype.float32) 
        }

        self.nst_loss = NSTLoss(
            style_features=style_features,
            content_features=dummy_content_features, # 传入占位符
            style_weights=config['style_weights'],
            content_weight=config['alpha'],
            style_weight=config['beta']
        )
        
    def construct(self, content_tensor):
        # 1. 生成风格化图像
        generated_tensor = self.generator(content_tensor)
        
        # 2. 提取 Content 图像和 Generated 图像的 VGG19 特征
        content_features = self.vgg19(content_tensor)
        generated_features = self.vgg19(generated_tensor)
        
        # 3. 计算损失
        # MindSpore VGG19 的输出特征的键名是 'reluX_Y'
        # 🌟 这里的赋值会覆盖 NSTLoss.__init__ 中设置的虚拟值
        self.nst_loss.content_target = content_features['relu4_1']
        
        # 计算总损失
        total_loss = self.nst_loss(generated_features)
        
        return total_loss
        

def train_generator(config):
    """主训练函数"""
    
    # 1. 初始化模型
    generator = StyleGenerator()
    
    # 2. 初始化优化器
    optimizer = Adam(generator.trainable_params(), learning_rate=config['learning_rate'])
    
    # 3. 创建 DataLoader
    print(f"Loading data from: {config['content_image_dir']}")
    dataloader = create_dataloader(
        config['content_image_dir'], 
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        # 强制设置 num_workers=1
        num_workers=1 
    )

    total_batches = dataloader.get_dataset_size()
    print(f"Total batches per epoch: {total_batches}")
    
    # 4. 初始化损失网络和训练步骤
    # 在这里初始化 LossNet，它会加载 VGG19 和 Style Gram 特征
    loss_net = TotalLossNet(generator, config['avg_style_ckpt_path'], config)
    
    # 封装训练步骤
    train_step = nn.TrainOneStepCell(loss_net, optimizer)
    
    print("Starting training loop...")
    
    # 5. 训练循环
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start_time = time.time()
        
        # 重置损失统计
        running_loss = 0.0
        
        # 遍历 dataloader
        for batch_idx, data in enumerate(dataloader.create_tuple_iterator()):
            content_tensor = data[0]

            # 🌟 关键修正：确保这里没有 'if batch_idx >= 5: break' 这样的调试代码！
            
            # 执行一步训练
            loss = train_step(content_tensor)
            loss_value = loss.asnumpy().item()
            running_loss += loss_value

            # 日志打印
            if (batch_idx + 1) % config['log_interval'] == 0:
                print(f"Epoch: {epoch}/{config['num_epochs']}, Batch: {batch_idx + 1}/{total_batches}, Loss: {loss_value:.4f}")
                
        # 6. Epoch 结束
        avg_loss = running_loss / total_batches
        epoch_duration = time.time() - epoch_start_time
        print(f"\n--- Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s ---\n")
        
        # 7. 保存 Checkpoint
        if epoch % config['save_interval'] == 0:
            os.makedirs(config['output_dir'], exist_ok=True)
            ckpt_name = f"generator_epoch_{epoch}.ckpt"
            ckpt_path = os.path.join(config['output_dir'], ckpt_name)
            
            # 只保存 StyleGenerator 的参数
            save_checkpoint(generator, ckpt_path)
            print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MindSpore Fast Style Transfer Training")
    
    # 必需参数
    parser.add_argument("--content_image_dir", type=str, required=True, help="Directory containing the content images.")
    parser.add_argument("--avg_style_ckpt_path", type=str, required=True, help="Path to the pre-calculated average style checkpoint (.ckpt).")
    
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--train_config_path", type=str, required=True, help="Path to training configuration file (.yaml).")
    
    # 可选参数 (覆盖 YAML)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs (overrides YAML).")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (overrides YAML).")
    parser.add_argument("--log_interval", type=int, default=1, help="Log loss every N batches.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs.")
    
    args = parser.parse_args()
    
    # 1. 加载 YAML 配置
    try:
        # 修复：显式指定编码为 UTF-8
        with open(args.train_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {args.train_config_path}: {e}")
        print("Please ensure your YAML file is encoded in UTF-8.")
        sys.exit(1)
        
    # 2. 合并命令行参数并设置配置
    config['content_image_dir'] = args.content_image_dir
    config['avg_style_ckpt_path'] = args.avg_style_ckpt_path 
    config['output_dir'] = args.output_dir
    
    # 覆盖 YAML 中的 num_epochs/lr/log_interval/save_interval
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.log_interval is not None:
        config['log_interval'] = args.log_interval
    if args.save_interval is not None:
        config['save_interval'] = args.save_interval
        
    # 3. 开始训练
    train_generator(config)