# src/train_model.py (MindSpore Version - FINAL - 包含 StyleGenerator)

import mindspore as ms
# 🌟 关键修改 1：导入 MindSpore 权重加载所需函数
from mindspore import nn, ops, load_checkpoint, load_param_into_net
from mindspore.common.initializer import XavierUniform, Constant
from mindspore.common.parameter import Parameter
import mindspore.numpy as ms_np
import os # 🌟 关键修改 2：导入 os 库用于路径检查

# --- 1. VGG19 Feature Extractor ---
class VGG19(nn.Cell):
    """VGG19 for feature extraction (partial implementation for NST layers)."""
    # 🌟 关键修改 3：添加 checkpoint_path 参数
    def __init__(self, requires_grad=False, checkpoint_path=None):
        super(VGG19, self).__init__()
        
        # VGG19 blocks (only necessary layers for style/content features)
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu1_2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu2_2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu3_3 = nn.ReLU()
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu3_4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu4_3 = nn.ReLU()
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu4_4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu5_3 = nn.ReLU()
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.relu5_4 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        
        # ----------------------------------------------------
        # 🌟 关键修改 4：VGG19 权重加载逻辑
        # ----------------------------------------------------
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[VGG19] Loading weights from: {checkpoint_path}")
            try:
                param_dict = load_checkpoint(checkpoint_path)
                load_param_into_net(self, param_dict)
                print("[VGG19] VGG19 Weights loaded successfully.")
            except Exception as e:
                # 如果 MindSpore 版本不匹配或权重文件结构异常，会在此处报错
                print(f"[VGG19] ERROR loading weights: {e}")
                print("[VGG19] Proceeding with randomly initialized weights (EXPECT FAILURE in forward pass).")
        else:
            print("[VGG19] WARNING: Checkpoint path not provided or file not found. Using random weights.")
            
        if not requires_grad:
            for param in self.get_parameters():
                param.requires_grad = False

    def construct(self, x):
        features = {}
        
        # Block 1
        x = self.relu1_1(self.conv1_1(x))
        features['relu1_1'] = x
        x = self.relu1_2(self.conv1_2(x))
        x = self.maxpool1(x)

        # Block 2
        x = self.relu2_1(self.conv2_1(x))
        features['relu2_1'] = x
        x = self.relu2_2(self.conv2_2(x))
        x = self.maxpool2(x)

        # Block 3
        x = self.relu3_1(self.conv3_1(x))
        features['relu3_1'] = x
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.relu3_4(self.conv3_4(x))
        x = self.maxpool3(x)

        # Block 4
        x = self.relu4_1(self.conv4_1(x))
        features['relu4_1'] = x
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.relu4_4(self.conv4_4(x))
        x = self.maxpool4(x)

        # Block 5
        x = self.relu5_1(self.conv5_1(x))
        features['relu5_1'] = x
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.relu5_4(self.conv5_4(x))
        # maxpool5 后的特征通常不用于风格迁移
        
        return features

# --- 2. Style Transfer Network Components (StyleGenerator 的辅助类) ---
# ... (ConvLayer, ResidualBlock, ConvTransposeLayer 类的代码保持不变) ...

# 假设这部分 StyleGenerator 代码在原文件中存在
class ConvLayer(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.reflect_pad = nn.Pad(paddings=((0, 0), (0, 0), (kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode="REFLECT")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, pad_mode='valid', has_bias=True, weight_init=XavierUniform())
        self.instance_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels) # MindSpore 使用 GroupNorm 代替 InstanceNorm
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.reflect_pad(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Cell):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 确保卷积核大小为 3, padding=1，保持尺寸不变
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True, weight_init=XavierUniform())
        self.in1 = nn.GroupNorm(num_groups=channels, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True, weight_init=XavierUniform())
        self.in2 = nn.GroupNorm(num_groups=channels, num_channels=channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        identity = x
        
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        
        out += identity # 残差连接
        return out

class ConvTransposeLayer(nn.Cell):
    """用于上采样的反卷积层（ConvTranspose2d）"""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTransposeLayer, self).__init__()
        
        # 🌟 关键修改：当 pad_mode='same' 时，必须设置 padding=0 (或省略)
        # MindSpore 会自动计算所需填充。
        self.conv_transpose = nn.Conv2dTranspose(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0,                 # 必须设为 0
            pad_mode='same',           # 保持 'same' 模式，让 MindSpore 自动处理填充
            has_bias=True, 
            weight_init=XavierUniform()
        )
        self.instance_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.conv_transpose(x)
        x = self.instance_norm(x)
        x = self.relu(x)
        return x

# --- 3. StyleGenerator (快速风格迁移网络) ---
class StyleGenerator(nn.Cell):
    def __init__(self):
        super(StyleGenerator, self).__init__()
        
        # 编码器部分 (下采样)
        self.encode1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.encode2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.encode3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        
        # 残差块部分
        self.res_blocks = nn.SequentialCell([
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        ])
        
        # 解码器部分 (上采样)
        self.decode1 = ConvTransposeLayer(128, 64, kernel_size=3, stride=2)
        self.decode2 = ConvTransposeLayer(64, 32, kernel_size=3, stride=2)
        
        # 输出层 (不带 InstanceNorm 和 ReLU)
        self.output_padding = nn.Pad(paddings=((0, 0), (0, 0), (9//2, 9//2), (9//2, 9//2)), mode="REFLECT")
        self.conv_out = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=0, pad_mode='valid', has_bias=True)
        
        # MindSpore 的 tanh 激活函数
        self.tanh = ops.Tanh()


    def construct(self, x):
        # 编码
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        
        # 残差
        x = self.res_blocks(x)
        
        # 解码
        x = self.decode1(x)
        x = self.decode2(x)
        
        # 输出层
        x = self.output_padding(x)
        x = self.conv_out(x)
        
        # 将输出限制在 -1 到 1 附近，与 VGG19 的归一化范围保持一致
        return x # 实际上不需要 tanh，因为输入图像是在 MindSpore 归一化后的范围

# --- 4. Gram Matrix and Total Loss Net ---

class GramMatrix(nn.Cell):
    """Compute the Gram matrix of a feature map."""
    def __init__(self):
        super(GramMatrix, self).__init__()
        self.reshape = ops.Reshape()
        # 🌟 关键修正 1：改为 ops.BatchMatMul
        self.matmul = ops.BatchMatMul() 
        
        # 🌟 关键修正 2：使用更稳定的 ops.Transpose 
        # (transpose_a=False, transpose_b=True 意味着计算 A @ B^T)
        self.transpose_op = ops.Transpose()
        
        self.shape_op = ops.Shape()
        self.float_cast = ops.Cast()

    def construct(self, x):
        # x is (B, C, H, W)
        
        # 1. 获取形状
        shape = self.shape_op(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3] 
        
        # 2. Flatten H and W dimensions: (B, C, H*W)
        # 此时 features 是 A
        features = self.reshape(x, (B, C, H * W)) 
        
        # 3. 计算 features_T (B, H*W, C)，此时 features_T 是 B^T
        # B^T = Transpose(A) = Transpose(B, (0, 2, 1))
        features_T = self.transpose_op(features, (0, 2, 1))
        
        # 4. MatMul: (B, C, H*W) x (B, H*W, C) -> (B, C, C)
        # 由于我们使用 ops.BatchMatMul，它的输入要求是 (B, M, K) x (B, K, N)
        # 这里是 (B, C, H*W) x (B, H*W, C)，所以 M=C, K=H*W, N=C
        gram = self.matmul(features, features_T) 
        
        # 5. Normalization
        norm_factor = self.float_cast(ms.Tensor(C * H * W), ms.float32)
        gram = gram / norm_factor
        
        return gram


# --- Loss Function (Combined Content and Style Loss) ---

class NSTLoss(nn.Cell):
    """Calculates the total loss for Neural Style Transfer."""
    # 用于传统风格迁移（图像迭代优化）
    def __init__(self, style_features, content_features, style_weights, content_weight, style_weight):
        super(NSTLoss, self).__init__()
        self.style_features = style_features
        self.content_features = content_features
        self.style_weights = style_weights
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.gram = GramMatrix()
        # MindSpore 的 MSELoss 默认是 mean，这里改为 sum 以匹配原始逻辑或简单相加
        self.mean_square_error = nn.MSELoss(reduction='sum') 
        
        # 默认只在 relu4_1 上计算内容损失
        self.content_target = content_features['relu4_1']
        
    def construct(self, generated_features):
        # generated_features 是由 GradWrap 中的 VGG19 调用的结果（特征字典）
        
        # 1. Content Loss (通常只在 relu4_1 上计算)
        content_loss_value = self.mean_square_error(generated_features['relu4_1'], self.content_target)
        
        # 2. Style Loss
        style_loss_value = ms.Tensor(0.0, ms.float32)
        
        # 遍历所有风格层
        for name, weight in self.style_weights.items():
            # 提取 Generated 图像的 Gram 矩阵
            # 输入是 4D 特征图 (B, C, H, W)，输出是 2D Gram 矩阵 (B, C, C) 或 (C, C)
            generated_gram = self.gram(generated_features[name]) 
            
            # 提取 Style 目标 Gram 矩阵
            # 🌟 关键修正：self.style_features[name] 已经是预计算好的 Gram 矩阵 (2D)，直接使用。
            style_gram = self.style_features[name]
            
            # 计算该层的风格损失
            layer_style_loss = self.mean_square_error(generated_gram, style_gram)
            
            # 累加加权损失
            style_loss_value += layer_style_loss * weight
            
        # 3. Total Loss
        total_loss = self.content_weight * content_loss_value + self.style_weight * style_loss_value
        
        return total_loss