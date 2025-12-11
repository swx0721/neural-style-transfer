import mindspore.nn as nn
import mindspore.ops as ops

class ConvLayer(nn.Cell):
    """卷积 + GroupNorm + ReLU/无激活"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pad_mode='pad'
        )
        self.in_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, eps=1e-5)
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        x = self.conv(x)
        x = self.in_norm(x)
        if self.relu:
            x = self.relu(x)
        return x

class ResidualBlock(nn.Cell):
    """残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)

    def construct(self, x):
        return x + self.conv2(self.conv1(x))

class UpsampleConvLayer(nn.Cell):
    """优化版上采样：仅在尺寸需要时插值，提升插值精度"""
    def __init__(self, in_channels, out_channels, kernel_size, upsample=None, high_res_mode=True):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.high_res_mode = high_res_mode  # 高分辨率模式使用双三次插值
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            pad_mode='pad'
        )
        self.in_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, eps=1e-5)

    def construct(self, x):
        if self.upsample:
            h, w = x.shape[2], x.shape[3]
            new_h, new_w = h * self.upsample, w * self.upsample
            # 高分辨率模式用双三次插值，普通模式用双线性插值
            mode = 'bicubic' if self.high_res_mode else 'bilinear'
            x = ops.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=True)
        x = self.conv(x)
        x = self.in_norm(x)
        return x

class TransformerNet(nn.Cell):
    """适配高分辨率的风格迁移网络"""
    def __init__(self, high_res_mode=True):
        super(TransformerNet, self).__init__()
        self.high_res_mode = high_res_mode
        
        # 下采样：高分辨率模式减少下采样次数，减少细节丢失
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        if high_res_mode:
            self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=1)  # 仅降维不降分辨率
            self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)  # 1次下采样
        else:
            self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
            self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)  # 2次下采样
        
        # 残差块：高分辨率模式增加数量，提升细节拟合能力
        res_blocks = 8 if high_res_mode else 5
        self.res_blocks = nn.SequentialCell(
            *[ResidualBlock(128) for _ in range(res_blocks)]
        )
        
        # 上采样：对应下采样次数调整
        if high_res_mode:
            self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, upsample=2, high_res_mode=high_res_mode)
            self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, upsample=1, high_res_mode=high_res_mode)  # 1次上采样
        else:
            self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, upsample=2, high_res_mode=high_res_mode)
            self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, upsample=2, high_res_mode=high_res_mode)  # 2次上采样
        
        self.conv_out = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4, pad_mode='pad')
        self.tanh = nn.Tanh()

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv_out(x)
        x = self.tanh(x)
        return x