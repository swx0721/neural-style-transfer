from mindcv.models import vgg19
import mindspore.nn as nn
from mindspore import Tensor, ops, dtype as mstype
import numpy as np

class VGG19_Feature(nn.Cell):
    """
    自动下载 MindCV 版 VGG19 预训练模型，统一特征提取流程
    """
    def __init__(self):
        super().__init__()
        
        # 加载预训练VGG19
        self.vgg = vgg19(pretrained=True)
        self.vgg_layers = self.vgg.features
        for p in self.vgg_layers.get_parameters():
            p.requires_grad = False  # 冻结VGG权重
        
        # BGR均值（适配VGG预训练的输入规范）
        mean_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 3, 1, 1)
        self.mean_tensor = Tensor(mean_bgr, mstype.float32)
        
        # VGG19关键特征层
        self.layer_names = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '17': 'relu3_4',
            '26': 'relu4_4',
            '35': 'relu5_4'
        }
        self.cat = ops.Concat(1)

    def construct(self, x):
        """
        VGG 特征提取
        输入 x: RGB Tensor [-1,1], shape (B,C,H,W)
        输出: dict, key 为 VGG 特征层名称, value 为对应特征
        """
        # 将 [-1,1] 映射到 [0,255]，符合预训练 VGG 输入规范
        x_scaled = (x + 1.0) * 127.5  # [-1,1] -> [0,255]

        # BGR 顺序
        x_bgr = self.cat([x_scaled[:, 2:3, :, :], x_scaled[:, 1:2, :, :], x_scaled[:, 0:1, :, :]])

        # 减去 VGG 均值
        out = x_bgr - self.mean_tensor

        features = {}
        for i, layer in enumerate(self.vgg_layers):
            out = layer(out)
            if str(i) in self.layer_names:
                features[self.layer_names[str(i)]] = out
            if i == 35:  # 只提取到 relu5_4
                break

        return features
