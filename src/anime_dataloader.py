# src/anime_dataloader.py (MindSpore DataLoader for Anime Avatars - 最终修正版)

import mindspore.dataset as ds
import os
import mindspore as ms
import numpy as np
import sys
# 🌟 修正 1：添加 mstype 的导入 (解决 NameError)
from mindspore import dtype as mstype 

# 假设 process_image.py 位于 src/ 目录下
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from process_image import load_image, tensor_convert # 导入 MindSpore 兼容的加载函数

class AnimeAvatarDataset:
    """
    MindSpore Dataset：加载目录中的所有图片，作为 Content Input。
    """
    def __init__(self, image_dir, image_size=(256, 256)):
        self.image_dir = image_dir
        self.image_size = image_size
        
        # 过滤并收集所有图片路径
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not self.image_paths:
            raise ValueError(f"No valid images found in directory: {image_dir}")
        
        print(f"Total images found for training: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # 1. 使用 load_image 加载和缩放图片
            image_np = load_image(image_path, target_size=self.image_size) 
            
            # 2. 转换为 MindSpore Tensor (NCHW, 已归一化)
            image_tensor = tensor_convert(image_np).squeeze(0) # 返回 (C, H, W) 格式
            
            return image_tensor

        except Exception as e:
            # 容错处理：如果加载失败，返回一个形状正确的全零张量
            print(f"⚠️ Warning: Failed to load {image_path}. Error: {e}. Returning zero tensor and continuing.")
            # 返回一个形状正确的 (3, H, W) 全零张量
            return ms.Tensor(np.zeros((3, self.image_size[0], self.image_size[1])).astype(np.float32))

# 辅助函数：创建 MindSpore DataLoader
def create_dataloader(image_dir, batch_size=4, image_size=(256, 256), num_workers=1): 
    """创建并配置 MindSpore DataLoader"""
    # 实例化自定义 Dataset
    dataset_generator = AnimeAvatarDataset(image_dir, image_size)
    
    # 🌟 修正 2/3/4：使用 GeneratorDataset，传入 mstype.float32 和 num_parallel_workers
    dataset = ds.GeneratorDataset( 
        source=dataset_generator, 
        column_names=["image"], 
        column_types=[mstype.float32], # 解决 NotImplementedError
        shuffle=True,
        # 解决 'GeneratorDataset' object has no attribute 'num_workers'
        num_parallel_workers=num_workers 
    )
    
    # 批量化
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset