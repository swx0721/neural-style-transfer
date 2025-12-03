# calculate_avg_style.py (计算平均风格特征脚本 - 修正版)

import mindspore as ms
from mindspore import nn, ops, context, save_checkpoint, load_checkpoint, Tensor
from mindspore import dtype as mstype
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm # 用于显示进度条 (pip install tqdm)

# 确保 src/ 在路径中
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# 导入 VGG19, GramMatrix, process_image 函数
from src.train_model import VGG19, GramMatrix 
from src.process_image import load_image, tensor_convert

# 🌟 关键修改 1：VGG19 权重文件路径定义 (使用您提供的新文件名)
VGG19_CHECKPOINT_PATH = './vgg19-5104d1ea-910v2.ckpt' 

# --- Global MindSpore Context Setup ---\r\n
# 关键：切换到 PYNATIVE_MODE 解决 VGG19 编译时的运行时错误
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=0) # 假设使用 CPU 模式


def calculate_avg_grams(image_dir, output_path, image_size=(256, 256)):
    """
    遍历指定目录下的所有图片，计算它们的平均 Gram 矩阵，并保存。
    """
    
    # 1. 初始化网络和组件
    # 🌟 关键修改 2：传入权重路径
    vgg19 = VGG19(requires_grad=False, checkpoint_path=VGG19_CHECKPOINT_PATH)
    vgg19.set_train(False) 
    gram_module = GramMatrix()
    
    # 2. 准备风格层名称和初始化累加器
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    avg_grams = {layer: None for layer in style_layers}
    image_count = 0
    
    print(f"Starting calculation of average Gram matrix from directory: {image_dir}")
    
    # 3. 遍历数据集
    # 确保 image_dir 是一个有效的目录
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return

    # 过滤文件，只保留图片
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Error: No image files found in the directory: {image_dir}")
        return
        
    for filename in tqdm(image_files):
        image_path = os.path.join(image_dir, filename)
        
        try:
            # 3.1 加载图像并转换为 Tensor (MindSpore)
            image_np = load_image(image_path, target_size=image_size) 
            
            if image_np is None:
                # load_image 函数返回 None，表示加载失败
                print(f"--- DEBUG INFO: load_image returned None for {image_path}. Skipping.")
                continue
                
            image_tensor = tensor_convert(image_np) 
            
            # 3.2 提取风格特征 (VGG19 模型在此处执行)
            features = vgg19(image_tensor)
            
            # 3.3 计算 Gram 矩阵并累加
            for layer in style_layers:
                gram = gram_module(features[layer])
                
                if avg_grams[layer] is None:
                    # 使用 asnumpy() 强制深拷贝并创建第一个 Tensor
                    avg_grams[layer] = ms.Tensor(gram.asnumpy(), mstype.float32)
                else:
                    # 累加
                    avg_grams[layer] += gram
                    
            image_count += 1
            
        except Exception as e:
            # 打印详细错误，这次不会再是 name 'features' is not defined 了（如果权重加载成功）
            print(f"❌ MindSpore processing FAILED for {image_path}. ERROR DETAILS: {e}")
            continue

    if image_count == 0:
        print("Error: No valid images found in the directory.")
        return

    # 4. 计算平均值并封装为 Checkpoint 字典
    param_list = []
    # 使用 ms.Tensor(image_count, ...) 来避免 MindSpore 的类型推断问题
    count_tensor = ms.Tensor(image_count, mstype.float32) 
    
    for layer in style_layers:
        if avg_grams[layer] is not None:
            # 除以总图像数得到平均值
            final_tensor = avg_grams[layer] / count_tensor
            # 使用 Parameter 封装以保存 Checkpoint
            param = ms.Parameter(final_tensor, name=f'avg_gram_{layer}')
            param_list.append(param)

    # 5. 保存结果
    save_checkpoint(param_list, output_path)
    print(f"\nSuccessfully processed {image_count} images.")
    print(f"Average style features saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Average Gram Matrix from a large dataset.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images to calculate average style from.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the resulting average style checkpoint (.ck).")
    
    args = parser.parse_args()
    
    # Run the calculation
    calculate_avg_grams(args.image_dir, args.output_path)