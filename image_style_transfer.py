# image_style_transfer.py (MindSpore CPU Version - FINAL - 快速推理版)

import mindspore as ms
from mindspore import nn, ops, context, Tensor
from mindspore.nn import Adam
from mindspore import dtype as mstype
import numpy as np
import time
import yaml
import argparse
import sys
import os
import cv2 
from mindspore import load_checkpoint, load_param_into_net

# 假设 src/ 在路径中
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# 导入所有必要的 MindSpore 兼容函数
# 注意：我们现在需要导入 StyleGenerator
from src.train_model import StyleGenerator 
from src.process_image import load_image, tensor_convert, image_convert, get_image_name_ext


# --- Global MindSpore Context Setup ---
# 推理时使用 GRAPH_MODE 也可以，但 PYNATIVE_MODE 更灵活
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


def infer_image(content_tensor, checkpoint_path):
    """
    Performs fast style transfer inference using a pre-trained StyleGenerator model.
    """
    
    # 1. 实例化 StyleGenerator 模型
    generator = StyleGenerator()
    
    # 2. 加载预训练的权重文件
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Error: Model checkpoint file not found at {checkpoint_path}")
        
    print(f"Loading pre-trained model from {checkpoint_path}...")
    
    # 使用 MindSpore 的 API 加载参数
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(generator, param_dict)
    
    # 3. 设置为推理模式 (关闭 Dropout/BatchNorm 的训练行为)
    generator.set_train(False)

    # 4. 执行单次前向传播 (即风格迁移)
    print("Performing fast style transfer inference...")
    
    # 确保输入在 CPU 上
    if content_tensor.context.device_target != 'CPU':
        content_tensor = ops.Cast()(content_tensor, mstype.float32)

    styled_tensor = generator(content_tensor)
    
    # StyleGenerator 的输出是 tanh 激活后的，需要 MindSpore 内部处理，
    # 这里的 styled_tensor 已经包含了风格信息。
    return styled_tensor


def image_style_transfer(config):
    """Implements neural style transfer on a content image using a style image, applying provided configuration."""
    
    # --- 路径处理逻辑 ---
    output_dir = config.get('output_dir')
    content_path = config.get('content_filepath')
    checkpoint_path = config.get('checkpoint_path')
    
    # 检查是否提供了预训练模型路径
    if not checkpoint_path:
        print("Error: Running in Fast Mode requires --checkpoint_path to a pre-trained StyleGenerator model.")
        print("Please train the StyleGenerator first, or provide a path to a ready-made model.")
        return

    verbose = not config.get('quiet')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")

    # Load content image
    # Note: style_filepath is no longer needed for inference, but load_image needs target_size
    content_np = load_image(content_path) 

    # Convert to MindSpore Tensor
    content_tensor = tensor_convert(content_np)

    # Run the MindSpore inference
    start_time = time.time()
    result_tensor = infer_image(content_tensor, checkpoint_path)
    end_time = time.time()
    
    print(f"\nStyle Transfer completed in {end_time - start_time:.4f} seconds.")

    # Convert result tensor back to image and save
    result_image = image_convert(result_tensor)
    
    # 确定输出文件名
    name, ext = get_image_name_ext(content_path) 
    # 推荐输出 jpg
    output_filename = f"{name}_styled_fast.jpg" 
    output_path = os.path.join(output_dir, output_filename)
    
    cv2.imwrite(output_path, result_image)
    print(f"Output saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Style Transfer (MindSpore Fast Inference)", formatter_class=argparse.RawTextHelpFormatter)
    
    # 关键参数：预训练模型路径
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the pre-trained StyleGenerator model checkpoint (.ckpt).")
    
    # 图像路径参数
    parser.add_argument("--content_filepath", type=str, required=True, help="Path to the content image.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory that stores the output image.")
    
    # 可选参数
    parser.add_argument("--output_image_size", nargs="+", type=int, help="Size of the output image. Will use the dimensions of content image if not provided.")
    # 训练配置路径和 style_filepath 不再需要，但如果原始脚本有，可以保留
    parser.add_argument("--style_filepath", type=str, help=argparse.SUPPRESS) # 隐藏 style 路径，因为推理不再需要
    parser.add_argument("--train_config_path", type=str, help=argparse.SUPPRESS) 
    parser.add_argument("--quiet", type=bool, default=False, help="True stops showing debugging messages.")
    
    # 其它兼容性参数...
    
    args = parser.parse_args()
    
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    
    image_style_transfer(config)