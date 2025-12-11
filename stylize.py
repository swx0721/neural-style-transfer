# stylize.py - 最终极致还原版 (禁用自适应推理和所有后处理)
import mindspore as ms
from mindspore import Tensor, context, ops
import transformer
import utils
import os
import time
from transformer import TransformerNet # 从 transformer.py 导入
import cv2
import numpy as np

# ------------------ GLOBAL SETTINGS ------------------
STYLE_TRANSFORM_PATH = "models/checkpoint_29250.ckpt" # 请替换为您的路径
PRESERVE_COLOR = False # 强制关闭色彩迁移
target_device = "CPU"
OUTPUT_DIR = "images/results/"

# MindSpore CPU 设置
context.set_context(mode=context.PYNATIVE_MODE, device_target=target_device)

# ------------------ 单图风格迁移 ------------------
def stylize():
    global STYLE_TRANSFORM_PATH
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 加载网络 (保持不变)
    while True:
        try:
            net = TransformerNet()
            # 检查模型文件是否存在
            if not os.path.exists(STYLE_TRANSFORM_PATH):
                 print(f"❌ 模型文件未找到: {STYLE_TRANSFORM_PATH}")
                 STYLE_TRANSFORM_PATH = input("请输入正确的 checkpoint 路径：").strip()
                 continue
                 
            param_dict = ms.load_checkpoint(STYLE_TRANSFORM_PATH)
            ms.load_param_into_net(net, param_dict)
            net.set_train(False)
            print("✅ Transformer Network Loaded Successfully. (Device: CPU)\n")
            break
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            STYLE_TRANSFORM_PATH = input("请输入正确的 checkpoint 路径：").strip()
            continue

    # 2. 推理循环
    while True:
        try:
            print("\n🎨 Stylize Image~ 输入 Ctrl+C 退出程序")
            content_image_path = input("请输入内容图像路径： ").strip()
            if content_image_path == "" or not os.path.isfile(content_image_path):
                print("⚠ 无效路径，请重新输入。")
                continue

            content_image = utils.load_image(content_image_path)
            if content_image is None:
                print("❌ 图像加载失败，请检查格式（支持jpg/png）。")
                continue

            starttime = time.time()
            # h, w = content_image.shape[:2] # 移除自适应推理的尺寸获取

            # ****************** 核心修正：使用纯净的推理步骤 ******************
            # 1. 图像 -> Tensor (使用 utils.itot，它默认会缩放和填充到训练时的 256x256 或 512x512)
            content_tensor = utils.itot(content_image, max_size=None) # max_size=None 使用图像原始尺寸或 utils 内默认缩放

            # 2. 网络推理
            generated_tensor = net(content_tensor)

            # 3. Tensor -> Image (使用 utils.ttoi，与 train.py 采样逻辑完全一致)
            generated_image = utils.ttoi(generated_tensor)
            
            # 4. 移除所有后处理 (PRESERVE_COLOR 确保 transfer_color 不运行)
            if PRESERVE_COLOR: 
                generated_image = utils.transfer_color(content_image, generated_image)
            # **************************************************

            output_filename = "styled_pure_" + os.path.basename(content_image_path) # 更改文件名以区分
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            utils.saveimg(generated_image, output_path)

            print(f"✅ 风格迁移完成，结果保存至: {output_path}")
            print(f"**注意：此图是模型原始输出，可能与原图分辨率不一致。**")
            print(f"⏱ 推理耗时: {time.time() - starttime:.2f} 秒\n")
            
        except KeyboardInterrupt:
            print("\n程序退出。")
            break
        except Exception as e:
            print(f"发生错误: {e}")

# ------------------ 文件夹批量风格迁移 (保留原自适应推理，但移除后处理) ------------------
# 注意：批量推理 stylize_folder 如果要复现纯净风格，也应该避免使用 cv2.resize
def stylize_folder(content_folder, save_folder=None, batch_size=1):
    if save_folder is None:
        save_folder = os.path.join(content_folder, "styled_results_cpu_pure")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    net = TransformerNet()
    param_dict = ms.load_checkpoint(STYLE_TRANSFORM_PATH)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    image_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [
        os.path.join(content_folder, f)
        for f in os.listdir(content_folder)
        if f.lower().endswith(image_ext)
    ]

    if not image_paths:
        print("⚠ 文件夹内未检测到图像文件")
        return

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        for img_path in batch_paths:
            content_image = utils.load_image(img_path)
            if content_image is None:
                print(f"❌ 跳过无效图像: {img_path}")
                continue
            
            # 批量推理中，我们必须决定是否继续使用自适应推理
            # 如果要极致还原，则需要修改 utils.infer_adaptive 或直接使用纯净模式
            
            # 这里我们假设用户希望批量模式下**不缩放/不拉伸**，只进行纯净推理。
            # 警告：这可能导致输入和输出图片尺寸不一致！
            
            content_tensor = utils.itot(content_image, max_size=None)
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor)
            
            if PRESERVE_COLOR: 
                generated_image = utils.transfer_color(content_image, generated_image)
            
            output_filename = "styled_pure_" + os.path.basename(img_path)
            output_path = os.path.join(save_folder, output_filename)
            utils.saveimg(generated_image, output_path)
            print(f"✅ 保存至: {output_path} (原始输出)")

if __name__ == '__main__':
    # 启用纯净模式
    stylize() 
    
    # 如果您需要批量处理，可以取消注释下面两行，注意输入和输出尺寸可能不一致
    # content_folder_path = input("请输入批量处理的文件夹路径：").strip()
    # stylize_folder(content_folder_path)