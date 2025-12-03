import cv2
import os

image_path = "./anime_dataset/63961.jpg"

try:
    # 尝试加载图片
    img = cv2.imread(image_path)
    # 检查是否加载成功
    if img is None:
        print(f"❌ 验证失败: {image_path} 加载失败 (cv2.imread 返回 None).")
        # 尝试删除它
        os.remove(image_path)
        print("✅ 文件已删除，请重新运行训练。")
    elif img.ndim != 3:
        print(f"❌ 验证失败: {image_path} 维度为 {img.ndim} (非 3 通道).")
        # 尝试删除它
        os.remove(image_path)
        print("✅ 文件已删除，请重新运行训练。")
    else:
        print(f"✅ 验证成功: {image_path} 可以正常加载。")

except Exception as e:
    print(f"❌ 验证失败: 无法处理文件 {image_path}. 错误: {e}")
    # 尝试删除它
    os.remove(image_path)
    print("✅ 文件已删除，请重新运行训练。")