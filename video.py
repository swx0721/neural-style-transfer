# video.py (MindSpore CPU 最终修正版 - 包含预热)
import utils
import transformer
import cv2
import os
import time
import mindspore as ms
from mindspore import context, Tensor
import numpy as np

# ------------------ GLOBAL SETTINGS ------------------
# MindSpore CPU 环境设置
target_device = "CPU"
context.set_context(mode=context.GRAPH_MODE, device_target=target_device) # 视频批量处理使用GRAPH_MODE求最大吞吐量

VIDEO_NAME = "input_video.mp4"       # 待处理的视频文件名
FRAME_SAVE_PATH = "frames/temp/"     # 视频帧提取的临时目录
STYLE_FRAME_SAVE_PATH = "style_frames/output/" # 风格化帧的输出目录
STYLE_VIDEO_NAME = "styled_output.mp4" # 最终输出的视频文件名
STYLE_PATH = "models/checkpoint_29250.ckpt" # 您的预训练模型路径
PRESERVE_COLOR = False                   # 确保关闭色彩迁移

FRAME_BASE_FILE_NAME = "frame"
FRAME_BASE_FILE_TYPE = ".jpg"
TRAIN_IMAGE_SIZE = 256 # 匹配 train.py 中的训练尺寸

# ------------------ 辅助函数 (纯 Python/OpenCV) ------------------
def getInfo(video_path):
    # ... (getInfo 保持不变) ...
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return int(height), int(width), fps

def getFrames(video_path, frames_path=FRAME_SAVE_PATH):
    # ... (getFrames 保持不变) ...
    os.makedirs(frames_path, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(frames_path, f"{FRAME_BASE_FILE_NAME}{count:06d}{FRAME_BASE_FILE_TYPE}"), image)
        count += 1
    vidcap.release()
    print(f"✅ 提取了 {count} 帧.")

def createVideo(frames_path, save_name, fps, height, width):    
    # ... (createVideo 保持不变) ...
    base_name_len = len(FRAME_BASE_FILE_NAME)
    filetype_len = len(FRAME_BASE_FILE_TYPE)
    # 确保自然排序
    images = [img for img in sorted(os.listdir(frames_path), key=lambda x : int(x[base_name_len:-filetype_len])) if img.endswith(FRAME_BASE_FILE_TYPE)]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_name, fourcc, fps, (int(width), int(height)))
    
    for image in images:
        img_path = os.path.join(frames_path, image)
        frame = cv2.imread(img_path)
        if frame is not None:
            # 兼容性检查：确保帧尺寸匹配，否则需要重新resize
            if frame.shape[0] != height or frame.shape[1] != width:
                 frame = cv2.resize(frame, (width, height))
            out.write(frame)

    out.release()
    print(f"✅ 视频已保存至 {save_name}")

# ------------------ 核心推理逻辑 ------------------
def stylize_frames(net, frame_folder, output_folder):
    """ 对一个文件夹内的所有帧进行风格迁移 """
    os.makedirs(output_folder, exist_ok=True)
    image_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    # 使用 sorted 确保处理顺序
    image_paths = sorted([
        os.path.join(frame_folder, f)
        for f in os.listdir(frame_folder)
        if f.lower().endswith(image_ext)
    ])
    
    if not image_paths:
        print("⚠ 文件夹内未检测到图像文件，跳过风格化。")
        return

    total_frames = len(image_paths)
    start_time = time.time()
    
    # 预热已在主函数中完成

    for idx, img_path in enumerate(image_paths):
        content_image = utils.load_image(img_path)
        if content_image is None:
            print(f"❌ 跳过无效图像: {img_path}")
            continue

        # ********** 纯净推理流程 **********
        content_tensor = utils.itot(content_image, max_size=None)
        generated_tensor = net(content_tensor)
        generated_image = utils.ttoi(generated_tensor)
        
        # 保持纯净输出
        if PRESERVE_COLOR:
            generated_image = utils.transfer_color(content_image, generated_image)
        # **********************************

        output_filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, output_filename)
        utils.saveimg(generated_image, output_path)

        if idx % 100 == 0:
            avg_time = (time.time() - start_time) / (idx + 1)
            print(f"▶ 帧处理进度: {idx+1}/{total_frames}. 平均帧处理时间: {avg_time*1000:.2f} ms")

# ------------------ 主函数 ------------------
def video_transfer(video_path, style_path):
    starttime = time.time()
    
    # 1. 提取视频信息
    H_orig, W_orig, fps = getInfo(video_path)
    print(f"📼 视频信息: H={H_orig}, W={W_orig}, FPS={fps:.2f}")

    # 2. 帧提取
    print("⏳ 提取视频帧...")
    getFrames(video_path, frames_path=FRAME_SAVE_PATH)
    
    # 3. 加载网络
    net = transformer.TransformerNet()
    param_dict = ms.load_checkpoint(style_path)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)
    print("✅ Transformer Network Loaded.")
    
    # ********** 预热步骤 **********
    print(f"🔥 正在预热网络 (Warm-up)... (目标尺寸: {TRAIN_IMAGE_SIZE})")
    dummy_image = np.zeros((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3), dtype=np.uint8)
    dummy_tensor = utils.itot(dummy_image, max_size=None)
    _ = net(dummy_tensor)
    print("✅ 预热完成，开始风格化。")
    # ******************************

    # 4. 对帧进行风格迁移 (注意：输出帧尺寸为 TRAIN_IMAGE_SIZE x TRAIN_IMAGE_SIZE)
    print("🎨 正在对帧进行风格迁移...")
    stylize_frames(net, FRAME_SAVE_PATH, STYLE_FRAME_SAVE_PATH)
    
    # 5. 合成视频 (使用 TRAIN_IMAGE_SIZE 作为视频尺寸)
    print("🎬 正在合成视频...")
    createVideo(STYLE_FRAME_SAVE_PATH, STYLE_VIDEO_NAME, fps, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE)

    stop_time = time.time()
    print(f"\n✨ 视频风格迁移完成! 总耗时: {stop_time - starttime:.2f} 秒")

if __name__ == '__main__':
    if not os.path.exists(STYLE_PATH):
         print(f"❌ 模型文件未找到: {STYLE_PATH}")
         STYLE_PATH = input("请输入正确的 checkpoint 路径：").strip()
    video_transfer(VIDEO_NAME, STYLE_PATH)