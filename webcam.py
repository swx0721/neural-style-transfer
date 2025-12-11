# webcam_multiprocess.py (修复了卡顿和回退原图问题)

import cv2
import time
import numpy as np
import os
import multiprocessing
import sys
# 导入您的 MindSpore 模块
import mindspore as ms
from mindspore import context, Tensor
import transformer
import utils

# ------------------ GLOBAL SETTINGS ------------------
STYLE_TRANSFORM_PATH = "models/checkpoint_29250.ckpt"
PRESERVE_COLOR = False
WIDTH = 640 
HEIGHT = 480 
TRAIN_IMAGE_SIZE = 256
FPS_LIMIT = 30 

# ------------------ 辅助函数：MindSpore 推理工作进程 (保持不变) ------------------

def inference_worker(in_queue, out_queue, style_path, train_size):
    """ 专门运行 MindSpore 推理的子进程 """
    
    # 1. 进程内 MindSpore 初始化
    try:
        # 使用 PYNATIVE_MODE 最小化开销
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        
        net = transformer.TransformerNet()
        param_dict = ms.load_checkpoint(style_path)
        ms.load_param_into_net(net, param_dict)
        net.set_train(False)
        
        # 2. 预热 (Warm-up)
        dummy_image = np.zeros((train_size, train_size, 3), dtype=np.uint8)
        dummy_tensor = utils.itot(dummy_image, max_size=None)
        _ = net(dummy_tensor)
        print("✅ [Worker] MindSpore 网络初始化及预热完成。")
        
    except Exception as e:
        print(f"❌ [Worker] 初始化失败: {e}")
        return

    # 3. 推理循环
    while True:
        try:
            # 从输入队列获取原始帧 (阻塞式等待新帧)
            frame_raw = in_queue.get()
            if frame_raw is None: # 结束信号
                break

            # ********** 纯净推理流程 **********
            content_tensor = utils.itot(frame_raw, max_size=None)
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor)
            
            if PRESERVE_COLOR:
                generated_image = utils.transfer_color(frame_raw, generated_image)
            # **********************************

            # 将风格化后的结果放入输出队列 (非阻塞，如果队列满则丢弃，确保低延迟)
            if out_queue.full():
                 out_queue.get_nowait() # 丢弃旧结果
            out_queue.put(generated_image)

        except Exception as e:
            print(f"❌ [Worker] 推理发生错误: {e}")
            break

# ------------------ 主函数：I/O 和显示 (已修复) ------------------

def webcam_multiprocess(style_transform_path, width=WIDTH, height=HEIGHT):
    
    if not os.path.exists(style_transform_path):
        print(f"❌ 模型文件未找到: {style_transform_path}")
        return
        
    # 1. 设置队列
    input_queue = multiprocessing.Queue(maxsize=1)  # 只需要存储最新的帧
    output_queue = multiprocessing.Queue(maxsize=1) # 只需要存储最新的风格化结果 (用于缓冲)

    # 2. 启动推理工作进程
    worker = multiprocessing.Process(
        target=inference_worker, 
        args=(input_queue, output_queue, style_transform_path, TRAIN_IMAGE_SIZE)
    )
    worker.daemon = True
    worker.start()
    
    # 3. 设置摄像头
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        worker.terminate()
        return
        
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    print("✅ [Main] 摄像头启动。")

    # 4. 修复：等待首次风格化完成，消除初始化等待时间
    last_styled_image = None
    print("[Main] ⏳ 正在等待首次风格化完成 (包含模型初始化和预热)...")

    # 在等待期间，不断捕获帧并发送给推理进程
    while last_styled_image is None:
        ret_val, frame = cam.read()
        if not ret_val: break
        
        # 镜像
        display_frame = cv2.flip(frame, 1)

        # 发送帧进行处理 (如果队列未满)
        if not input_queue.full():
            input_queue.put(frame)
        
        # 尝试获取结果
        if not output_queue.empty():
            last_styled_image = output_queue.get()
            break # 拿到结果，退出等待循环

        # 显示等待提示
        cv2.putText(display_frame, "INITIALIZING (Wait 2-3s)...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.imshow('Real-time Multiprocess Style Transfer', display_frame)
        if cv2.waitKey(1) == ord('q'):
             # 在等待期间如果用户按下退出键
             input_queue.put(None) 
             worker.join()
             cam.release()
             cv2.destroyAllWindows()
             return

    # 5. 主循环：捕获、发送、接收、显示
    print("✅ [Main] 初始化完成，开始实时显示。")
    while True:
        start_time = time.time()
        
        # 捕获帧
        ret_val, frame = cam.read()
        if not ret_val:
            print("[Main] Failed to grab frame")
            break

        frame = cv2.flip(frame, 1) # 镜像
        
        # 将新帧放入输入队列 (只放最新的，不排队，保证低延迟)
        if not input_queue.full():
             input_queue.put(frame)
        
        # 接收结果：尝试从输出队列获取新结果，并更新 last_styled_image
        if not output_queue.empty():
            last_styled_image = output_queue.get()
            
        # 核心修正：始终显示最新的风格化结果 (利用 last_styled_image 缓冲)
        if last_styled_image is not None:
            # 调整回原始捕获分辨率显示
            display_image = cv2.resize(last_styled_image, (width, height), interpolation=cv2.INTER_LINEAR)
            
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            
            cv2.putText(display_image, f"FPS: {fps:.2f} (Multiprocess)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Real-time Multiprocess Style Transfer', display_image)
        else:
             # 如果初始化失败，显示原图并提示错误
             cv2.putText(frame, "Error: Worker Failed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             cv2.imshow('Real-time Multiprocess Style Transfer', frame)


        # 按 'q' 退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 6. 清理
    input_queue.put(None) # 发送终止信号给子进程
    worker.join(timeout=2)
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam_multiprocess(STYLE_TRANSFORM_PATH)