# src/process_image.py (MindSpore Version - Complete)

import numpy as np
import os 
from mindspore.common.tensor import Tensor
import mindspore.numpy as ms_np
from mindspore import dtype as mstype
import cv2 

# ImageNet pre-trained normalization constants (BGR convention)
MEAN = np.array([103.939, 116.779, 123.68])  # BGR

# src/process_image.py 文件中的 load_image 函数

def load_image(image_path, target_size=None):
    """Load image using OpenCV, convert to BGR, and resize."""
    import cv2
    import numpy as np 
    
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR) # 尝试以彩色模式 (3 通道) 读取

    if img is None:
        # 如果 imdecode 仍然失败，则抛出错误
        raise RuntimeError(f"Error: Could not decode/load image from path: {image_path}. The file may be corrupt or non-standard.")

    # 🌟 修正 2：检查并强制转换为 3 通道
    # VGG19 要求输入为 3 通道
    if img.ndim == 2:
        # 如果是 2 维数组（标准灰度图），强制转换为 3 通道 BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        # 如果是 4 通道 (包含 Alpha 通道)，去除 Alpha 通道
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[-1] != 3:
        # 捕获其他非 3 通道的异常情况
        raise RuntimeError(f"Error: Image {image_path} has {img.shape[-1]} channels, but 3 channels are required.")

    # 后续的 resize 逻辑保持不变
    if target_size is not None:
        # target_size 预期是 (H, W)
        h, w = target_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return img # 返回 HWC, BGR 的 NumPy 数组

def tensor_convert(image_np):
    """Converts a BGR NumPy image (HWC) to a MindSpore Tensor (NCHW) and normalizes."""
    
    # Convert HWC BGR to NCHW
    image_tensor = ms_np.transpose(Tensor(image_np, mstype.float32), (2, 0, 1))
    image_tensor = ms_np.expand_dims(image_tensor, 0)
    
    # Normalization (subtraction of mean)
    # Reshape mean to (3, 1, 1) for broadcasting
    mean_tensor = ms_np.reshape(Tensor(MEAN, mstype.float32), (3, 1, 1))
    
    # Apply subtraction
    normalized_tensor = image_tensor - mean_tensor
    
    return normalized_tensor

def image_convert(tensor):
    """Converts a MindSpore Tensor (NCHW) back to a NumPy image (HWC BGR)."""
    
    # Denormalization (add mean back)
    mean_tensor = ms_np.reshape(Tensor(MEAN, mstype.float32), (3, 1, 1))
    denormalized_tensor = tensor + mean_tensor
    
    # Convert NCHW to HWC
    img_np = ms_np.transpose(denormalized_tensor.asnumpy(), (0, 2, 3, 1))[0]
    
    # Clip values to 0-255 and cast to unsigned 8-bit integer
    img_np = np.clip(img_np, 0, 255).astype('uint8')
    
    return img_np

def get_image_name_ext(img_path):
    """Get name and extension of the image file from its path."""
    return os.path.splitext(os.path.basename(img_path))[0], os.path.splitext(os.path.basename(img_path))[1][1:]