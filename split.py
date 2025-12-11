# split.py
# 用于从大型数据集中抽取小规模子集以供训练使用
import os
import random
import shutil

# ==================== 配置参数 ====================
# 1. 原始数据集目录 (包含 25000 张图片的目录)
# 请修改为您的绝对路径或相对路径
# 示例：如果您使用绝对路径，请修改为:
# SOURCE_DIR = "E:/task/学校任务/计算机视觉项目/fast-neural-style-mindspore-Ver/mscoco_miniVer"
SOURCE_DIR = "mscoco_miniVer"

# 2. 目标小规模数据集目录 (将被创建)
# 我们将使用这个目录作为 train.py 中的 DATASET_PATH
DEST_DIR = "mscoco_sampled_100"

# 3. 抽取图片的数量
SAMPLE_SIZE = 100

# 允许的图片扩展名 (统一小写)
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
# =================================================

def split_dataset(source_dir, dest_dir, sample_size):
    """
    从源目录中随机抽取指定数量的图片文件并复制到目标目录。
    """
    print(f"源目录: {source_dir}")
    
    # 1. 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录 '{source_dir}' 不存在。请检查路径是否正确。")
        return
        
    # 2. 获取所有符合条件的图片文件列表
    all_files = os.listdir(source_dir)
    image_files = [
        f for f in all_files 
        if os.path.isfile(os.path.join(source_dir, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    
    total_count = len(image_files)
    print(f"在源目录中找到 {total_count} 张图片。")

    if total_count < sample_size:
        print(f"警告: 找到的图片数量 ({total_count}) 少于目标抽取数量 ({sample_size})。将抽取所有图片。")
        sample_size = total_count

    # 3. 随机抽取文件
    sampled_files = random.sample(image_files, sample_size)
    print(f"已随机抽取 {sample_size} 张图片。")

    # 4. 创建目标目录
    os.makedirs(dest_dir, exist_ok=True)
    
    # 5. 复制文件
    print(f"开始复制文件到目标目录: {dest_dir} ...")
    
    for filename in sampled_files:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        try:
            # copy2 会尝试保留文件的元数据
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            print(f"复制文件 {filename} 时发生错误: {e}")
            
    print("✅ 复制完成!")
    print(f"现在您可以在 MindSpore 训练中使用目录 '{dest_dir}' 作为内容数据集路径。")

if __name__ == '__main__':
    # 为了保证随机性，设置随机种子
    random.seed(42) 
    
    split_dataset(SOURCE_DIR, DEST_DIR, SAMPLE_SIZE)