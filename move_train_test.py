import os
import shutil
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy video files from train and test directories to a new structure.")
    parser.add_argument("--src_base", type=str, required=True)
    parser.add_argument("--dst_base", type=str, required=True)
    args = parser.parse_args()

    src_base = args.src_base
    dst_base = args.dst_base

    # 确保目标目录存在
    for split in ["train", "test"]:
        os.makedirs(os.path.join(dst_base, split), exist_ok=True)

    # 处理train和test文件夹
    for split in ["train", "test"]:
        src_dir = os.path.join(src_base, split)
        dst_dir = os.path.join(dst_base, split)
        
        if not os.path.exists(src_dir):
            continue
        # 遍历源目录中的所有用户文件夹
        for user_dir in os.listdir(src_dir):
            user_path = os.path.join(src_dir, user_dir)
            
            # 确保这是一个目录
            if os.path.isdir(user_path):
                # 遍历用户目录中的所有视频文件
                for video in os.listdir(user_path):
                    if video.endswith(".mp4"):
                        src_file = os.path.join(user_path, video)
                        dst_file = os.path.join(dst_dir, video)
                        
                        # 复制文件
                        shutil.copy2(src_file, dst_file)
                        print(f"已复制: {src_file} -> {dst_file}")

    print("复制过程已完成。")
