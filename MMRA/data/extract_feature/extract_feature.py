import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import pickle
import numpy as np
from tqdm import tqdm
from textual_engineering import load_angle_bert_model, angle_bert_textual_feature_extraction
from visual_engineering import load_vit_model, vit_visual_feature_extraction
import argparse
import pandas as pd

import time
import logging

def get_logger(path, suffix, timebool=True):
    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S',time.localtime(time.time()))
    logger = logging.getLogger(__name__+cur_time)
    logger.setLevel(level = logging.INFO)
    if timebool:
        handler = logging.FileHandler(os.path.join(path, f"{suffix}_{cur_time}.log"))
    else:
        handler = logging.FileHandler(os.path.join(path, f"{suffix}.log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def extract_text_feature(data):

    angle_model = load_angle_bert_model()

    textual_features = []
    for i, text in enumerate(data['text']): # 替换为实际的文本列名
        if text=="" or text is None: # 检查文本是否为空
            logger.info(f"{data['item_id'][i]}:文本为空")
        textual_features.append(angle_bert_textual_feature_extraction(angle_model, text))

    data["textual_feature_embedding"] = textual_features
    return data

def extract_video_feature(data):

    vit_processor, vit_model = load_vit_model()

    visual_features_cls = []
    visual_features_mean = []
    image_base_path = "./datasets/tiktok/video_frames" # 图片所在的目录路径

    zero_vector = [0.0] * 768 # 假设视觉特征维度为 768

    for video_id in tqdm(data["item_id"]): # 遍历视频 ID 列
        video_features = []
        for i in range(10): # 每个视频有 10 张图片
            image_path = f"{image_base_path}/{video_id}_{i}.jpg" # 构建图片路径
            if os.path.exists(image_path): # 检查图片文件是否存在
                try:
                    feature = vit_visual_feature_extraction(vit_processor, vit_model, image_path)
                    if feature is not None: # 检查是否成功提取特征
                        video_features.append(feature)
                    else:
                        logger.info(f"Feature extraction failed for image {image_path}.")
                        video_features.append(zero_vector) # 无效特征填充零向量
                except Exception as e:
                    logger.info(f"Error processing image {image_path}: {e}")
                    video_features.append(zero_vector) # 如果出现错误，填充零向量
            else:
                logger.info(f"Image {image_path} does not exist.")
                video_features.append(zero_vector) # 如果图片不存在，填充零向量

        # 确保每个视频都有 10 个特征
        while len(video_features) < 10:
            video_features.append(zero_vector)
            logger.info(f"Padding zero vector for video {video_id} to ensure 10 features.")

        # 将 10 张图片的视觉特征保存到 visual_feature_embedding_cls 列
        visual_features_cls.append(video_features)
        # 计算每个视频的平均视觉特征，并保存到 visual_feature_embedding_mean 列
        avg_visual_features = np.mean(video_features, axis=0).tolist()
        visual_features_mean.append(avg_visual_features)

    data["visual_feature_embedding_cls"] = visual_features_cls
    data["visual_feature_embedding_mean"] = visual_features_mean

    return data

def copy_feature(data, copy_data, copy_list):
    for feature_name in copy_list:
        if feature_name not in data.columns:
            data[feature_name] = [None] * len(data)  # 初始化列，填充默认值

    for data_index, video_id in enumerate(data['item_id']):
        bool_find = False
        for copy_index, copy_video_id in enumerate(copy_data['item_id']):
            if copy_video_id == video_id:
                bool_find = True
                break
        if bool_find == False:
            print(f"Error {video_id}")
        for copy_feature in copy_list:
            data.at[data_index, copy_feature] = copy_data[copy_feature][copy_index]
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, default='')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--part', type=str, default='all')
    parser.add_argument('--method', type=str, default='produce')
    args = parser.parse_args()

    if args.part not in ['text', 'images', 'all']:
        print(f"Error {args.part}!")
        exit

    os.makedirs(args.log_path, exist_ok=True)
    logger = get_logger(args.log_path, 'result')

    data_path = f"{args.pkl_path}"
    with open(data_path,'rb') as f:
        data = pickle.load(f)

    if args.part == 'text' or args.part == 'all':
        if args.method == 'produce':
            data = extract_text_feature(data)
        elif args.method == 'copy':
            copy_data_path = input("输入待复制的pkl的地址")
            with open(copy_data_path,'rb') as f:
                copy_data = pickle.load(f)
            data = copy_feature(data, copy_data, ['textual_feature_embedding'])

    if args.part == 'images' or args.part == 'all':
        if args.method == 'produce':
            data = extract_video_feature(data)
        elif args.method == 'copy':
            copy_data_path = input("输入待复制的pkl的地址")
            with open(copy_data_path,'rb') as f:
                copy_data = pickle.load(f)
            data = copy_feature(data, copy_data, ['visual_feature_embedding_cls', 'visual_feature_embedding_mean'])

    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    print("特征提取完成，数据已保存！")
    logger.info("特征提取完成，数据已保存！")

