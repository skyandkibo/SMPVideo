import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import pickle
import argparse
from image_to_text_multi_threads import loading_model as loading_image_to_text_mode
from image_to_text_multi_threads import convert_image_to_text
from text_semantic_embedding import loading_model as loading_text_to_embbeding_model
from text_semantic_embedding import convert_text_to_embedding
from tqdm import tqdm

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

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)

    logger.addHandler(handler)
    # logger.addHandler(console)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, default='')
    parser.add_argument('--haszhen', action='store_true', help='是否使用帧数据集')
    parser.add_argument('--log_path', type=str, default='logs')
    args = parser.parse_args()

    os.makedirs(args.log_path, exist_ok=True)
    logger = get_logger(args.log_path, 'result')

    data_path = f"{args.pkl_path}"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    image_processor, image_model = loading_image_to_text_mode()
    angle_model = loading_text_to_embbeding_model()

    retrieval_features = []

    if args.haszhen:
        logger.info("使用帧数据集进行检索向量计算")
        image_base_path = "/sss/MyDrive/MMRA/datasets/tiktok/video_frames" # 图片所在的目录路径

        for item_id in tqdm(data["item_id"]): # 遍历每个视频 ID
            frame_texts = []
            for j in range(0, 10):
                image_path = os.path.join(image_base_path, f"{item_id}_{j}.jpg")
                if os.path.exists(image_path):
                    try:
                        text = convert_image_to_text(image_processor, image_model, image_path)
                    except Exception as e:
                        logger.info(f"{item_id}_{j}:图片异常，{e}")
                        continue
                else:
                    logger.info(f"{item_id}_{j}:图片不存在")
                    continue
                frame_texts .append(text)
            # 合成视频标题（原始标题 + 帧字幕）
            video_title = data.loc[data['item_id'] == item_id, 'text'].values[0]  # 获取原始视频标题
            combined_text = video_title+ " "+ " ".join(frame_texts)   # 拼接原始标题和视频帧标题

            # 获取检索向量
            retrieval_vector = convert_text_to_embedding(angle_model, combined_text)

            # 添加检索向量到列表
            retrieval_features.append(retrieval_vector[0].tolist())  # 取出numpy数组并转换为列表
    else:
        logger.info("使用视频标题进行检索向量计算")
        for item_id in tqdm(data["item_id"]): # 遍历每个视频 ID
            video_title = data.loc[data['item_id'] == item_id, 'text'].values[0]  # 获取原始视频标题
            # 获取检索向量
            retrieval_vector = convert_text_to_embedding(angle_model, video_title)
            # 添加检索向量到列表
            retrieval_features.append(retrieval_vector[0].tolist())  # 取出numpy数组并转换为列表

    data["retrieval_feature"] = retrieval_features
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    print(f"数据更新完成，检索向量已添加到 {data_path}!")

