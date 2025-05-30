import os
import cv2
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
from fer import FER
from tqdm import tqdm
import argparse
import random
import torch
import multiprocessing as mp
import time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_video_details(video_path):
    try:
        import moviepy.editor as mpy
        clip = mpy.VideoFileClip(video_path)
        n_frames = clip.reader.nframes
        frame_rate = clip.fps
        clip.close()
        return int(n_frames), float(frame_rate)
    except Exception as e:
        print(f"[get_video_details] Error for {video_path}: {e}")
        return None, None

def extract_face_features_worker(video_path, frame_skip, device):
    detector = MTCNN(keep_all=True, device=device)
    emotion_detector = FER(mtcnn=False)
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    face_frames = 0
    face_counts = []
    face_areas = []
    close_up_count = 0
    emotion_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if total_frames % frame_skip != 0:
            total_frames += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(rgb_frame)

        if boxes is None or len(boxes) == 0:
            total_frames += 1
            continue

        face_frames += 1
        frame_area = frame.shape[0] * frame.shape[1]
        areas = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            area = w * h
            areas.append(area)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            emos = emotion_detector.detect_emotions(face_crop)
            if emos:
                dominant = max(emos[0]['emotions'], key=emos[0]['emotions'].get)
                emotion_results.append(dominant)

        face_counts.append(len(boxes))
        face_areas.extend([a / frame_area for a in areas])
        close_up_count += sum(1 for a in areas if a / frame_area > 0.3)
        total_frames += 1

    cap.release()

    if face_frames == 0:
        return {
            "has_face": 0,
            "face_frame_rate": 0.0,
            "avg_faces_per_frame": 0.0,
            "multi_person_ratio": 0.0,
            "avg_face_area_ratio": 0.0,
            "max_face_area_ratio": 0.0,
            "close_up_ratio": 0.0,
            "dominant_emotion": "none",
            "emotion_distribution": {}
        }

    vals, counts = np.unique(emotion_results, return_counts=True)
    dist = dict(zip(vals, (counts / counts.sum()).round(4)))

    return {
        "has_face": 1,
        "face_frame_rate": round(face_frames / (total_frames // frame_skip), 4),
        "avg_faces_per_frame": round(np.mean(face_counts), 4),
        "multi_person_ratio": round(sum(1 for c in face_counts if c > 1) / len(face_counts), 4),
        "avg_face_area_ratio": round(np.mean(face_areas), 4),
        "max_face_area_ratio": round(np.max(face_areas), 4),
        "close_up_ratio": round(close_up_count / len(face_counts), 4),
        "dominant_emotion": max(dist, key=dist.get, default="none"),
        "emotion_distribution": dist
    }

def worker_process(gpu_id, task_list, return_dict, progress_counter):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for idx, video_file, frame_skip in task_list:
        feats = extract_face_features_worker(video_file, frame_skip, device)
        return_dict[idx] = feats
        progress_counter.value += 1

def process_video_folder(folder_path, input_xlsx, output_csv, sample_per_second=1):
    info_csv = input_xlsx.replace('.xlsx', '_info.csv')
    if not os.path.exists(info_csv):
        df = pd.read_excel(input_xlsx, engine='openpyxl', keep_default_na=False)
        df['total_frames'] = np.nan
        df['frame_rate'] = np.nan
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="获取视频信息"):
            vid = row['vid']
            video_file = os.path.join(folder_path, f"{vid}.mp4")
            if not os.path.exists(video_file):
                print(f"[Info] 视频不存在，跳过: {video_file}")
                continue
            n_frames, fps = get_video_details(video_file)
            df.at[idx, 'total_frames'] = n_frames
            df.at[idx, 'frame_rate'] = fps
        df.to_csv(info_csv, index=False)
        print(f"[Done] 视频信息已保存到：{info_csv}")
    else:
        df = pd.read_csv(info_csv)
        print(f"[Info] 已加载视频信息：{info_csv}")

    face_keys = [
        "has_face", "face_frame_rate", "avg_faces_per_frame", "multi_person_ratio",
        "avg_face_area_ratio", "max_face_area_ratio", "close_up_ratio",
        "dominant_emotion", "emotion_distribution"
    ]
    for key in face_keys:
        df[key] = None

    task_list = []
    for idx, row in df.iterrows():
        vid = row['vid']
        video_file = os.path.join(folder_path, f"{vid}.mp4")
        if not os.path.exists(video_file):
            continue
        fps = row['frame_rate']
        frame_skip = max(int(fps // sample_per_second), 1) if fps and fps > 0 else 1
        task_list.append((idx, video_file, frame_skip))

    n_gpus = torch.cuda.device_count()
    print(f"[Info] 发现 GPU 数量: {n_gpus}")

    manager = mp.Manager()
    return_dict = manager.dict()
    progress_counter = manager.Value('i', 0)

    chunks = [task_list[i::n_gpus] for i in range(n_gpus)]
    processes = []
    for gpu_id in range(n_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, chunks[gpu_id], return_dict, progress_counter))
        p.start()
        processes.append(p)

    # 主进程进度条轮询
    pbar = tqdm(total=len(task_list), desc="处理进度")
    last_count = 0
    while any(p.is_alive() for p in processes):
        current = progress_counter.value
        pbar.update(current - last_count)
        last_count = current
        time.sleep(0.5)
    pbar.update(progress_counter.value - last_count)
    pbar.close()

    for p in processes:
        p.join()

    for idx, feats in return_dict.items():
        for k, v in feats.items():
            df.at[idx, k] = v if not isinstance(v, dict) else str(v)

    df.to_csv(output_csv, index=False)
    print(f"[Done] 人脸特征已保存到：{output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract face and emotion features from videos (multi-GPU multiprocessing).")
    parser.add_argument("--model", type=str, default="train", help="子目录名，通常 'train' 或 'test'")
    parser.add_argument("--xlsx", type=str, required=True, help="输入的 Excel 路径")
    parser.add_argument("--out_csv", type=str, required=True, help="输出 CSV 路径")
    parser.add_argument("--sample_per_second", type=int, default=1, help="每秒抽取多少帧")
    args = parser.parse_args()

    set_seed(42)
    mp.set_start_method('spawn')
    video_folder = f"../dataset/repair_video_train_test/{args.model}/"
    process_video_folder(
        folder_path=video_folder,
        input_xlsx=args.xlsx,
        output_csv=args.out_csv,
        sample_per_second=args.sample_per_second
    )
