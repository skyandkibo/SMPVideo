import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import multiprocessing as mp
import os
import sys

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(current_script_path))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from feature_engineering.classfy_content import build_cluster_features
from feature_engineering.feature_engineering import calculate_popularity_score

def process_fold(fold, train_idx, valid_idx, data, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[Fold {fold}] 使用 GPU {gpu_id}")
    train_df = data.iloc[train_idx].reset_index(drop=True)
    valid_df = data.iloc[valid_idx].reset_index(drop=True)

    train_df, valid_df = build_cluster_features(
        train_df, valid_df,
        post_content_col='post_content',
        suggested_col='post_suggested_words',
        seed=42,
        min_cluster_size=2
    )

    # train_df, valid_df = calculate_popularity_score(train_df, valid_df, ['main_cluster'])

    fold_dir = f"./kfold/fold_{fold}"
    os.makedirs(fold_dir, exist_ok=True)
    train_df.to_csv(os.path.join(fold_dir, 'training_data.csv'), index=False)
    valid_df.to_csv(os.path.join(fold_dir, 'validing_data.csv'), index=False)

    print(f"[Fold {fold}] 处理完成。")

if __name__ == "__main__":
    # 🔧 必须添加这句来支持 CUDA + 多进程
    mp.set_start_method('spawn', force=True)

    data = pd.read_csv('./output/merge_train_data.csv', keep_default_na=False)
    columns_to_remove = ['cluster_total_count', 'main_cluster', 'main_cluster_count', 'main_cluster_freq']
    data.drop(columns=[col for col in columns_to_remove if col in data.columns], inplace=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(kf.split(data))
    gpu_ids = [0]*5

    processes = []
    for fold, (train_idx, valid_idx) in enumerate(fold_indices):
        p = mp.Process(target=process_fold, args=(fold, train_idx, valid_idx, data, gpu_ids[fold]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[Done] 所有折并行处理完毕。")