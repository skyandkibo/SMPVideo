import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
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

# 支持 CUDA + 多进程
mp.set_start_method('spawn', force=True)

def process_fold(fold, train_df, valid_df, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Fold {fold}] 使用 GPU {gpu_id}")

    # 构造聚类特征
    train_proc, valid_proc = build_cluster_features(
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        post_content_col='post_content',
        suggested_col='post_suggested_words',
        seed=42,
        min_cluster_size=2
    )

    # 保存结果
    fold_dir = f"./kfold_time/fold_{fold}"
    os.makedirs(fold_dir, exist_ok=True)
    train_proc.to_csv(os.path.join(fold_dir, 'training_data.csv'), index=False)
    valid_proc.to_csv(os.path.join(fold_dir, 'validing_data.csv'), index=False)
    print(f"[Fold {fold}] 处理完成。")

if __name__ == "__main__":
    # 读取并排序数据
    data = pd.read_csv(
        './output/merge_train_data.csv',
        parse_dates=['post_time_normalized'],
        keep_default_na=False
    )
    data.sort_values('post_time_normalized', inplace=True)

    # 去除旧的聚类特征列
    drop_cols = ['cluster_total_count', 'main_cluster', 'main_cluster_count', 'main_cluster_freq']
    data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)

    # 使用时间序列切分
    tscv = TimeSeriesSplit(n_splits=10)
    gpu_ids = [0]*10

    processes = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
        train_df = data.iloc[train_idx]
        valid_df = data.iloc[val_idx]
        p = mp.Process(
            target=process_fold,
            args=(fold, train_df, valid_df, gpu_ids[fold])
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[Done] 所有时序折并行处理完毕。")
