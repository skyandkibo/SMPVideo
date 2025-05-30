import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from collections import Counter
import ast
from sentence_transformers import SentenceTransformer
import hdbscan
from tqdm.auto import tqdm
import pandas as pd
import argparse
import sys

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(current_script_path))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from function.function import set_seed

def extract_keywords(row, post_content_col, suggested_col):
    keywords = []
    content = row.get(post_content_col)
    if pd.notna(content):
        keywords += [w.strip() for w in content.split(",") if w.strip()]
    suggested = row.get(suggested_col)
    if pd.notna(suggested):
        try:
            suggested_list = ast.literal_eval(suggested)
            if isinstance(suggested_list, list):
                keywords += [w.strip() for w in suggested_list if isinstance(w, str)]
        except:
            pass
    return keywords

def count_semantic_clusters(keywords, keyword_to_cluster):
    counter = Counter()
    for kw in keywords:
        label = keyword_to_cluster.get(kw)
        if label is not None:
            counter[label] += 1
    return counter

def get_main_cluster(count_dict):
    if not count_dict:
        return -1
    return max(count_dict.items(), key=lambda x: x[1])[0]

def build_cluster_features(X_train, X_test,
                           post_content_col='post_content',
                           suggested_col='post_suggested_words',
                           seed=42,
                           min_cluster_size=2):
    """
    对训练集关键词聚类，并在训练/测试集中生成以下特征：
      - main_cluster: 出现频次最多的聚类ID
      - main_cluster_count: main_cluster 出现次数
      - cluster_total_count: 所有聚类关键词总次数
      - main_cluster_freq: main_cluster 频次 = main_cluster_count / cluster_total_count

    优化点：缓存每行的聚类计数，避免重复计算 count_semantic_clusters。
    """
    tqdm.pandas()

    # 训练集关键词提取
    train_kw = X_train.progress_apply(
        lambda row: extract_keywords(row, post_content_col, suggested_col), axis=1
    )

    # 构造唯一关键词列表并嵌入
    flat = [kw for sub in train_kw for kw in sub]
    unique_keywords = sorted(set(flat))
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(unique_keywords, batch_size=64, show_progress_bar=True)

    # 在训练集上做聚类
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                metric='euclidean', approx_min_span_tree=False)
    labels = clusterer.fit_predict(embeddings)
    keyword_to_cluster = {kw: lbl for kw, lbl in zip(unique_keywords, labels) if lbl != -1}
    # 缓存训练集每行的聚类次数dict
    train_counts = train_kw.progress_apply(
        lambda kws: count_semantic_clusters(kws, keyword_to_cluster)
    )
    # 生成训练特征
    X_train['cluster_total_count'] = train_counts.apply(lambda d: sum(d.values()))
    X_train['main_cluster'] = train_counts.apply(get_main_cluster)
    X_train['main_cluster_count'] = train_counts.apply(
        lambda d: d.get(get_main_cluster(d), 0)
    )
    X_train['main_cluster_freq'] = X_train.apply(
        lambda row: row['main_cluster_count'] / row['cluster_total_count'] if row['cluster_total_count'] > 0 else 0,
        axis=1
    )

    # 测试集关键词提取并映射
    test_kw = X_test.progress_apply(
        lambda row: extract_keywords(row, post_content_col, suggested_col), axis=1
    )
    # 缓存测试集每行的聚类次数dict
    test_counts = test_kw.progress_apply(
        lambda kws: count_semantic_clusters(kws, keyword_to_cluster)
    )
    # 生成测试特征
    X_test['cluster_total_count'] = test_counts.apply(lambda d: sum(d.values()))
    X_test['main_cluster'] = test_counts.apply(get_main_cluster)
    X_test['main_cluster_count'] = test_counts.apply(
        lambda d: d.get(get_main_cluster(d), 0)
    )
    X_test['main_cluster_freq'] = X_test.apply(
        lambda row: row['main_cluster_count'] / row['cluster_total_count'] if row['cluster_total_count'] > 0 else 0,
        axis=1
    )

    return X_train, X_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract semantic clusters from keywords.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    args = parser.parse_args()

    set_seed(42)

    training_data = pd.read_csv(args.train_path)
    testing_data = pd.read_csv(args.test_path)

    # 检查并删除旧的聚类相关列，以重新构建 fresh 版本（避免遗留）
    columns_to_remove = ['cluster_total_count', 'main_cluster', 'main_cluster_count', 'main_cluster_freq']
    training_data.drop(columns=[col for col in columns_to_remove if col in training_data.columns], inplace=True)
    testing_data.drop(columns=[col for col in columns_to_remove if col in testing_data.columns], inplace=True)

    training_data, testing_data = build_cluster_features(
        training_data, testing_data,
        post_content_col='post_content',
        suggested_col='post_suggested_words'
    )

    from feature_engineering import calculate_popularity_score

    training_data, testing_data = calculate_popularity_score(training_data, testing_data, ['main_cluster'])

    training_data.to_csv(args.train_path, index=False)
    testing_data.to_csv(args.test_path, index=False)
    print(f"[Done] 训练集和测试集的聚类特征已保存到：{args.train_path} 和 {args.test_path}")