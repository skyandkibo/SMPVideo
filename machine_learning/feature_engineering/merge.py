import pandas as pd
import numpy as np
import ast
from math import log2
import argparse

positive_emotions = {"happy", "surprise", "neutral"}

def parse_distribution(s):
    try:
        if isinstance(s, dict):
            return s
        return ast.literal_eval(s)
    except Exception:
        return {}

def compute_entropy_and_positive_ratio(dist):
    if not dist:
        return 0.0, 0.0
    probs = np.array(list(dist.values()), dtype=np.float32)
    entropy = -np.sum(probs * np.log2(probs + 1e-9))  # 避免 log(0)
    pos_ratio = sum(dist.get(em, 0.0) for em in positive_emotions)
    return round(entropy, 4), round(pos_ratio, 4)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Merge DataFrames')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    args = parser.parse_args()

    pd_audio=pd.read_csv(f'./output/{args.model}_audio.csv')
    pd_face = pd.read_csv(f'./output/{args.model}_face.csv')
    pd_feature = pd.read_csv(f'./output/{args.model}ing_data.csv')

    merged_df = pd.merge(pd_feature, pd_face, on='vid', how='outer', suffixes=('', '_drop'))
    merged_df = merged_df[[col for col in merged_df.columns if not col.endswith('_drop')]]

    merged_df = pd.merge(merged_df, pd_audio, on='vid', how='outer')
    print(merged_df.columns)
    print(merged_df.head())

    merged_df["emotion_distribution"] = merged_df["emotion_distribution"].apply(parse_distribution)
    merged_df[["emotion_entropy", "positive_emotion_ratio"]] = merged_df["emotion_distribution"].apply(
        lambda d: pd.Series(compute_entropy_and_positive_ratio(d))
    )

    merged_df.to_csv(f'./output/merge_{args.model}_data.csv', index=False)