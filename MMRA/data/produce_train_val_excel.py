import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="split train and val data")
    parser.add_argument('--split', type=int, default=10,
                        help="split ratio for train and val data")
    args = parser.parse_args()

    root_path = '/sss/Video'
    # Load training and testing data
    train_excel_file = f'{root_path}/train.xlsx'

    df_train = pd.read_excel(train_excel_file)

    # Filter out invalid video_ids from df_train
    invalid_video_ids = ['VIDEO00001914']

    # Clean video_id and apply transformation
    df_train['vid'] = df_train['vid'].astype(str).str.replace("'", "")
    df_train = df_train[~df_train['vid'].isin(invalid_video_ids)]

    # 按2:1的比例拆分数据集
    df_train_new, df_val_new = train_test_split(df_train, test_size=1/args.split, random_state=42)

    # 输出拆分后的数据集大小
    print(f"训练集样本数: {len(df_train_new)}")
    print(f"验证集样本数: {len(df_val_new)}")

    if not os.path.exists(f'./output/train_val/{args.split-1}:1'):
        os.makedirs(f'./output/train_val/{args.split-1}:1')
        
    # 保存拆分后的数据集为Excel文件
    df_train_new.to_excel(f'./output/train_val/{args.split-1}:1/train_new.xlsx', index=False)
    df_val_new.to_excel(f'./output/train_val/{args.split-1}:1/val_new.xlsx', index=False)

    print(f"已将训练集保存至: ./output/train_val/{args.split-1}:1/train_new.xlsx")
    print(f"已将验证集保存至: ./output/train_val/{args.split-1}:1/val_new.xlsx")