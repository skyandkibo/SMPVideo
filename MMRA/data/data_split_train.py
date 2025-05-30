import pandas as pd
import argparse
import pickle
import os

def write_pkl(input, excel_data, output, output_dir):

    results = []
    for vid in excel_data['vid']:
        result = input.loc[input['item_id'] == vid]
        results.append(result)

    df = pd.concat(results, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/{output}.pkl', "wb") as f:
        pickle.dump(df, f)

    print(f"数据更新完成，检索向量已添加到 {output_dir}/{output}.pkl!")

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data split")
    parser.add_argument('--pkl_path', type=str, default='./datasets/tiktok/only_title/data/train_test/train.pkl',
                         help='input pkl path')
    parser.add_argument('--split', type=int, default=10,
                        help="split ratio for train and val data")
    args = parser.parse_args()

    train_excel = pd.read_excel(f'/sss/MyDrive/smp_inform/output/train_val/{args.split-1}:1/train_new.xlsx' , engine='openpyxl')
    val_excel = pd.read_excel(f'/sss/MyDrive/smp_inform/output/train_val/{args.split-1}:1/val_new.xlsx' , engine='openpyxl')
    test_excel = pd.read_excel(f'/sss/MyDrive/smp_inform/output/train_val/{args.split-1}:1/val_new.xlsx' , engine='openpyxl')

    input_pkl = pd.read_pickle(args.pkl_path)

    output_dir = '/'.join(args.pkl_path.split('/')[:-1])
    output_dir = f'{output_dir}/train_val'

    write_pkl(input_pkl, train_excel, "train", output_dir)
    write_pkl(input_pkl, val_excel, "valid", output_dir)
    write_pkl(input_pkl, test_excel, "test", output_dir)
