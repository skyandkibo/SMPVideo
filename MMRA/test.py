import argparse
import os
from datetime import datetime
import logging
from functools import partial
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from dataloader.MicroLens100k.dataset import MyData, custom_collate_fn
import random
import numpy as np
from scipy.stats import spearmanr
from model.MicroLens100k.MMRA import Model
import pandas as pd
import csv

BLUE = '\033[94m'
ENDC = '\033[0m'


def seed_init(seed):

    seed = int(seed)

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


def print_init_msg(logger, args):

    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")

    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")

    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_path} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")

    # logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Number of retrieved items used in this testing: " + ENDC + f"{args.num_of_retrieved_items}")

    logger.info(BLUE + "Alpha: " + ENDC + f"{args.alpha}")

    logger.info(BLUE + "Number of frames: " + ENDC + f"{args.frame_num}")

    logger.info(BLUE + "Testing Starts!" + ENDC)


def delete_special_tokens(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:

        content = file.read()

    content = content.replace(BLUE, '')

    content = content.replace(ENDC, '')

    with open(file_path, 'w', encoding='utf-8') as file:

        file.write(content)


# 定义一个通用查询函数
def query_info(data_store, vid=None, pid=None, uid=None):
    result = []
    for (v, p, u), info in data_store.items():
        if (vid is None or v == vid) and (pid is None or p == pid) and (uid is None or u == uid):
            result.append({"vid":v, "pid":p, "uid":u})
    return result

def test(args, result_excel):

    device = torch.device(args.device)

    model_id = args.model_id

    dataset_id = args.dataset_id

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f"test_{model_id}_{dataset_id}"

    father_folder_name = args.save

    if not os.path.exists(father_folder_name):

        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    logger = logging.getLogger()

    logger.handlers = []

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')

    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    batch_size = args.batch_size

    test_data = MyData(os.path.join(os.path.join(args.input_dir, 'test.pkl')), is_train=False)

    custom_collate_fn_partial = partial(custom_collate_fn, num_of_retrieved_items=args.num_of_retrieved_items,
                                        num_of_frames=args.frame_num, is_train=False)

    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=custom_collate_fn_partial)

    model = torch.load(args.model_path)

    total_test_step = 0

    total_MAE = 0

    total_nMSE = 0

    total_SRC = 0

    print_init_msg(logger, args)

    model.eval()

    test_predictions = {}

    with torch.no_grad():

        for batch in tqdm(test_data_loader, desc='Testing'):

            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            visual_feature, textual_feature, similarity, retrieved_visual_feature, retrieved_textual_feature, retrieved_label, item_id = batch

            output = model.forward(visual_feature, textual_feature, similarity, retrieved_visual_feature,
                                   retrieved_textual_feature,
                                   retrieved_label)

            output = output.to('cpu')

            output = np.array(output)
            
            for x_i, single_item_id in enumerate(item_id):
                vpu_list = query_info(result_excel, vid=single_item_id, pid=None, uid=None)
                if len(output[x_i])==1:
                    test_predictions[vpu_list[0]['pid']] = float(output[x_i][0])
                else:
                    print("Error: ", single_item_id, output[x_i])
    # print(result_pid)
    # print(result_popularity)
    with open(f'./{folder_path}/test_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pid', f'popularity_score'])
        for post_id, prediction in test_predictions.items():
            writer.writerow([post_id, prediction])

    #         label = label.to('cpu')

    #         label = np.array(label)

    #         MAE = mean_absolute_error(label, output)

    #         SRC, _ = spearmanr(output, label)

    #         nMSE = np.mean(np.square(output - label)) / (label.std() ** 2)

    #         total_test_step += 1

    #         total_MAE += MAE

    #         total_SRC += SRC

    #         total_nMSE += nMSE

    # logger.warning(f"[ Test Result ]:  \n {args.metric[0]} = {total_nMSE / total_test_step}"
    #                f"\n{args.metric[1]} = {total_SRC / total_test_step}\n{args.metric[2]} = {total_MAE / total_test_step}\n")

    logger.info("Test is ended!")

    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")


def main(args, result_excel):

    seed_init(args.seed)

    test(args, result_excel)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default='2024', type=str, help='value of random seed')

    parser.add_argument('--device', default='cuda:0', type=str, help='device used in testing')

    # parser.add_argument('--metric', default=['nRMSE', 'SRC', 'MAE'], type=list, help='the judgement of the testing')

    parser.add_argument('--save', default='test_results', type=str, help='folder to save the results')

    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')

    parser.add_argument('--dataset_id', default='tiktok', type=str, help='id of dataset')

    parser.add_argument('--dataset_path', default='datasets', type=str, help='path of dataset folder')

    parser.add_argument('--input_dir', type=str, default='', help='input directory for the pkl file')

    parser.add_argument('--dataset_model', type=str, default='train_val', help='Path to the dataset')

    parser.add_argument('--model_id', default='MMRA', type=str, help='id of model')

    parser.add_argument('--num_of_retrieved_items', default=10, type=int, help='number of retrieved items used this training, hyper-parameter')

    parser.add_argument('--alpha', default=0.6, type=int, help='Alpha, hyper-parameter')

    parser.add_argument('--frame_num', default=10, type=int, help='frame number of each video, hyper-parameter')

    parser.add_argument('--feature_num', type=int, default=2, help='Number of features')

    parser.add_argument('--feature_dim', type=int, default=768, help='Dimension of features')

    parser.add_argument('--label_dim', type=int, default=1, help='Dimension of labels')

    parser.add_argument('--model_path',
                        default=r'',
                        type=str, help='path of trained model')

    args = parser.parse_args()

    # 读取 Excel 文件
    file_path = '../dataset/test.xlsx'  # 替换为你的 Excel 文件路径
    sheet_name = 'Sheet1'  # 替换为你的工作表名称

    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 将 NaN 替换为空字符串
    df = df.fillna("")

    # 初始化一个空字典
    result_excel = {}

    for index, row in df.iterrows():
        result_excel[(row['vid'],row['pid'],row['uid'])] = row[3:].to_dict()
    
    main(args, result_excel)
