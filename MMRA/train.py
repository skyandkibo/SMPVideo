import logging
import os
import sys
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from dataloader.MicroLens100k.dataset import MyData, custom_collate_fn
from model.MicroLens100k.MMRA import Model
import random
from functools import partial
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

    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")

    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")

    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")

    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")

    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")

    logger.info(BLUE + "Number of retrieved items used in this training: " + ENDC + f"{args.num_of_retrieved_items}")

    logger.info(BLUE + "Alpha: " + ENDC + f"{args.alpha}")

    logger.info(BLUE + "Number of frames: " + ENDC + f"{args.frame_num}")

    logger.info(BLUE + "Training Starts!" + ENDC)


def make_saving_folder_and_logger(args):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f"train_{args.model_id}_{args.dataset_id}_{args.metric}"

    father_folder_name = args.save

    if not os.path.exists(father_folder_name):

        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    os.makedirs(os.path.join(folder_path, "trained_model"), exist_ok=True)

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

    return father_folder_name, folder_name, logger


def delete_model(father_folder_name, folder_name, min_turn):

    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}/trained_model")

    for i in range(len(model_name_list)):

        if model_name_list[i] != f'model_{min_turn}.pth':

            os.remove(os.path.join(f'{father_folder_name}/{folder_name}/trained_model', model_name_list[i]))


def force_stop(msg):

    print(msg)

    sys.exit(1)


def delete_special_tokens(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:

        content = file.read()

    content = content.replace(BLUE, '')

    content = content.replace(ENDC, '')

    with open(file_path, 'w', encoding='utf-8') as file:

        file.write(content)


def train_val(args, result_excel):

    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)

    device = torch.device(args.device)

    custom_collate_fn_partial = partial(custom_collate_fn, num_of_retrieved_items=args.num_of_retrieved_items,
                                        num_of_frames=args.frame_num)

    train_data = MyData(os.path.join(args.input_dir, 'train.pkl'))
    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate_fn_partial)

    if args.dataset_model == 'train_val':
        valid_data = MyData(os.path.join(os.path.join(args.input_dir, 'valid.pkl')))
        valid_data_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, collate_fn=custom_collate_fn_partial)
    else:
        valid_data_loader = None

    model = Model(feature_dim=args.feature_dim, alpha=args.alpha, frame_num=args.frame_num)

    model = model.to(device)

    if args.loss == 'BCE':

        loss_fn = torch.nn.BCELoss()

    elif args.loss == 'MSE':

        loss_fn = torch.nn.MSELoss()

    else:

        force_stop('Invalid parameter loss!')

    loss_fn.to(device)

    if args.optim == 'Adam':

        optim = Adam(model.parameters(), args.lr)

    elif args.optim == 'SGD':

        optim = SGD(model.parameters(), args.lr)

    else:

        force_stop('Invalid parameter optim!')

    min_total_valid_loss = 1008611

    min_turn = 0

    print_init_msg(logger, args)

    for i in range(args.epochs):

        logger.info(f"-----------------------------------Epoch {i + 1} Start!-----------------------------------")

        min_train_loss, total_valid_loss = run_one_epoch(i+1, model, loss_fn, optim, train_data_loader, valid_data_loader,
                                                         device, logger, result_excel, f"{father_folder_name}/{folder_name}/val")

        logger.info(f"[ Epoch {i + 1} (train) ]: loss = {min_train_loss}")

        if valid_data_loader is not None:
            logger.info(f"[ Epoch {i + 1} (valid) ]: total_loss = {total_valid_loss}")

            if total_valid_loss < min_total_valid_loss:
                min_total_valid_loss = total_valid_loss

                min_turn = i + 1

            logger.critical(
                f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}")

            torch.save(model, f"{father_folder_name}/{folder_name}/trained_model/model_{i + 1}.pth")

            logger.info("Model has been saved successfully!")

            if (i + 1) - min_turn > args.early_stop_turns:
                break
        else:
            torch.save(model, f"{father_folder_name}/{folder_name}/trained_model/model_{i + 1}.pth")
            min_turn = i + 1
            logger.info("Model has been saved successfully!")

    delete_model(father_folder_name, folder_name, min_turn)

    logger.info(BLUE + "Training is ended!" + ENDC)

    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")

# 定义一个通用查询函数
def query_info(data_store, vid=None, pid=None, uid=None):
    result = []
    for (v, p, u), info in data_store.items():
        if (vid is None or v == vid) and (pid is None or p == pid) and (uid is None or u == uid):
            result.append({"vid":v, "pid":p, "uid":u})
    return result

def run_one_epoch(epoch, model, loss_fn, optim, train_data_loader, valid_data_loader, device, logger, result_excel, output_folder_path):

    os.makedirs(output_folder_path, exist_ok=True)
    
    model.train()

    min_train_loss = 1008611

    for batch in tqdm(train_data_loader, desc='Training Progress'):

        batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

        visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
            retrieved_textual_feature_embedding, retrieved_label, label, item_id = batch

        output = model.forward(visual_feature_embedding, textual_feature_embedding, similarity,
                               retrieved_visual_feature_embedding,
                               retrieved_textual_feature_embedding, retrieved_label)

        loss = loss_fn(output, label)

        optim.zero_grad()

        loss.backward()

        optim.step()

        if min_train_loss > loss:

            min_train_loss = loss

    if valid_data_loader is None:
        return min_train_loss, 0
    

    model.eval()
    
    label_list, prediction_list = [], []
    label_dict, prediction_dict = {}, {}

    with torch.no_grad():

        for batch in tqdm(valid_data_loader, desc='Validating Progress'):

            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
                retrieved_textual_feature_embedding, retrieved_label, label, item_id = batch

            output = model.forward(visual_feature_embedding, textual_feature_embedding, similarity,
                                   retrieved_visual_feature_embedding, retrieved_textual_feature_embedding,
                                   retrieved_label)

            output = output.to('cpu')

            label = label.to('cpu')

            output = np.array(output)

            label = np.array(label)

            for x_i, single_item_id in enumerate(item_id):
                vpu_list = query_info(result_excel, vid=single_item_id, pid=None, uid=None)
                if len(output[x_i])==1:
                    prediction_dict[vpu_list[0]['pid']] = float(output[x_i][0])
                    label_dict[vpu_list[0]['pid']] = float(label[x_i][0])
                else:
                    print("Error: ", single_item_id, output[x_i])

    with open(f'{output_folder_path}/val_prediction_{epoch}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pid', f'popularity_score'])
        for post_id, prediction in prediction_dict.items():
            writer.writerow([post_id, prediction])

    prediction_list = np.array(list(prediction_dict.values()))

    label_list = np.array(list(label_dict.values()))

    MAE = mean_absolute_error(label_list, prediction_list)

    nMSE = np.mean(np.square(prediction_list - label_list)) / (label_list.std() ** 2)

    total_valid_loss = MAE + nMSE 
            
    mape = mean_absolute_percentage_error(label_list, prediction_list)

    logger.info(f'Valid nMSE: {nMSE:.4f}, Valid MAPE: {mape}')

    return min_train_loss, total_valid_loss


def main(args, result_excel):

    seed_init(args.seed)

    train_val(args, result_excel)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=str, default='2024', help='Seed for reproducibility')

    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training')

    parser.add_argument('--metric', type=str, default='MSE', help='Metric for evaluation')

    parser.add_argument('--save', type=str, default='train_results', help='Directory to save results')

    parser.add_argument('--epochs', type=int, default=12, help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    parser.add_argument('--early_stop_turns', type=int, default=20, help='Number of turns for early stopping')

    parser.add_argument('--loss', type=str, default='MSE', help='Loss function for training')

    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer for training')

    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')

    parser.add_argument('--dataset_id', type=str, default='tiktok', help='Dataset identifier')

    parser.add_argument('--dataset_path', type=str, default='datasets', help='Path to the dataset')

    parser.add_argument('--input_dir', type=str, default='', help='input directory for the pkl file')

    parser.add_argument('--dataset_model', type=str, default='train_val', help='Path to the dataset')

    parser.add_argument('--model_id', type=str, default='MMRA', help='Model id')

    parser.add_argument('--feature_num', type=int, default=2, help='Number of features')

    parser.add_argument('--num_of_retrieved_items', type=int, default=10, help='Number of retrieved items, hyper-parameter')

    parser.add_argument('--feature_dim', type=int, default=768, help='Dimension of features')

    parser.add_argument('--label_dim', type=int, default=1, help='Dimension of labels')

    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha, hyper-parameter')

    parser.add_argument('--frame_num', type=int, default=10, help='Number of frames, hyper-parameter')

    args_ = parser.parse_args()

    if args_.dataset_model == 'train_val':
        # 读取 Excel 文件
        file_path = '../dataset/train.xlsx'  # 替换为你的 Excel 文件路径
        sheet_name = 'Sheet1'  # 替换为你的工作表名称

        # 使用 pandas 读取 Excel 文件
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 将 NaN 替换为空字符串
        df = df.fillna("")

        # 初始化一个空字典
        result_excel = {}

        for index, row in df.iterrows():
            result_excel[(row['vid'],row['pid'],row['uid'])] = row[3:].to_dict()
    else:
        result_excel = {}

    main(args_, result_excel)
