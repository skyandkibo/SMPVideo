import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import matplotlib.pyplot as plt
import argparse
import os
import sys
import json
import csv
import logging
from datetime import datetime
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(current_script_path))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from function.function import custom_exp, custom_log, set_seed
from function.util_logger import get_logger
from feature_engineering.feature_engineering import calculate_popularity_score

# 配置Optuna的日志回调函数
class OptunaCallback:
    def __init__(self, logger):
        self.logger = logger
    
    def __call__(self, study, trial):
        self.logger.info(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}. "
                         f"Best is trial {study.best_trial.number} with value: {study.best_trial.value}.")

# 添加Optuna的早停回调函数
class EarlyStoppingCallback:
    def __init__(self, patience=20, min_trials=25):
        self.patience = patience
        self.min_trials = min_trials
        self.best_value = float('inf')
        self.best_trial = 0
        self.stale_count = 0
        
    def __call__(self, study, trial):
        if trial.number < self.min_trials:
            return
            
        if study.best_value < self.best_value:
            self.best_value = study.best_value
            self.best_trial = trial.number
            self.stale_count = 0
        else:
            self.stale_count += 1
            
        if self.stale_count >= self.patience:
            study.stop()
            print(f"Early stopping triggered after {self.patience} trials without improvement.")

# 定义删除异常值的函数，使用IQR（四分位距）
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def get_importance(training_data, testing_data, features, targets, importance_method, logger, args):
    # 数据划分    
    X_train, X_val = training_data[features], testing_data[features]
    y_train, y_val = training_data[targets[0]], testing_data[targets[0]]

    # 标准化数据
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if importance_method == 'xgboost':
        # 构建XGBoost模型进行特征筛选
        base_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            n_jobs=4
        )
    elif importance_method == 'catboost':
        base_model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=5,
            loss_function='MAPE',
            verbose=0,
            random_seed=args.seed
        )
    elif importance_method == 'lightbgm':
        base_model = LGBMRegressor(
            objective='regression',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            verbose=-1
        )

    # 使用RFECV进行特征筛选
    rfecv = RFECV(
        estimator=base_model,
        step=1,
        cv=5,
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=4,
        verbose=1
    )
    
    logger.info("开始RFECV特征筛选(使用XGBoost)...")
    rfecv.fit(X_train_scaled, y_train)
 
    # 获取选中的特征
    selected_features = [features[i] for i in range(len(features)) if rfecv.support_[i]]
    
    logger.info(f"最优特征数量: {rfecv.n_features_}")
    logger.info(f"特征选择得分: {rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - 1]}")
    logger.info(f"选中的特征: {selected_features}")
    logger.info(f"选中特征数量: {len(selected_features)}")
        
    return selected_features

def extract_categorical(training_data, testing_data, categorical_features, logger):
    """只对没有流行度特征的分类变量应用频率编码，且仅用训练集分布编码测试集"""
    # 深拷贝原始数据，避免修改原始数据
    train_data = training_data.copy()
    test_data = testing_data.copy()
        
    features_to_encode = categorical_features.copy()
    
    if not features_to_encode:
        return train_data, test_data
    
    # 对需要频率编码的特征在训练集上计算频率
    for feature in features_to_encode:
        # 计算训练集中每个类别的频率
        value_counts = train_data[feature].value_counts(normalize=True)
        freq_map = value_counts.to_dict()
        
        # 应用到训练集
        train_data[feature] = train_data[feature].map(freq_map)
        
        # 应用到测试集
        test_data[feature] = test_data[feature].map(freq_map)
        
        # 对于测试集中可能出现的新类别值，使用一个小的默认值
        default_val = (min(freq_map.values()) / 10) if freq_map else 0.001
        train_data[feature].fillna(default_val, inplace=True)
        test_data[feature].fillna(default_val, inplace=True)
    
    return train_data, test_data

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义PyTorch MLP模型
class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation='relu'):
        super(PyTorchMLP, self).__init__()
        
        # 定义激活函数
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
            
        # 构建层
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()

# PyTorch训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

# PyTorch评估函数
def evaluate(model, X, y, args, device):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    
    if args.excel_log:
        mape = mean_absolute_percentage_error(custom_exp(y), custom_exp(preds))
    else:
        mape = mean_absolute_percentage_error(y, preds)
        
    return mape, preds

def train_with_early_stopping(X_train, y_train, X_val, y_val, params, args,
                              max_epochs=300, patience=10, tol=1e-4):
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True
    )
    
    # 创建模型
    model = PyTorchMLP(
        input_size=X_train_scaled.shape[1],
        hidden_sizes=params['hidden_layer_sizes'],
        activation=params['activation']
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params['learning_rate_init'],
        weight_decay=params['alpha']
    )
    
    best_mape = float('inf')
    best_epoch = 0
    no_improve = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 计算验证集MAPE
        val_mape, _ = evaluate(model, X_val_scaled, y_val, args, device)
        
        if val_mape + tol < best_mape:
            best_mape = val_mape
            best_epoch = epoch
            no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    # 恢复到最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_mape, best_epoch, model, scaler

def objective(trial, train_list, test_list, selected_features, targets, args):
    # 超参数采样
    params = {
        'hidden_layer_sizes': (
            trial.suggest_int('hidden_size_1', 100, 250),
            trial.suggest_int('hidden_size_2', 50, 120),
        ),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
    }

    mape_scores = []
    best_epochs = []
    
    for i in range(len(train_list)):
        X_train = train_list[i][selected_features]
        y_train = train_list[i][targets[0]]
        X_val = test_list[i][selected_features]
        y_val = test_list[i][targets[0]]

        mape, epoch, _, _ = train_with_early_stopping(
            X_train, y_train, X_val, y_val, 
            params, args, 
            max_epochs=200,
            patience=8
        )
        mape_scores.append(mape)
        best_epochs.append(epoch)

    trial.set_user_attr('iterations', best_epochs)
    trial.set_user_attr('avg_iterations', float(np.mean(best_epochs)))
    return float(np.mean(mape_scores))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLP Regressor")
    parser.add_argument('--model', type=str, default='train_val',
                        help="train_val or train_test")
    parser.add_argument('--excel_log', action='store_true',
                        help="video network for log or not")
    parser.add_argument('--importance', action='store_true')      
    parser.add_argument('--importance_method', type=str, default='catboost', help="importance method")
    parser.add_argument('--output_dir', type=str, default='./machine_learning/mlp/output', help="output directory")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--gpu', action='store_true', help="use GPU acceleration")
    parser.add_argument('--n_jobs', type=int, default=8, help="并行任务数量，使用GPU时建议为GPU数量的1-4倍")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # 如果使用多GPU

    output_dir = os.path.join(args.output_dir, f'importance_{args.importance}', f'{args.importance_method}')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建全局logger
    logger = get_logger(output_dir, f'result_{args.model}_{args.seed}')
    logger.info("开始运行MLP回归器...")

    training_data_list, testing_data_list = [], []
    if args.model == 'train_val':
        for fold in range(5):
            fold_dir = f"./kfold/fold_{fold}"
            training_data = pd.read_csv(os.path.join(fold_dir, 'training_data.csv'), keep_default_na=False)
            testing_data = pd.read_csv(os.path.join(fold_dir, 'validing_data.csv'), keep_default_na=False)
            training_data_list.append(training_data)
            testing_data_list.append(testing_data)
    else:
        training_data = pd.read_csv(f'./output/merge_train_data.csv', keep_default_na=False)
        testing_data = pd.read_csv(f'./output/merge_test_data.csv', keep_default_na=False)
        training_data_list.append(training_data)
        testing_data_list.append(testing_data)
    
    for i in range(len(training_data_list)):
        training_data_list[i] = remove_outliers(training_data_list[i], 'popularity')

    # 转换分类变量
    categorical_features = ['uid', 'time_period', 'is_holiday', 'post_text_language', 'post_location', 'is_title', 'is_suggested', 'big_music', 'music_title', 'video_is_heng', 'main_cluster', 'has_face', 'dominant_emotion']    
    
    # 流行度计算变量 - 保存原始名称用于后续查找
    col_features = ['uid', 'time_period', 'post_text_language', 'post_location',  'big_music', 'music_title', 'main_cluster', 'dominant_emotion']
    for i in range(len(training_data_list)):
        training_data_list[i], testing_data_list[i] = calculate_popularity_score(training_data_list[i], testing_data_list[i], col_features)

    # 需要频率编码的分类特征（那些没有流行度特征的）
    features_to_encode = [col for col in categorical_features if col not in col_features]
    logger.info(f"需要频率编码的分类特征: {features_to_encode}")

    # 对需要频率编码的分类特征进行编码
    logger.info("开始对分类特征进行频率编码...")
    for i in range(len(training_data_list)):
        training_data_list[i], testing_data_list[i] = extract_categorical(training_data_list[i], testing_data_list[i], features_to_encode, logger)

    middle_features = ['user_follower_count', 'user_following_count', 'user_likes_count', 'user_digg_count', 'user_video_count']

    can_normalized_features = ['year', 'month', 'day', 'hour', 'music_duration', 'len_title', 'len_suggested', 'video_duration', 'total_frames', 'video_width', 'video_height', 'ratio_number', 'video_area', 'video_width/height', 'cluster_total_count', 'main_cluster_count', 'main_cluster_freq']
    can_normalized_features += ['face_frame_rate', 'avg_faces_per_frame', 'multi_person_ratio', 'avg_face_area_ratio', 'max_face_area_ratio', 'close_up_ratio', 'emotion_entropy', 'positive_emotion_ratio']
    for middle_feature in middle_features:
        can_normalized_features += [f'log_{middle_feature}']
    can_normalized_features += ['post_time_normalized', 'frame_rate']
    can_normalized_features += ['log_likes_per_video', 'log_followers_per_following', 'log_digg_pre_like_ratio']
    can_normalized_features += [f"{col}_popularity" for col in col_features]
    
    # 构建完整的特征列表
    features = []
    # 只添加需要频率编码的分类特征（它们已经被编码成数值了）
    features.extend(features_to_encode)    
    features.extend(can_normalized_features)    
    logger.info(f"最终特征数量: {len(features)}")

    # 在对数据进行处理之前，先对目标变量进行处理
    targets = ['popularity']
    if args.excel_log:
        for i in range(len(training_data_list)):
            training_data_list[i][targets[0]] = training_data_list[i][targets[0]].apply(custom_log)
        logger.info("对目标变量应用了对数转换")
    
    # 确保所有特征都是数值类型
    for feat in features:
        for i in range(len(training_data_list)):
            if not pd.api.types.is_numeric_dtype(training_data_list[i][feat]):
                logger.warning(f"警告: 特征 {feat} 不是数值类型，当前类型: {training_data_list[i][feat].dtype}")
            
    selected_features = features.copy()

    if args.model == 'train_val':
        logger.info(f"特征重要性分析开关: {args.importance}")
        if args.importance:
            logger.info("开始进行特征重要性分析...")
            selected_features_list = []
            for i in range(len(training_data_list)):
                selected_features_list.append(get_importance(training_data_list[i], testing_data_list[i], features, targets, args.importance_method, logger, args))
            # 统计每个特征在几折中出现
            counter = Counter()
            for feat_list in selected_features_list:
                counter.update(feat_list)
            print(counter)
            # 保留至少出现在 N 折中的特征
            selected_features = [feat for feat, count in counter.items() if count >= 1]  # 比如出现 ≥1次
            print(selected_features)
            print(len(selected_features))           

            logger.info(f"根据特征重要性选择了{len(selected_features)}个特征")
        else:
            selected_features = features.copy()
            logger.info("使用所有特征，不进行特征选择")
        
        # 配置Optuna中的日志
        optuna_logger = logging.getLogger("optuna")
        
        # 添加早停回调
        early_stopping = EarlyStoppingCallback(patience=25, min_trials=30)
        callbacks = [OptunaCallback(logger), early_stopping]
        
        logger.info("开始Optuna超参数优化...")
        # 使用TPESampler以加快搜索速度
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(
            lambda trial: objective(trial, training_data_list, testing_data_list, selected_features, targets, args), 
            n_trials=200,  # 减少试验次数
            n_jobs=args.n_jobs,  # 使用命令行参数控制并行数量
            callbacks=callbacks
        )

        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Best MAPE: {study.best_value}")
        logger.info(f"Best trial's average iterations: {study.best_trial.user_attrs['avg_iterations']}")
        logger.info(f"Best trial's iterations by fold: {study.best_trial.user_attrs['iterations']}")

        best_params = study.best_params
        best_iterations = int(np.ceil(study.best_trial.user_attrs['avg_iterations']))
        
        # 使用最佳超参数构建模型参数
        best_mlp_params = {
            'hidden_layer_sizes': (best_params['hidden_size_1'], best_params['hidden_size_2']),
            'activation': best_params['activation'],
            'alpha': best_params['alpha'],
            'batch_size': best_params['batch_size'],
            'learning_rate_init': best_params['learning_rate_init'],
            'max_iter': best_iterations,
        }
        
        logger.info(f"最佳MLP参数: {best_mlp_params}")

        with open(f'{output_dir}/best_params.json', 'w') as f:
            json.dump(best_mlp_params, f, indent=4)
        with open(f'{output_dir}/selected_features.json', 'w') as f:
            json.dump(selected_features, f, indent=4)
    else:
        with open(f'{output_dir}/best_params.json', 'r') as f:
            best_mlp_params = json.load(f)
        with open(f'{output_dir}/selected_features.json', 'r') as f:
            selected_features = json.load(f)
            
        # 标准化数据并训练最终模型
        logger.info("开始标准化数据并训练最终模型...")
        logger.info(f"训练设备: {device}")
        
        X_train = training_data_list[0][selected_features]
        y_train = training_data_list[0][targets[0]]
        X_test = testing_data_list[0][selected_features]
        
        # 训练最终模型
        _, _, final_model, final_scaler = train_with_early_stopping(
            X_train, y_train, X_train, y_train,  # 使用训练数据作为验证（只是为了结构完整）
            best_mlp_params, args,
            max_epochs=best_mlp_params['max_iter'],
            patience=999999  # 不实际使用早停
        )
        
        # 预测测试集
        logger.info("生成测试集预测结果...")
        X_test_scaled = final_scaler.transform(X_test)
        _, y_pred_test = evaluate(final_model, X_test_scaled, np.zeros(len(X_test)), args, device)  # y不重要，只需要predictions
        X_pid = testing_data_list[0]['pid']

        if args.excel_log:
            logger.info("转换对数预测值回原始值...")
            y_pred_test = custom_exp(y_pred_test)   
        
        test_predictions={}
        for pid, pred in zip(X_pid, y_pred_test):
            test_predictions[pid] = pred

        logger.info(f"保存预测结果到 {output_dir}/test_predictions.csv")
        with open(f'./{output_dir}/test_predictions.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pid', f'popularity_score'])
            for post_id, prediction in test_predictions.items():
                writer.writerow([post_id, prediction])
        
        with open(f'{output_dir}/best_params.json', 'w') as f:
            json.dump(best_mlp_params, f, indent=4)
            
        with open(f'{output_dir}/selected_features.json', 'w') as f:
            json.dump(selected_features, f, indent=4)

        logger.info("任务完成")

