import os
import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import matplotlib.pyplot as plt
import argparse
import sys
import json
import csv
import logging
from sklearn.inspection import permutation_importance
import io
import sys
from pandas.api.types import CategoricalDtype
from collections import Counter
from sklearn.preprocessing import StandardScaler

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

# 配置Optuna日志处理器重定向到我们的logger
class OptunaLoggerAdapter:
    def __init__(self, logger):
        self.logger = logger
        
    def info(self, msg):
        self.logger.info(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def debug(self, msg):
        self.logger.debug(msg)
        
    def error(self, msg):
        self.logger.error(msg)

columns_to_plot = ['popularity']

# 定义删除异常值的函数，使用IQR（四分位距）
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def select_importance_gain(important_features, threshold, logger):
    imp_df = pd.DataFrame([
        {'feature': feat, 'gain': important_features[feat]}
        for feat in important_features
    ]).sort_values('gain', ascending=False).reset_index(drop=True)
    
    pos_df = imp_df[imp_df['gain'] > 0]

    pos_df['rel_gain'] = pos_df['gain'] / pos_df['gain'].sum()
    pos_df['cum_gain'] = pos_df['rel_gain'].cumsum()

    logger.info("Cumulative Gain (重要性增量)：")
    logger.info(f"\n{pos_df}")
    # 3a. 累计阈值法：取前 k 个使累计重要性 ≥ 阈值（如 0.95）
    selected_by_cumsum_gain = pos_df.loc[pos_df['cum_gain'] <= threshold, 'feature'].tolist()
    # +1 保证覆盖阈值
    if len(selected_by_cumsum_gain) < len(pos_df):
        selected_by_cumsum_gain.append(pos_df.loc[~pos_df['feature'].isin(selected_by_cumsum_gain), 'feature'].iloc[0])

    logger.info(f"按累计重要性≥{threshold*100:.0f}% 选出的特征数：{len(selected_by_cumsum_gain)}")
    logger.info(f"选出的特征: {selected_by_cumsum_gain}")
    
    return selected_by_cumsum_gain

def select_importance_mape(important_features, threshold, logger):
    # 构造 DataFrame 并排序
    imp_df = pd.DataFrame([
        {'feature': feat, 'importance': important_features[feat]}
        for feat in important_features
    ]).sort_values('importance', ascending=False).reset_index(drop=True)

    logger.info("Permutation Importance (MAPE 增量)：")
    logger.info(f"\n{imp_df}")

    pos_df = imp_df[imp_df['importance'] > 0]
    logger.info("Positive Importance (MAPE 增量)：")
    logger.info(f"\n{pos_df}")

    pos_df['rel_imp'] = pos_df['importance'] / pos_df['importance'].sum()
    pos_df['cum_imp'] = pos_df['rel_imp'].cumsum()

    logger.info("Cumulative Importance (MAPE 增量)：")
    logger.info(f"\n{pos_df}")
    selected_by_cumsum = pos_df.loc[pos_df['cum_imp'] <= threshold, 'feature'].tolist()

    #+1,确保>threshold
    if len(selected_by_cumsum) < len(pos_df):
        next_feature = pos_df.loc[~pos_df['feature'].isin(selected_by_cumsum), 'feature'].iloc[0]
        selected_by_cumsum.append(next_feature)

    logger.info(f"选出的特征: {selected_by_cumsum}")

    return selected_by_cumsum

# 创建上下文管理器来捕获所有输出
class SuppressStdoutStderr:
    """
    一个上下文管理器，用于捕获和抑制所有的标准输出和标准错误，但保留logger输出
    """
    def __init__(self, logger):
        self.logger = logger
        
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.stdout = io.StringIO()
        sys.stderr = self.stderr = io.StringIO()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
def get_importance(training_data, testing_data, features, targets, threshold, logger, args):    
    X_train, X_val = training_data[features], testing_data[features]
    y_train, y_val = training_data[targets[0]], testing_data[targets[0]]
    
    # 模型训练 - 使用默认参数，不指定分类特征
    model = LGBMRegressor(
        objective='regression',
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        verbose=-1
    )
    
    # 设置回调函数
    callbacks = [
        early_stopping(10),
        log_evaluation(0)
    ]

    # 简化训练过程，完全依赖Pandas category类型标记
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=callbacks
    )

    # 获取LightGBM特征重要性
    rel_importance = {features[i]: model.feature_importances_[i] / sum(model.feature_importances_) for i in range(len(features))}

    # 计算置换重要性——直接以 MAPE 为指标
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring='neg_mean_absolute_percentage_error',
        n_repeats=10,
        random_state=args.seed,
        n_jobs=-1
    )
    rel_permutation_importance = {features[i]: -result.importances_mean[i] / sum(-result.importances_mean) for i in range(len(features))}

    return rel_importance, rel_permutation_importance

def objective(trial, train_list, test_list, selected_features, target, logger, args):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': 1000,
        'random_state': args.seed,
        'verbose': -1
    }

    mape_scores = []
    best_iters = []

    for i in range(len(train_list)):
        X_train, X_val = train_list[i][selected_features], test_list[i][selected_features]
        y_train, y_val = train_list[i][target[0]], test_list[i][target[0]]

        # 设置回调函数 - 使用verbose=-1完全禁用输出
        callbacks = [
            early_stopping(20),
            log_evaluation(0)  # 修改为-1完全禁用输出
        ]

        model = LGBMRegressor(**params)

        with SuppressStdoutStderr(logger):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )

        best_iters.append(model.best_iteration_)
        y_pred = model.predict(X_val)

        if args.excel_log:
            mape = mean_absolute_percentage_error(custom_exp(y_val), custom_exp(y_pred))
        else:
            mape = mean_absolute_percentage_error(y_val, y_pred)

        mape_scores.append(mape)

    trial.set_user_attr("avg_best_iteration", int(np.mean(best_iters)))
    return np.mean(mape_scores)

def extract_categorical(training_data, testing_data, categorical_features, unknown_label='__UNKNOWN__'):
    """
    将分类特征统一为 Pandas 的 category dtype，训练集和测试集共用相同类别。

    参数：
    - training_data: pd.DataFrame，训练集
    - testing_data:  pd.DataFrame，测试集
    - categorical_features: list of str，需要转换的列名
    - unknown_label: str，测试集中未出现在训练集里的类别统一标记

    返回：
    - training_data, testing_data: 对应列已转为 category dtype
    """
    for feature in categorical_features:
        if feature in training_data.columns and feature in testing_data.columns:
            # 1. 先转为字符串
            training_data[feature] = training_data[feature].astype(str)
            testing_data[feature]  = testing_data[feature].astype(str)

            # 2. 训练集唯一类别列表
            cats = training_data[feature].unique().tolist()
            # 将 unknown_label 加入类别，作为测试集中未知值的占位
            if unknown_label not in cats:
                cats.append(unknown_label)

            # 3. 定义统一的 CategoricalDtype
            cat_type = CategoricalDtype(categories=cats, ordered=False)

            # 4. 应用到训练和测试
            training_data[feature] = training_data[feature].astype(cat_type)
            testing_data[feature]  = testing_data[feature].astype(cat_type)

            # 5. 将测试集中训练集中未出现的值置为 unknown_label
            testing_data.loc[
                ~testing_data[feature].isin(training_data[feature].cat.categories), 
                feature
            ] = unknown_label

    return training_data, testing_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="lightgbm")
    parser.add_argument('--model', type=str, default='train_val',
                        help="train_val or train_test")
    parser.add_argument('--excel_log', action='store_true',
                        help="video network for log or not")
    parser.add_argument('--importance', action='store_true')      
    parser.add_argument('--threshold', type=float, default=0.85, help="importance threshold")
    parser.add_argument('--output_dir', type=str, default='./machine_learning/lightbgm/output', help="output directory")
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    args = parser.parse_args()
    set_seed(args.seed)
    
    if not args.importance:
        output_dir = os.path.join(args.output_dir, f'importance_{args.importance}')
    else:
        output_dir = os.path.join(args.output_dir, f'importance_{args.importance}', f'{args.threshold}')

    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(output_dir, f'result_{args.model}_{args.seed}')
    
    # 配置optuna日志重定向到我们的logger - 修复方法
    optuna_logger = logging.getLogger('optuna')
    optuna_logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    for handler in optuna_logger.handlers[:]:
        optuna_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(stream=sys.stdout)
    optuna_logger.addHandler(console_handler)
    
    # 添加自定义处理器，将日志重定向到我们的logger
    class LoggerHandler(logging.Handler):
        def __init__(self, target_logger):
            super().__init__()
            self.target_logger = target_logger
            
        def emit(self, record):
            log_msg = self.format(record)
            self.target_logger.info(log_msg)
    
    # 创建一个用于将optuna日志重定向到我们logger的处理器
    optuna_to_our_logger = LoggerHandler(logger)
    optuna_logger.addHandler(optuna_to_our_logger)
    
    # 设置optuna的日志级别
    optuna.logging.set_verbosity(optuna.logging.INFO)

    training_data_list, testing_data_list = [], []

    if args.model == 'train_val':
        for fold in range(5,10):
            fold_dir = f"./kfold_time/fold_{fold}"
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
        for col in columns_to_plot:
            training_data_list[i] = remove_outliers(training_data_list[i], col)

    # 转换分类变量
    categorical_features = ['uid', 'time_period', 'is_holiday', 'post_text_language', 'post_location', 'is_title', 'is_suggested', 'big_music', 'music_title', 'video_is_heng', 'main_cluster', 'has_face', 'dominant_emotion']    
    for i in range(len(training_data_list)):
        training_data_list[i], testing_data_list[i] = extract_categorical(training_data_list[i], testing_data_list[i], categorical_features)

    #流行度计算变量
    col_features = ['uid', 'time_period', 'post_text_language', 'post_location',  'big_music', 'music_title', 'main_cluster', 'dominant_emotion']
    for i in range(len(training_data_list)):
        training_data_list[i], testing_data_list[i] = calculate_popularity_score(training_data_list[i], testing_data_list[i], col_features)

    middle_features = ['user_follower_count', 'user_following_count', 'user_likes_count', 'user_digg_count', 'user_video_count']

    can_normalized_features = ['year', 'month', 'day', 'hour', 'music_duration', 'len_title', 'len_suggested', 'video_duration', 'total_frames', 'video_width', 'video_height', 'ratio_number', 'video_area', 'video_width/height', 'cluster_total_count', 'main_cluster_count', 'main_cluster_freq']
    can_normalized_features += ['face_frame_rate', 'avg_faces_per_frame', 'multi_person_ratio', 'avg_face_area_ratio', 'max_face_area_ratio', 'close_up_ratio', 'emotion_entropy', 'positive_emotion_ratio']
    for middle_feature in middle_features:
        can_normalized_features += [f'log_{middle_feature}']
    can_normalized_features += ['post_time_normalized', 'frame_rate']
    can_normalized_features += ['log_likes_per_video', 'log_followers_per_following', 'log_digg_pre_like_ratio']
    can_normalized_features += [f"{col}_popularity" for col in col_features]

    # 定义特征和目标
    features = categorical_features.copy()
    features += can_normalized_features    
    
    targets = [
        'popularity'
    ]
        
    logger.info(f"feature_number: {len(features)}")

    if args.excel_log:
        for i in range(len(training_data_list)):
            training_data_list[i][targets[0]] = training_data_list[i][targets[0]].apply(custom_log)

    selected_features = features
    for i in range(len(training_data_list)):
        scaler = StandardScaler()
        # 先fit训练集
        scaler.fit(training_data_list[i][can_normalized_features])
        
        # 变换训练集，注意用.loc保持列名
        training_data_list[i].loc[:, can_normalized_features] = scaler.transform(training_data_list[i][can_normalized_features])
        
        # 用相同的scaler变换测试集
        testing_data_list[i].loc[:, can_normalized_features] = scaler.transform(testing_data_list[i][can_normalized_features])

    if args.model == 'train_val':
        logger.info(f"importance: {args.importance}, threshold: {args.threshold}")
        if args.importance:
            rel_importance_dict, rel_permutation_importance_dict = {}, {}
            for i in range(len(training_data_list)):
                rel_importance, rel_permutation_importance = get_importance(training_data_list[i], testing_data_list[i], features, targets, args.threshold ,logger, args)
                for k, v in rel_importance.items():
                    if k not in rel_importance_dict:
                        rel_importance_dict[k] = v
                    else:
                        rel_importance_dict[k] += v
                for k, v in rel_permutation_importance.items():
                    if k not in rel_permutation_importance_dict:
                        rel_permutation_importance_dict[k] = v
                    else:
                        rel_permutation_importance_dict[k] += v

            rel_importance_dict = {k: v / len(training_data_list) for k, v in rel_importance_dict.items()}
            rel_permutation_importance_dict = {k: v / len(training_data_list) for k, v in rel_permutation_importance_dict.items()}
            selected_by_cumsum_gain = select_importance_gain(rel_importance_dict, args.threshold, logger)
            selected_by_cumsum_mape = select_importance_mape(rel_permutation_importance_dict, 0.95, logger)
            selected_features = sorted(set(selected_by_cumsum_gain + selected_by_cumsum_mape))
            logger.info(f"合并后的特征: {selected_features}, 共{len(selected_features)}个")
        else:
            selected_features = features
        
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, training_data_list, testing_data_list, selected_features, targets, logger, args), n_trials=600, n_jobs=4)

        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Best MAPE: {study.best_value}")

        # 获取平均的 best_iteration
        avg_best_iter = int(np.mean([
            t.user_attrs["avg_best_iteration"]
            for t in study.trials if t.value is not None and "avg_best_iteration" in t.user_attrs
        ]))

        # 用最优参数训练全部数据（或分训练集测试集训练）
        best_params = study.best_params
        #更新LightGBM的最佳参数 - 移除categorical_feature参数
        best_params.update({
            'objective': 'regression',
            'n_estimators': avg_best_iter,
            'random_state': args.seed,
            'verbose': -1,
        })
        logger.info(f"最优参数: {best_params}")
        logger.info(f"avg_best_iter: {avg_best_iter}")

        with open(f'{output_dir}/best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        with open(f'{output_dir}/selected_features.json', 'w') as f:
            json.dump(selected_features, f, indent=4)
    else:
        with open(f'{output_dir}/best_params.json', 'r') as f:
            best_params = json.load(f)
        with open(f'{output_dir}/selected_features.json', 'r') as f:
            selected_features = json.load(f)
        logger.info(f"Best params: {best_params}")
        logger.info(f"selected_features: {selected_features}")

        model = LGBMRegressor(**best_params)
        with SuppressStdoutStderr(logger):
            model.fit(
                training_data_list[0][selected_features], 
                training_data_list[0][targets[0]]
            )
    
        y_pred_test = model.predict(testing_data_list[0][selected_features])
        X_pid = testing_data_list[0]['pid']

        if args.excel_log:
            y_pred_test = custom_exp(y_pred_test)   
        
        test_predictions={}
        for pid, pred in zip(X_pid, y_pred_test):
            test_predictions[pid] = pred

        with open(f'{output_dir}/test_predictions.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pid', f'popularity_score'])
            for post_id, prediction in test_predictions.items():
                writer.writerow([post_id, prediction])
        
        with open(f'{output_dir}/best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)
            
        with open(f'{output_dir}/selected_features.json', 'w') as f:
            json.dump(selected_features, f, indent=4)

