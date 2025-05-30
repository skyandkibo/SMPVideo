import pandas as pd
import numpy as np
import optuna
import argparse
import os
import sys
import json
import csv
import logging
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
from pandas.api.types import CategoricalDtype
from collections import Counter
from catboost import Pool
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

# 删除异常值（IQR方法）
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

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

# CatBoost 特征选择
def get_importance(training_data, testing_data, features, targets, cat_features, threshold, logger, args):
    X_train, X_val = training_data[features], testing_data[features]
    y_train, y_val = training_data[targets[0]], testing_data[targets[0]]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=5,
        loss_function='MAPE',
        cat_features=cat_features,
        verbose=0,
        random_seed=args.seed
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)

    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)

    importances = model.get_feature_importance(data=train_pool, type='LossFunctionChange')

    # 转成 DataFrame 并排序
    imp_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    logger.info("特征重要性：")
    logger.info(imp_df)

    pos_df = imp_df[imp_df['importance'] > 0]
    logger.info("正重要性特征：")
    logger.info(pos_df)

    # 2. 归一化、计算累计重要性
    pos_df['rel_importance'] = pos_df['importance'] / pos_df['importance'].sum()

    return pos_df

def select_features(important_features, threshold, logger):
    imp_df = pd.DataFrame([
        {'feature': feat, 'gain': important_features[feat]}
        for feat in important_features
    ]).sort_values('gain', ascending=False).reset_index(drop=True)

    pos_df = imp_df[imp_df['gain'] > 0]

    pos_df['rel_importance'] = pos_df['gain'] / pos_df['gain'].sum()
    pos_df['cum_importance'] = pos_df['rel_importance'].cumsum()
    logger.info("正重要性特征（归一化、累计）：")
    logger.info(pos_df)
    
    # 3a. 累计阈值法：取前 k 个使累计重要性 ≥ 阈值
    selected_by_cumsum = pos_df.loc[pos_df['cum_importance'] <= threshold, 'feature'].tolist()
    # +1 保证覆盖阈值
    if len(selected_by_cumsum) < len(pos_df):
        next_feature = pos_df.loc[~pos_df['feature'].isin(selected_by_cumsum), 'feature'].iloc[0]
        selected_by_cumsum.append(next_feature)

    logger.info(f"按累计重要性≥{threshold*100:.0f}% 选出的特征数：{len(selected_by_cumsum)}")
    logger.info(selected_by_cumsum)

    return selected_by_cumsum

# Optuna 优化目标函数
def objective(trial, train_list, test_list, selected_features, cat_features, target, logger, args):
    params = {
        'loss_function': 'MAPE',
        'iterations': 10000,
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        'depth': trial.suggest_int("depth", 4, 10),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'random_seed': args.seed,
        'verbose': 0
    }

    mape_scores = []
    best_iters = []

    for i in range(len(train_list)):
        X_train, X_val = train_list[i][selected_features], test_list[i][selected_features]
        y_train, y_val = train_list[i][target[0]], test_list[i][target[0]]

        model = CatBoostRegressor(**params, cat_features=cat_features)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=0)

        best_iters.append(model.get_best_iteration())
        y_pred = model.predict(X_val)

        mape = mean_absolute_percentage_error(custom_exp(y_val), custom_exp(y_pred)) if args.excel_log else mean_absolute_percentage_error(y_val, y_pred)
        mape_scores.append(mape)

    trial.set_user_attr("avg_best_iteration", int(np.mean(best_iters)))
    return np.mean(mape_scores)

# 添加Optuna回调函数来将优化过程中的信息输出到logger
class OptunaCallback:
    def __init__(self, logger):
        self.logger = logger
        
    def __call__(self, study, trial):
        self.logger.info(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}. Best is trial {study.best_trial.number} with value: {study.best_value}.")

if __name__ == "__main__":
    # CLI 参数
    parser = argparse.ArgumentParser(description="CatBoost Regressor for Popularity Prediction")
    parser.add_argument('--model', type=str, default='train_val')
    parser.add_argument('--excel_log', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--importance', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./machine_learning/catboost/output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    if not args.importance:
        output_dir = os.path.join(args.output_dir, f'importance_{args.importance}')
    else:
        output_dir = os.path.join(args.output_dir, f'importance_{args.importance}', f'{args.threshold}')

    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(output_dir, f'result_{args.model}_{args.seed}')

    training_data_list, testing_data_list = [], []

    if args.model == 'train_val':
        for fold in range(5):
            fold_dir = f"./kfold/fold_{fold}"
            training_data = pd.read_csv(os.path.join(fold_dir, 'training_data.csv'), keep_default_na=False)
            testing_data = pd.read_csv(os.path.join(fold_dir, 'validing_data.csv'), keep_default_na=False)
            training_data_list.append(training_data)
            testing_data_list.append(testing_data)
    else:
        training_data = pd.read_csv('./output/merge_train_data.csv', keep_default_na=False)
        testing_data = pd.read_csv('./output/merge_test_data.csv', keep_default_na=False)
        training_data_list.append(training_data)
        testing_data_list.append(testing_data)

    # 异常值处理
    for i in range(len(training_data_list)):
        training_data_list[i] = remove_outliers(training_data_list[i], 'popularity')
   
    # 分类变量
    categorical_features = ['uid', 'time_period', 'is_holiday', 'post_text_language', 'post_location', 'is_title', 'is_suggested', 'big_music', 'music_title', 'video_is_heng', 'main_cluster', 'has_face', 'dominant_emotion']    
    for i in range(len(training_data_list)):
        training_data_list[i], testing_data_list[i] = extract_categorical(training_data_list[i], testing_data_list[i], categorical_features)

    # 其他特征
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

    features = categorical_features.copy()
    features += can_normalized_features    
    
    logger.info("特征列表：")
    logger.info(features)
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
            rel_importance_dict = {}
            for i in range(len(training_data_list)):
                rel_importance = get_importance(training_data_list[i], testing_data_list[i], features, targets, categorical_features, args.threshold, logger, args)
                for _, row in rel_importance.iterrows():
                    k, v = row['feature'], row['rel_importance']  
                    if k not in rel_importance_dict:
                        rel_importance_dict[k] = v
                    else:
                        rel_importance_dict[k] += v
            
            rel_importance_dict = {k: v / len(training_data_list) for k, v in rel_importance_dict.items()}
            selected_features = select_features(rel_importance_dict, args.threshold, logger)
        else:
            selected_features = features

        cat_new_features = [x for x in selected_features if x in categorical_features]
        study = optuna.create_study(direction="minimize")
        
        # 配置Optuna日志
        optuna_logger = logging.getLogger("optuna")
        # 移除所有默认的处理程序
        for handler in optuna_logger.handlers[:]:
            optuna_logger.removeHandler(handler)
        # 禁用Optuna的日志传播
        optuna_logger.propagate = False
        
        # 创建回调函数将Optuna的输出重定向到我们的logger
        callback = OptunaCallback(logger)
        
        # 运行优化并添加回调
        study.optimize(
            lambda trial: objective(trial, training_data_list, testing_data_list, selected_features, cat_new_features, targets, logger, args), 
            n_trials=600, 
            n_jobs=4,
            callbacks=[callback]
        )

        best_params = study.best_params
        avg_best_iter = int(np.mean([t.user_attrs["avg_best_iteration"] for t in study.trials if "avg_best_iteration" in t.user_attrs]))
        
        best_params.update({
            'iterations': avg_best_iter,
            'loss_function': 'MAPE',
            'random_seed': args.seed,
            'verbose': 0
        })
        logger.info(f"Best params: {best_params}")
        logger.info(f"avg_best_iter: {avg_best_iter}")
        logger.info(f"selected_features: {selected_features}")
    
        # 训练最终模型
        with open(f'{output_dir}/best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        with open(f'{output_dir}/selected_features.json', 'w') as f:
            json.dump(selected_features, f, indent=4)
    else:
        with open(f'{output_dir}/best_params.json', 'r') as f:
            best_params = json.load(f)
        with open(f'{output_dir}/selected_features.json', 'r') as f:
            selected_features = json.load(f)
        cat_new_features = [x for x in selected_features if x in categorical_features]

        final_model = CatBoostRegressor(**best_params, cat_features=cat_new_features)
        final_model.fit(training_data_list[0][selected_features], training_data_list[0][targets[0]])

        # 预测
        y_pred_test = final_model.predict(testing_data_list[0][selected_features])
        if args.excel_log:
            y_pred_test = custom_exp(y_pred_test)

        test_predictions = dict(zip(testing_data_list[0]['pid'], y_pred_test))

        with open(f'{output_dir}/test_predictions.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pid', 'popularity_score'])
            for pid, score in test_predictions.items():
                writer.writerow([pid, score])

        with open(f'{output_dir}/best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        with open(f'{output_dir}/selected_features.json', 'w') as f:
            json.dump(selected_features, f, indent=4)
