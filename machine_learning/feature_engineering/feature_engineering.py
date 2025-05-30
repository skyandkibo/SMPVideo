import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
import holidays
import os
import sys
import torch
from tqdm import tqdm
import argparse
import ast
import json

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(current_script_path))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from function.function import custom_log, set_seed
from moviepy.editor import VideoFileClip

def normalize_column(train, test, column_name):
    max_value = train[column_name].max()
    min_value = train[column_name].min()
    train['normalized_' + column_name] = (train[column_name] - min_value) / (max_value - min_value)
    test['normalized_' + column_name] = (test[column_name] - min_value) / (max_value - min_value)
    return train, test

def extract_ratio_number(row):
    data_ratio = row['video_ratio']
    ratio_number = data_ratio.split("p")[0]
    return pd.Series({
        'ratio_number': int(ratio_number)
    })

def extract_big_music(row):
    data_music = row['music_title']
    big_music = data_music.split("-")[0]
    return pd.Series({
        'big_music': big_music
    })

def extract_is_title_is_suggested(row):
    data_title_string = row['post_content']
    data_suggested_string = ast.literal_eval(row["post_suggested_words"])
    if " ".join(data_title_string.split(","))!= "":
        data_is_title = True
    else:
        data_is_title = False
    
    if len(data_suggested_string) > 0:
        data_is_suggested = True
    else:
        data_is_suggested = False
    
    return pd.Series({
        'is_title': data_is_title,
        'len_title': len(data_title_string.split(",")),
        'is_suggested': data_is_suggested,
        'len_suggested': len(data_suggested_string)
    })

# 语言代码兜底映射表（可以扩展）
LANGUAGE_TO_COUNTRY = {
    'en': 'US',
    'zh': 'CN',
    'ja': 'JP',
    'hi': 'IN',
    'es': 'ES',
    'fr': 'FR',
    'de': 'DE'
}

# Function to extract date components and check for US holidays
def extract_date_components_and_check_holidays(row):
    date_string = row['post_time']
    date_format = '%Y-%m-%d %H:%M'  # 根据你的日期时间格式调整
    # 将字符串转换为 datetime 对象
    date = datetime.strptime(date_string, date_format)

    # 将 datetime 对象转换为 date 对象
    date_date = date.date()

    # 国家优先：post_location，其次 post_text_language 映射
    country_code = row.get('post_location')
    if not country_code or pd.isna(country_code):
        lang = row.get('post_text_language', '')
        country_code = LANGUAGE_TO_COUNTRY.get(lang.lower(), 'US')  # 默认用 US

    # 获取对应国家节假日对象（用 getattr 简化扩展）
    try:
        holiday_class = getattr(holidays, country_code.upper())
        country_holidays = holiday_class()
    except AttributeError:
        country_holidays = holidays.UnitedStates()  # 兜底用美国节假日

    # 判断是否节假日
    is_holiday = date_date in country_holidays

    if 9 <= date.hour < 18:
        time_period = 'Work Time'
    elif 18 <= date.hour < 23:
        time_period = 'Leisure Time'
    else:
        time_period = 'Sleep Time'
    return pd.Series({
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'hour': date.hour,
        'is_holiday': is_holiday,
        'time_period': time_period
    })
# 定义一个函数来获取视频的时长
from moviepy.editor import VideoFileClip
import os
import pandas as pd

# 定义一个函数来获取视频的详细信息
def get_video_details(video_path):
    try:
        clip = VideoFileClip(video_path)
        n_frames = clip.reader.nframes  # 视频总帧数
        frame_rate = clip.fps  # 帧率
        clip.close()
        return n_frames, round(frame_rate)
    except Exception as e:
        print(video_path, e)
        return None, None

def extract_video_details(training_data, testing_data):
    tqdm.pandas(desc=f"Processing train video details")
    # 应用函数并将结果保存到新的DataFrame列中
    training_data['video_details'] = training_data['vid'].progress_apply(
        lambda x: get_video_details(os.path.join(f'{root_path}/train/', f'{x}.mp4'))
    )
    training_data[['total_frames', 'frame_rate']] = pd.DataFrame(training_data['video_details'].tolist(), index=training_data.index)

    tqdm.pandas(desc=f"Processing test video details")
    testing_data['video_details'] = testing_data['vid'].progress_apply(
        lambda x: get_video_details(os.path.join(f'{root_path}/test/', f'{x}.mp4'))
    )
    testing_data[['total_frames', 'frame_rate']] = pd.DataFrame(testing_data['video_details'].tolist(), index=testing_data.index)

    training_data[['ratio_number']] = training_data.apply(extract_ratio_number, axis=1)
    testing_data[['ratio_number']] = testing_data.apply(extract_ratio_number, axis=1)

    training_data['video_is_heng'] = training_data['video_width'] > training_data['video_height']
    testing_data['video_is_heng'] = testing_data['video_width'] > testing_data['video_height']

    training_data['video_area'] = training_data['video_width'] * training_data['video_height']
    testing_data['video_area'] = testing_data['video_width'] * testing_data['video_height']

    training_data['video_height/width'] = training_data['video_height'] / training_data['video_width']
    testing_data['video_height/width'] = testing_data['video_height'] / testing_data['video_width']

    training_data['video_width/height'] = training_data['video_width'] / training_data['video_height']
    testing_data['video_width/height'] = testing_data['video_width'] / testing_data['video_height']

    return training_data, testing_data

def repair_video_details(training_data, testing_data):
    cols = ['total_frames', 'frame_rate']

    # Step 1: 修补训练集（先 groupby(uid) 填补）
    for col in cols:
        training_data[col] = training_data.groupby('uid')[col].transform(lambda x: x.fillna(x.median()))
        training_data[col] = training_data[col].fillna(training_data[col].median())  # fallback 全局中位数

    # Step 2: 提取修补后的训练集 uid 中位数映射
    uid_medians = training_data.groupby('uid')[cols].median()

    # Step 3: 用训练集中的统计量修补测试集
    for col in cols:
        testing_data = testing_data.merge(
            uid_medians[[col]].rename(columns={col: f'{col}_uid_median'}),
            left_on='uid', right_index=True, how='left'
        )
        testing_data[col] = testing_data[col].fillna(testing_data[f'{col}_uid_median'])
        testing_data[col] = testing_data[col].fillna(training_data[col].median())  # fallback 全局
        testing_data.drop(columns=[f'{col}_uid_median'], inplace=True)
    
    return training_data, testing_data

def extract_time(training_data, testing_data):
    # 获取video_create_date的最小值和最大值
    min_date = pd.to_datetime(training_data['post_time']).map(pd.Timestamp.timestamp).min()
    max_date = pd.to_datetime(training_data['post_time']).map(pd.Timestamp.timestamp).max()

    # 对training_data和testing_data中的video_create_date进行0-1归一化
    training_data['post_time_normalized'] = (pd.to_datetime(training_data['post_time']).map(pd.Timestamp.timestamp) - min_date) / (max_date - min_date) 
    testing_data['post_time_normalized'] = (pd.to_datetime(testing_data['post_time']).map(pd.Timestamp.timestamp) - min_date) / (max_date - min_date) 

    # 查看归一化后的数据
    print("Training Data after normalization:")
    print(training_data[['post_time', 'post_time_normalized']].head())

    print("Testing Data after normalization:")
    print(testing_data[['post_time', 'post_time_normalized']].head())

    # Apply functions to extract date components and check holidays
    training_data[['year', 'month', 'day', 'hour', 'is_holiday', 'time_period']] = training_data.apply(extract_date_components_and_check_holidays, axis=1)
    testing_data[['year', 'month', 'day', 'hour', 'is_holiday', 'time_period']] = testing_data.apply(extract_date_components_and_check_holidays, axis=1)

    return training_data, testing_data


def extract_title_suggested(training_data, testing_data):
    training_data[['is_title', 'len_title', 'is_suggested', 'len_suggested']] = training_data.apply(extract_is_title_is_suggested, axis=1)
    testing_data[['is_title', 'len_title', 'is_suggested', 'len_suggested']] = testing_data.apply(extract_is_title_is_suggested, axis=1)

    return training_data, testing_data

def extract_music(training_data, testing_data):
    training_data[['big_music']] = training_data.apply(extract_big_music, axis=1)
    testing_data[['big_music']] = testing_data.apply(extract_big_music, axis=1)

    return training_data, testing_data

def calculate_popularity_score(training_data, testing_data, col_list):
    """
    只使用训练集计算流行度，再映射到训练集和测试集。

    参数：
    - training_data: pd.DataFrame，训练集
    - testing_data: pd.DataFrame，测试集
    - col_list: list of str，需要计算流行度的列名列表

    返回：
    - training_data, testing_data: 添加了 '{col}_popularity' 特征的两个 DataFrame
    """
    for col in col_list:
        # 仅用训练集计算流行度
        col_popularity = training_data[col].value_counts().to_dict()

        # 映射回训练集
        training_data[f'{col}_popularity'] = (
            training_data[col].map(col_popularity)
                            .fillna(0)
                            .astype(int)
        )

        # 映射到测试集：训练集从未出现的映射为 0
        testing_data[f'{col}_popularity'] = (
            testing_data[col].map(col_popularity)
                            .fillna(0)
                            .astype(int)
        )

    return training_data, testing_data


def chajia(csv_data):

    csv_data['likes_per_video'] = csv_data['user_likes_count'] / (np.abs(csv_data['user_video_count']) + 1)
    csv_data['followers_per_following'] = csv_data['user_follower_count'] / (np.abs(csv_data['user_following_count']) + 1)
    csv_data['digg_pre_like_ratio'] = csv_data['user_digg_count'] / (np.abs(csv_data['user_likes_count']) + 1)

    csv_data['log_likes_per_video'] = csv_data['likes_per_video'].apply(custom_log)
    csv_data['log_followers_per_following'] = csv_data['followers_per_following'].apply(custom_log)
    csv_data['log_digg_pre_like_ratio'] = csv_data['digg_pre_like_ratio'].apply(custom_log)

    csv_data['log_likes_per_log_video'] = csv_data['log_user_likes_count'] / (np.abs(csv_data['log_user_video_count']) + 1)
    csv_data['log_followers_per_log_following'] = csv_data['log_user_follower_count'] / (np.abs(csv_data['log_user_following_count']) + 1)
    csv_data['log_digg_pre_log_like_ratio'] = csv_data['log_user_digg_count'] / (np.abs(csv_data['log_user_likes_count']) + 1) 

    return csv_data

root_path = '../dataset/repair_video_train_test'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train and predict video popularity")
    parser.add_argument('--part', type=str, default='all')
    parser.add_argument('--seed', type=int, default=42,
                        help="seed for random number generation")
    args = parser.parse_args()
    
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True) 

    output_dir = f'./output/'
    os.makedirs(output_dir, exist_ok=True)
    
    training_data = pd.read_excel(f'../dataset/train.xlsx', engine='openpyxl', keep_default_na=False).fillna("")
    # 将 NaN 替换为空字符串
    testing_data = pd.read_excel(f'../dataset/test.xlsx', engine='openpyxl', keep_default_na=False).fillna("")
    # 将 NaN 替换为空字符串

    if args.part == 'all' or args.part == 'video_details':
        training_data, testing_data = extract_video_details(training_data, testing_data)
        training_data, testing_data = repair_video_details(training_data, testing_data)

    if args.part == 'all' or args.part == 'time':
        training_data, testing_data = extract_time(training_data, testing_data)

    if args.part == 'all' or args.part == 'title_suggested':
        training_data, testing_data = extract_title_suggested(training_data, testing_data)
    
    if args.part == 'all' or args.part == 'music':
        training_data, testing_data = extract_music(training_data, testing_data)
    
    middle_name_list=['user_following_count','user_follower_count','user_likes_count','user_video_count','user_digg_count','user_heart_count']
    middle_name_list += ['music_duration', 'video_duration']
        
    for middle_name in middle_name_list:
        training_data['log_'+middle_name] = training_data[middle_name].apply(custom_log)
        testing_data['log_'+middle_name] = testing_data[middle_name].apply(custom_log)

    training_data = chajia(training_data)
    testing_data = chajia(testing_data)
   
    # Display some of the updated data to verify
    print("Transformed Training Data Sample:")
    print(training_data.head())
    training_data.to_csv(f'{output_dir}/training_data.csv', index=False)

    print("\nTransformed Testing Data Sample:")
    print(testing_data.head())
    testing_data.to_csv(f'{output_dir}/testing_data.csv', index=False)
        