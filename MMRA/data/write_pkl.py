import pandas as pd
import os
import ast
import argparse
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from moviepy.editor import VideoFileClip

features = ['log_user_following_count', 'len_suggested', 'video_height', 'post_location', 'video_width', 'day', 'uid', 'video_duration', 'post_text_language', 'hour', 'log_user_follower_count', 'len_title', 'music_duration', 'big_music', 'log_user_likes_count', 'frame_rate', 'is_title', 'year', 'post_time_normalized', 'log_user_video_count', 'is_suggested', 'total_frames', 'log_user_digg_count']    

def extract_date_components_and_check_holidays(row):
    date_string = row['post_time']
    date_format = '%Y-%m-%d %H:%M'  # 根据你的日期时间格式调整
    # 将字符串转换为 datetime 对象
    date = datetime.strptime(date_string, date_format)

    date_date = date.date()

    date = datetime.strptime(date_string, date_format)

    return pd.Series({
        'year': date.year,
        'day': date.day,
        'hour': date.hour
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

def extract_big_music(row):
    data_music = row['music_title']
    big_music = data_music.split("-")[0]
    return pd.Series({
        'big_music': big_music
    })

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

def buchong(train_excel, test_excel):
    train_excel['log_user_following_count'] = np.log(1+abs(train_excel['user_following_count']))
    test_excel['log_user_following_count'] = np.log(1+abs(test_excel['user_following_count']))
    max_year = pd.to_datetime(train_excel['post_time']).dt.year.max()
    max_datetime = datetime(year=max_year, month=12, day=31, hour=23, minute=59)
    max_date = max_datetime.timestamp()
    min_date = pd.to_datetime(train_excel['post_time']).map(pd.Timestamp.timestamp).min()
    train_excel['post_time_normalized'] = (pd.to_datetime(train_excel['post_time']).map(pd.Timestamp.timestamp) - min_date) / (max_date - min_date) 
    test_excel['post_time_normalized'] = (pd.to_datetime(test_excel['post_time']).map(pd.Timestamp.timestamp) - min_date) / (max_date - min_date) 
    train_excel[['year', 'day', 'hour']] = train_excel.apply(extract_date_components_and_check_holidays, axis=1)
    test_excel[['year', 'day', 'hour']] = test_excel.apply(extract_date_components_and_check_holidays, axis=1)
    train_excel['log_user_follower_count'] = np.log(1+abs(train_excel['user_follower_count']))
    test_excel['log_user_follower_count'] = np.log(1+abs(test_excel['user_follower_count']))
    train_excel[['is_title', 'len_title', 'is_suggested', 'len_suggested']] = train_excel.apply(extract_is_title_is_suggested, axis=1)
    test_excel[['is_title', 'len_title', 'is_suggested', 'len_suggested']] = test_excel.apply(extract_is_title_is_suggested, axis=1)
    train_excel[['big_music']] = train_excel.apply(extract_big_music, axis=1)
    test_excel[['big_music']] = test_excel.apply(extract_big_music, axis=1)
    train_excel['log_user_likes_count'] = np.log(1+abs(train_excel['user_likes_count']))
    test_excel['log_user_likes_count'] = np.log(1+abs(test_excel['user_likes_count']))
    tqdm.pandas(desc=f"Processing train video total_frames and frame_rate")
    train_excel['video_details'] = train_excel['vid'].progress_apply(
        lambda x: get_video_details(os.path.join(f'../dataset/repair_video_train_test/train/', f'{x}.mp4'))
    )
    train_excel[['total_frames', 'frame_rate']] = pd.DataFrame(train_excel['video_details'].tolist(), index=train_excel.index)
    tqdm.pandas(desc=f"Processing test video total_frames and frame_rate")
    test_excel['video_details'] = test_excel['vid'].progress_apply(
        lambda x: get_video_details(os.path.join(f'../dataset/repair_video_train_test/test/', f'{x}.mp4'))
    )
    test_excel[['total_frames', 'frame_rate']] = pd.DataFrame(test_excel['video_details'].tolist(), index=test_excel.index)
    columns_to_fill = ['total_frames', 'frame_rate']
    medians = train_excel.groupby('uid')[columns_to_fill].median()
    for column in columns_to_fill:
        train_excel[column] = train_excel.groupby('uid')[column].transform(lambda x: x.fillna(x.median()))
    train_excel['log_user_video_count'] = np.log(1+abs(train_excel['user_video_count']))
    test_excel['log_user_video_count'] = np.log(1+abs(test_excel['user_video_count']))
    train_excel['log_user_digg_count'] = np.log(1+abs(train_excel['user_digg_count']))
    test_excel['log_user_digg_count'] = np.log(1+abs(test_excel['user_digg_count']))
    return train_excel, test_excel

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

def text_final_easy_excel(row):
    post_content = " ".join(row['post_content'].split(','))
    post_suggested_words = ast.literal_eval(row["post_suggested_words"])
        
    data_list = []
    
    for feature in features:
        data_list.append(row[feature])
    
    prompt = ",".join([str(i) if type(i)!=str else i  for i in data_list])

    prompt = f"{post_content}, {post_suggested_words}, {prompt}" 

    return prompt

def write_pkl(excel_data, output, text_type):
    # 将 NaN 替换为空字符串
    excel_data = excel_data.fillna("")
    
    vid, title, label = [], [], []
    for x in excel_data.iterrows():
        vid.append(x[1]['vid'])
        
        title.append(text_final_easy_excel(x[1]))

        if output == 'train':
            label.append(x[1]['popularity'])

    if output == 'train':
        data = {
            'item_id': vid,
            'text': title,
            'label': label
        }
    else:
        data = {
            'item_id': vid,
            'text': title
        }

    df = pd.DataFrame(data)

    os.makedirs(f"./datasets/tiktok/{args.text_type}", exist_ok=True)

    # 将DataFrame保存为Pickle文件
    df.to_pickle(f'./datasets/tiktok/{args.text_type}/{output}.pkl')

    print(f"数据帧已成功保存为./datasets/tiktok/{args.text_type}/{output}.pkl文件。")
    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Write Excel data to Pickle files")
    parser.add_argument('--text_type', type=str, default='final_easy_excel', help='Text type')
    args = parser.parse_args()

    train_excel = pd.read_excel(f'../dataset/train.xlsx' , engine='openpyxl', keep_default_na=False).fillna("")
    test_excel = pd.read_excel(f'../dataset/test.xlsx' , engine='openpyxl', keep_default_na=False).fillna("")

    train_excel, test_excel = buchong(train_excel, test_excel)
    
    write_pkl(train_excel, "train",  args.text_type)
    write_pkl(test_excel, "test", args.text_type)
