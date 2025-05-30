import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import argparse
import logging

error_list=[]
def extract_audio(input_file, output_file):
    """
    从指定的视频文件中提取音频，并保存为 WAV 文件。
    :param input_file: 输入的视频文件路径
    :param output_file: 输出的音频文件路径
    """
    video = VideoFileClip(input_file)
    audio = video.audio
    if audio is not None:
        audio.write_audiofile(output_file)
    else:
        error_list.append(input_file)
        print(f"No audio found in {input_file}")

def extract_audio_from_folder(input_folder, output_folder):
    """
    从指定文件夹中的所有视频文件中提取音频，并保存到输出文件夹中。
    :param input_folder: 包含视频文件的输入文件夹路径
    :param output_folder: 保存音频文件的输出文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder)):
        # 检查文件是否是视频文件（可以根据需要扩展支持更多格式）
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_file = os.path.join(input_folder, filename)
            # 生成输出文件名（将视频文件扩展名替换为 .wav）
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')
            extract_audio(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from video files in a folder.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the input folder containing video files.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder to save audio files.")
    args = parser.parse_args()

    extract_audio_from_folder(args.input_folder, args.output_folder)
    print(error_list)