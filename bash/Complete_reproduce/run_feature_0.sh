#!/bin/bash

# 目标环境路径
ENV_PATH=../mmra

# 检查环境是否存在
if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists at $ENV_PATH. Skipping creation."
else
    echo "Creating environment at $ENV_PATH..."
    conda create -p="$ENV_PATH" python==3.10 pip --yes
fi

# 获取当前 conda 环境名
current_env=$(basename "$CONDA_PREFIX")

if [ "$current_env" != "base" ]; then
  echo "当前环境是 $current_env，切换到 base 环境..."
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate base
else
  echo "当前已经在 base 环境"
fi

# 激活环境
conda activate "$ENV_PATH"

# 安装指定版本的包
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements/mmra_requirement.txt

if [ ! -d "./dataset/repair_video_train_test" ]; then
  python move_train_test.py --src_base ./dataset/repair_video --dst_base ./dataset/repair_video_train_test
fi
python extract_wav.py --input_folder ./dataset/repair_video_train_test/train --output_folder ./dataset/audio/train
python extract_wav.py --input_folder ./dataset/repair_video_train_test/test --output_folder ./dataset/audio/test

cd machine_learning
python feature_engineering/feature_extraction.py
python feature_engineering/classfy_content.py --train_path ./output/training_data.csv --test_path ./output/testing_data.csv --output_path ./output/classified_content.csv
python feature_engineering/audio.py --audio_folder ../dataset/audio/train --output_csv ./output/train_audio.csv
python feature_engineering/audio.py --audio_folder ../dataset/audio/test --output_csv ./output/test_audio.csv
