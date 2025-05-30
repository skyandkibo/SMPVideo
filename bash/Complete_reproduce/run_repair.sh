#!/bin/bash

# 获取当前 conda 环境名
current_env=$(basename "$CONDA_PREFIX")

if [ "$current_env" != "base" ]; then
  echo "当前环境是 $current_env，切换到 base 环境..."
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate base
else
  echo "当前已经在 base 环境"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# 判断 repair 环境是否存在
if conda info --envs | grep -q "^repair\s"; then
  echo "环境 repair 已存在，跳过创建"
else
  echo "环境 repair 不存在，创建环境"
  conda create -n repair python=3.8 pip --yes
fi

# 激活环境
conda activate repair

# 安装 pip 包
pip install tqdm==4.67.1

# 安装 conda-forge 的依赖
conda install -c conda-forge ffmpeg=4.3.2 x264=1\!161.3030=h7f98852_1 --yes

# 运行脚本
python repair.py --input_dir ./dataset/Video --output_dir ./dataset/repair_video
