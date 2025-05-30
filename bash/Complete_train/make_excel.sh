#!/bin/bash

export TMPDIR=../tmp_smp

# 如果目录不存在，则创建
if [ ! -d "$TMPDIR" ]; then
  mkdir -p "$TMPDIR"
fi

# 目标环境路径
ENV_PATH=../mmra

# 获取当前 conda 环境名
current_env=$(basename "$CONDA_PREFIX")

if [ "$current_env" != "base" ]; then
  echo "当前环境是 $current_env，切换到 base 环境..."
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate base
else
  echo "当前已经在 base 环境"
fi

# 检查环境是否存在
if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists at $ENV_PATH. Skipping creation."
else
    echo "Creating environment at $ENV_PATH..."
    conda create -p="$ENV_PATH" python==3.10 pip --yes
fi

# 激活环境
conda activate "$ENV_PATH"

# 安装指定版本的包
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements/mmra_requirement.txt

python make_excel.py