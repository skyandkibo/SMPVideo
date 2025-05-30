#!/bin/bash

export TMPDIR=../tmp_smp

# 如果目录不存在，则创建
if [ ! -d "$TMPDIR" ]; then
  mkdir -p "$TMPDIR"
fi

# 目标环境路径
ENV_PATH=../mech

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

python merge.py --model train_test --paths MMRA/test_results/test_MMRA_tiktok/test_predictions.csv,machine_learning/machine_learning/catboost/output/importance_True/0.95/test_predictions.csv,machine_learning/machine_learning/lightbgm/output/importance_True/0.85/test_predictions.csv,machine_learning/machine_learning/mlp/output/importance_True/catboost/test_predictions.csv