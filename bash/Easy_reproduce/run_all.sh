#!/bin/bash

export TMPDIR=../tmp_smp

# 如果目录不存在，则创建
if [ ! -d "$TMPDIR" ]; then
  mkdir -p "$TMPDIR"
fi

set -e  # 遇到任何一个脚本报错就中止执行

bash ./bash/Easy_reproduce/run_mech.sh

bash ./bash/Easy_reproduce/run_mmra.sh

echo "全部执行完毕"
