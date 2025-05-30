SKIP_STEPS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --skip)
      shift
      # collect all non-flag args
      while [[ "$#" -gt 0 && "$1" != --* ]]; do
        SKIP_STEPS+=("$1")
        shift
      done
      ;;
    *)
      # other flags / positionals
      shift
      ;;
  esac
done

export TMPDIR=../tmp_smp

# 如果目录不存在，则创建
if [ ! -d "$TMPDIR" ]; then
  mkdir -p "$TMPDIR"
fi

set -e  # 遇到任何一个脚本报错就中止执行

bash ./bash/Complete_train/make_excel.sh

if [[ " ${SKIP_STEPS[@]} " =~ " preprocess " ]]; then
  echo "Skipping preprocess step..."
else
  echo "Running preprocess step..."
  bash ./bash/Complete_train/run_repair.sh
fi

bash ./bash/Complete_train/run_feature_0.sh

bash ./bash/Complete_train/run_feature_1.sh

bash ./bash/Complete_train/run_feature_2.sh

echo "全部执行完毕"