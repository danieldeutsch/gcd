expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
results_dir=${expt_dir}/results
model_config=${expt_dir}/model.jsonnet

if [ -d ${results_dir} ]; then
  read -p "remove directory ${results_dir}? [y/n] " yn
  case $yn in
        [Yy]* ) rm -rf ${results_dir};;
        [Nn]* ) ;;
        * ) echo "Please answer yes or no.";;
  esac
fi

for percent in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  work_dir="${results_dir}/${percent}"
  model_dir="${work_dir}/model"
  mkdir -p ${work_dir}

  python -m gcd.data.dataset_setup.sample_data data/ptb/train.jsonl ${percent} ${work_dir}/train.jsonl
  python -m gcd.data.dataset_setup.sample_data data/ptb/valid.jsonl ${percent} ${work_dir}/valid.jsonl
  cp data/ptb/test.jsonl ${work_dir}/test.jsonl
  export TRAIN_PATH="${work_dir}/train.jsonl"
  export VALID_PATH="${work_dir}/valid.jsonl"

  allennlp train \
    --include-package gcd \
    -s ${model_dir} \
    ${model_config}
done
