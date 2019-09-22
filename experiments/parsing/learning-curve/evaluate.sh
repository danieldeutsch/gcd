expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
results_dir=${expt_dir}/results

for percent in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  work_dir="${results_dir}/${percent}"
  output_dir="${work_dir}/output"
  metrics_dir="${work_dir}/metrics"
  mkdir -p ${metrics_dir}

  for model in unconstrained constrained post-hoc; do
    for split in test; do
      python -m gcd.metrics.parsing \
        ${work_dir}/${split}.jsonl \
        ${output_dir}/${split}.${model}.jsonl \
        > ${metrics_dir}/${split}.${model}.json
    done
  done
done
