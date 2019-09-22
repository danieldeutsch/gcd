expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
results_dir=${expt_dir}/results

for percent in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  work_dir="${results_dir}/${percent}"
  model_dir="${work_dir}/model"
  output_dir="${work_dir}/output"
  mkdir -p ${output_dir}

  for split in test; do
    allennlp predict \
      --include-package gcd \
      --predictor parsing_predictor \
      --output-file ${output_dir}/${split}.unconstrained.jsonl \
      --overrides '{"model.beam_search": {"type": "unconstrained", "beam_size": 10, "namespace": "nonterminals"}}' \
      ${model_dir}/model.tar.gz \
      ${work_dir}/${split}.jsonl

    allennlp predict \
       --include-package gcd \
       --predictor parsing_predictor \
       --output-file ${output_dir}/${split}.constrained.jsonl \
       --overrides '{"model.beam_search": {"type": "active_set", "beam_size": 10, "namespace": "nonterminals"}}' \
       ${model_dir}/model.tar.gz \
       ${work_dir}/${split}.jsonl

    allennlp predict \
      --include-package gcd \
      --predictor parsing_predictor \
      --output-file ${output_dir}/${split}.post-hoc.jsonl \
      --overrides '{"model.beam_search": {"type": "post_hoc", "beam_size": 10, "namespace": "nonterminals"}}' \
      ${model_dir}/model.tar.gz \
      ${work_dir}/${split}.jsonl
  done
done
