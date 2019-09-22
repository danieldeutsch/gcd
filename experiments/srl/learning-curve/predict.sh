#!/usr/bin/env bash
expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

results_dir="${expt_dir}/results"
output_dir="${expt_dir}/output"
mkdir -p ${output_dir}

data_dir=data/srl/conll2005

function run_model() {

    nodup=$1
    argcands=$2
    legal=$3
    full=$4
    beam=$5

    outfile=${output_dir}/${split}.${percent}.FULL_${full}_BEAM_${beam}.NODUP_${nodup}_LEGAL_${legal}_ARGCANDS_${argcands}.jsonl
    metfile=${output_dir}/${split}.${percent}.FULL_${full}_BEAM_${beam}.NODUP_${nodup}_LEGAL_${legal}_ARGCANDS_${argcands}.metrics.json

    if [ "${full}" == "true" ]; then
      method="full_intersection"
    else
      method="active_set"
    fi

    allennlp predict \
             --include-package gcd \
             --use-dataset-reader \
             --predictor srl-tagger \
             --overrides "{\"model.beam_search\": {\"type\": \"${method}\", \"beam_size\": ${beam}, \"namespace\": \"labels\"}, \"model.use_argument_candidates\":${argcands},\"model.use_no_duplicates_constraint\": ${nodup}, \"model.use_disallow_arg\": ${legal}}" \
             --output-file ${outfile} \
             ${model_dir}/model.tar.gz \
             ${data_dir}/${split}.jsonl

    python -m gcd.metrics.srl \
      --model ${outfile} \
      --gold ${data_dir}/${split}.jsonl \
      --output-file ${metfile}
}

for percent in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
    work_dir="${results_dir}/${percent}"
    model_dir="${work_dir}/model"

    for split in dev-set; do
        for BEAM in 1; do
            ##################
            FULL="false"

            NODUP="true"
            ARGCANDS="true"
            LEGAL="true"
            run_model ${NODUP} ${ARGCANDS} ${LEGAL} ${FULL} ${BEAM}

            NODUP="true"
            ARGCANDS="false"
            LEGAL="true"
            run_model ${NODUP} ${ARGCANDS} ${LEGAL} ${FULL} ${BEAM}

            NODUP="true"
            ARGCANDS="false"
            LEGAL="false"
            run_model ${NODUP} ${ARGCANDS} ${LEGAL} ${FULL} ${BEAM}

            NODUP="false"
            ARGCANDS="false"
            LEGAL="false"
            run_model ${NODUP} ${ARGCANDS} ${LEGAL} ${FULL} ${BEAM}

            ##################
            FULL="true"
            METFILE=${output_dir}/${split}.${percent}.FULL_${FULL}_BEAM_${BEAM}.metrics.json

            NODUP="true"
            ARGCANDS="true"
            LEGAL="true"
            run_model ${NODUP} ${ARGCANDS} ${LEGAL} ${FULL} ${BEAM}
        done
    done
done
