#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: sh setup.sh <conll2005-path>"
    exit
fi

# CONLL05_PATH="/mnt/castor/seas_home/s/shyamupa/deep_srl/data/srl/conll05st-release/"
CONLL05_PATH=$1
TRAINSET="train-set"
DEVELSET="dev-set"
mkdir -p data/srl/conll2005

if [[ ! -f data/srl/conll2005/${TRAINSET}.jsonl ]]; then
    echo "preparing jsonl file from conll2005 train"
    python -m gcd.data.dataset_setup.srl_convertor --gold_file ${CONLL05_PATH}/${TRAINSET} --core_args_only
    mv ${CONLL05_PATH}/${TRAINSET}.jsonl data/srl/conll2005/
else
    echo "train jsonl already there"
fi
if [[ ! -f data/srl/conll2005/${DEVELSET}.jsonl ]]; then
    echo "preparing jsonl file from conll2005 dev"
    python -m gcd.data.dataset_setup.srl_convertor --gold_file ${CONLL05_PATH}/${DEVELSET} --core_args_only
    mv ${CONLL05_PATH}/${DEVELSET}.jsonl data/srl/conll2005/
    echo "adding parse trees to the dev set json ..."
    python -m gcd.data.dataset_setup.add_parse_trees \
    --in data/srl/conll2005/${DEVELSET}.jsonl \
    --out data/srl/conll2005/${DEVELSET}_parse.jsonl
    mv data/srl/conll2005/${DEVELSET}_parse.jsonl data/srl/conll2005/${DEVELSET}.jsonl
else
    echo "dev jsonl already there"
fi

if [[ ! -d data/srl/propbank-frames ]]; then
    cd data/srl/
    git clone https://github.com/propbank/propbank-frames.git
else
    echo "propbank-frames already there"
fi

if [[ $? == 0 ]]        # success
then
    :                   # do nothing
else                    # something went wrong
    echo "SOME PROBLEM OCCURED";            # echo file with problems
fi
