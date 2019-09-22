python -m gcd.data.dataset_setup.ptb \
  --train-output data/ptb/train.jsonl \
  --valid-output data/ptb/valid.jsonl \
  --test-output data/ptb/test.jsonl \
  --split-tags true \
  --filter-none true \
  --replace-pos-tags true \
  --merge-closing-paren true
