local train_path = std.extVar("TRAIN_PATH");
local valid_path = std.extVar("VALID_PATH");
local test_path = std.extVar("TEST_PATH");
local propbank_root = std.extVar("PROPBANK_DIR");

{
  "dataset_reader": {
    "type": "conll05_srl",
    "tag_label": "srl",
    "core_args_only": true,
    "coding_scheme": "BIO",
    "year": "2005",
    "propbank_root": propbank_root
  },
  "train_data_path": train_path,
  "validation_data_path": valid_path,
  "test_data_path": test_path,
  "vocabulary": {
    "only_include_pretrained_words": true,
    "pretrained_files": {
      "tokens": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"
    },
    "tokens_to_add": {
      "labels": ["@start@", "@end@"]
    }
  },
  "model": {
    "type": "constrained_srl",
    "ignore_span_metric": false,
    "use_no_duplicates_constraint": true,
    "use_disallow_arg": true,
    "use_argument_candidates": false,
    "beam_search": {
      "type": "unconstrained",
      "beam_size": 1,
      "namespace": "labels"
    },
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "embedding_dim": 100,
        "trainable": true,
      },
    },
    "initializer": [
      ["tag_projection_layer.*weight", {"type": "orthogonal"}]
    ],
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 200,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1,
      "use_highway": true
    },
    "binary_feature_dim": 100,
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 80
  },
  "trainer": {
    "num_epochs": 500,
    "grad_clipping": 1.0,
    "patience": 20,
    "validation_metric":"+f1-measure-overall",
    "cuda_device": 0,
    "shuffle": true,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95,
    },
    "num_serialized_models_to_keep": 1
  },
  "random_seed": 0,
  "numpy_seed": 0,
  "pytorch_seed": 0,
}
