# Parsing Experiments
The parsing experiments use the Penn Treebank for training and evaluating, so you first need to setup the data using the instructions under `data/ptb`.

In the following instructions, all of the scripts should be run from the root of the repository (e.g., `sh experiments/parsing/download-dependencies.sh` instead of cd-ing into the directory and running `download-dependencies.sh`).
To setup the parsing experiments, run the `download-dependencies.sh` script which will download and compile EVALB, the tool for evaluating parse trees.
Then, the `learning-curve` directory contains the code to train and evaluate the parsing model at different levels of supervision.
First, run `train.sh` to train all of the models.
Then, run `predict.sh` to use the three different types of inference (unconstrained, post-hoc, constrained) to generate the parse trees.
Finally, run `evaluate.sh` to compute the formatting and F1 metrics for all of the models and inference algorithms.
The results can be plotted by running `python path/to/analyze_results.py`, which will plot the performances into the `plots` directory.
