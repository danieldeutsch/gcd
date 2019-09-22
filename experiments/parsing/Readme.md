# Parsing Experiments
To setup the parsing experiments, run the `download-dependencies.sh` script which will download and compile EVALB, the tool for evaluating parse trees.
Then, the `learning-curve` directory contains the code to train and evaluate the parsing model at different levels of supervision.
First, run `train.sh` to train all of the models.
Then, run `predict.sh` to use the three different types of inference (unconstrained, post-hoc, constrained) to generate the parse trees.
Finally, run `evaluate.sh` to compute the formatting and F1 metrics for all of the models and inference algorithms.
The results can be plotted by running the `analyze_results.py` script, which will plot the performances into the `plots` directory.
