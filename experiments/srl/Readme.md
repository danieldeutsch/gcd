# SRL Experiments
To setup the SRL experiments, first run the `setup.sh` script under `data/srl/conll2005`.
Then, the `learning-curve` directory has the scripts to train and evaluate the SRL models at different levels of supervision.
All of the scripts should be run from the root of the repository.
Run `train.sh` to train all of the models follwed by `predict.sh` to run inference and evaluate the outputs.
The results can be plotted using the `analyze_results.py` script, which will write data used to produce the plots into a `plots` directory.
