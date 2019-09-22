# A General-Purpose Algorithm for Constrained Sequential Inference
This repository contains the code for the CoNLL 2019 paper [A General-Purpose Algorithm for Constrained Sequential Inference](https://cogcomp.seas.upenn.edu/papers/DeutschUpRo19.pdf).

Many current approaches for inference make decisions in a left-to-right manner.
However, there is no general framework to specify constraints in these approaches.  
In this work, we express constraints as an automaton which is traversed in lock-step during inference to guide the search to only valid outputs.
We demonstrate that this framework is able to express many commonly used constraints in NLP and show the benefits of our algorithm on two tasks, constituency parsing and semantic role labelling.

If you use any of the code from this repository, please cite the following paper
```
@inproceedings{DeutschUpRo19,
    author = {Daniel Deutsch and Shyam Upadhyay and Dan Roth},
    title = {{A General-Purpose Algorithm for Constrained Sequential Inference}},
    booktitle = {Proc. of the Conference on Computational Natural Language Learning (CoNLL)},
    year = {2019},
    url = "https://cogcomp.seas.upenn.edu/papers/DeutschUpRo19.pdf"
}
```

## Experiments
Our experiments were based on two tasks, constituency parsing and semantic role labelling.
The constraints for each task are defined as automata under `gcd/inference/constraints`.
We experiment with several different inference approaches: unconstrained inference, a method which applies constraints in a post-hoc manner, and the constrained inference that uses automata proposed in this work.
Each of these three inference algorithms is implemented based on AllenNLP's beam search code and can be found under `gcd/inference/beam_search`.

More details on how to reproduce the results for each task can be found under `experiments/parsing` and `experiments/srl`.

## Installing Dependencies
The Python dependencies which can be directly installed via pip are in `requirements.txt`.
The code also relies on the [Pynini](http://www.openfst.org/twiki/bin/view/GRM/Pynini) library for implementing the automata.
In order to install Pynini, you also need to install OpenFST and Re2.
The following instructions assume that you do not have sudo access.

### OpenFST
```bash
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.9.tar.gz
tar -zxvf openfst-1.6.9.tar.gz
cd openfst-1.6.9
./configure --enable-grm --prefix=/path/to/openfst_install_dir
make
make install
```
For instance, the prefix can be `/home1/s/shyamupa/bin/openfst`.

### Re2
```bash
git clone https://github.com/google/re2
cd re2
git checkout 2018-04-01; git pull
make
make test
make install DESTDIR=/path/to/re2_install_dir
make testinstall
```
For instance the prefix can be `/home1/s/shyamupa/bin/re2`.

You might need to add the following to the `CPATH`, `LD_LIBRARY_PATH`, and `LIBRARY_PATH` to make the installation work properly.
```bash
export CPATH=/home1/s/shyamupa/bin/re2/include
export LD_LIBRARY_PATH=/home1/s/shyamupa/bin/re2/lib
export LIBRARY_PATH=/home1/s/shyamupa/bin/re2/lib

export CPATH=${CPATH}:/home1/s/shyamupa/bin/openfst/include
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home1/s/shyamupa/bin/openfst/lib (it could be lib64 on your machine)
export LIBRARY_PATH=${LIBRARY_PATH}:/home1/s/shyamupa/bin/openfst/lib (it could be lib64 on your machine)
```

### Pynini
```bash
wget http://www.opengrm.org/twiki/pub/GRM/PyniniDownload/pynini-2.0.0.tar.gz
tar -zxvf pynini-2.0.0.tar.gz
cd pynini-2.0.0
python setup.py install
```

### Installation Issues
Here are some common errors and what worked for us to fix them.

#### TypeError: Couldn't build proto file into descriptor pool!
```
pip uninstall protobuf
pip install --no-binary=protobuf protobuf
```
Solution via https://github.com/ValvePython/csgo/issues/8

#### TypeError: Conflict register for file "tensorboard/compat/proto/resource_handle.proto
```
pip uninstall tensorboard
```
Solution via https://github.com/NVIDIA/tacotron2/issues/149
