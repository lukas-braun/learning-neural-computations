# Online Learning Of Neural Computations From Sparse Temporal Feedback

This repository is the official implementation of the [NeurIPS 2021](https://neurips.cc/Conferences/2021) paper [Online Learning Of Neural Computations From Sparse Temporal Feedback](https://openreview.net/pdf?id=nJUDGEc69a5).

## Requirements

Experiments are implemented in C++ using the [Eigen software library](https://eigen.tuxfamily.org), which can be install via 

```eigensetup
sudo apt install libeigen3-dev
```

For plotting we are using python and jupyter notebooks. To install all requirements, run

```setup
pip3 install -r requirements.txt
```

## Running experiments

In order to replicate one of the experiments, navigate to the respective folder (e.g. ./figure3/a) and run 

```train
g++ ./lib/lif.cpp ./lib/lrf.cpp ./lib/inputs.cpp ./lib/adam.cpp ./lib/logger.cpp experiment.cpp -o experiment -O3 && ./experiment
```

this will compile all necessary files and execute the binary. Results are stored as .csv files in the respective results folders (e.g. ./figure3/a/results). Once the experiment terminates, you can plot the results using the ipython notebooks provided in the figure's main folder (e.g. ./figure3/figure-3.ipynb).

By default, most scripts start 30 processes to run the experiment from 30 different random seeds. If this is too much for your hardware or you would like to increase the amount of seeds, you can adjust the number by changing 

```processes
#define SEEDS_N 30
```

on the top of the respective experiment.cpp file to an appropriate number.

## Citing

If you find the implementation or any of the plots useful and you use it, please cite:

Lukas Braun, & Tim P. Vogels (2021). Online Learning Of Neural Computations From Sparse Temporal Feedback. In _Thirty-Fifth Conference on Neural Information Processing Systems_.

Url: https://openreview.net/forum?id=nJUDGEc69a5

Bibtex:
```
@inproceedings{
    braun2021online,
    title={Online Learning Of Neural Computations From Sparse Temporal Feedback},
    author={Lukas Braun and Tim P. Vogels},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021},
    url={https://openreview.net/forum?id=nJUDGEc69a5}
}
```
