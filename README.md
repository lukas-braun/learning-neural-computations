# Online Learning Of Neural Computations From Sparse Temporal Feedback

This repository is the official implementation of the [NeuroIPS 2021](https://neurips.cc/Conferences/2021) paper [Online Learning Of Neural Computations From Sparse Temporal Feedback]().

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

