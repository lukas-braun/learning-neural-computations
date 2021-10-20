////////////////
// Include project headers

#include "inputs.h"


////////////////
// Include libraries

#include <eigen3/Eigen/Dense>
#include <random>
#include <iostream>


////////////////
// Constructor

Poisson::Poisson(int excitatory_n, int inhibitory_n, float excitatory_hz, float inhibitoy_hz, unsigned int seed) {
    this->excitatory_n = excitatory_n;
    this->inhibitory_n = inhibitory_n;
    
    calls_n = 0;
    repeat_steps = 100000;
    
    std::srand(seed);
    
    p1 = 1. - (excitatory_hz / 1000.);
    p2 = 1. - (inhibitoy_hz / 1000.);
    
    xs = Eigen::MatrixXf::Zero(excitatory_n + inhibitory_n, repeat_steps);
}


////////////////
// Methods

// Return input spikes
Eigen::MatrixXf Poisson::sample() {
    if (calls_n % repeat_steps == 0) {
        xs << (Eigen::MatrixXf::Random(excitatory_n, repeat_steps).array() > p1).cast<float>(),
              (Eigen::MatrixXf::Random(inhibitory_n, repeat_steps).array() > p2).cast<float>();
        calls_n = 0;
    }
    
    calls_n++;
    
    return xs.col(calls_n - 1);
}

