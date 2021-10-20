#pragma once

////////////////
// Include system libraries

#include <eigen3/Eigen/Dense>

class Poisson {
    private:
        long calls_n;
        long repeat_steps;
        float p1;
        float p2;
        
    public:
        // Constructor
        Poisson(int excitatory_n, int inhibitory_n, float excitatory_hz, float inhibitoy_hz, unsigned int seed);
        
        // Methods
        Eigen::MatrixXf sample();
        
        // Properties
        int excitatory_n;
        int inhibitory_n;
        Eigen::MatrixXf A;
        Eigen::MatrixXf B;
        Eigen::MatrixXf xs;
};

