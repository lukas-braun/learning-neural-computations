#pragma once

////////////////
// Include system libraries

#include <eigen3/Eigen/Dense>
#include "adam.h"
#include "inputs.h"

class Lrf {
    private:
        Eigen::MatrixXf tis;
        Eigen::MatrixXf a;
        Eigen::MatrixXf c1;
        Eigen::MatrixXf d1;
        Eigen::MatrixXf c2;
        Eigen::MatrixXf d2;
        
        float tj;
        
        unsigned long updated;
        
        bool use_adam;
        Adam* adam;
        
    public:
        // Constructor
        Lrf(int excitatory_n, int inhibitory_n, float v_reset, float i_reset, float v_thr, float b, float omega, bool use_adam, float learning_rate);
        
        // Methods
        void reset();
        bool forward(Eigen::VectorXf input);
        void init_synaptic_weights(float beta, int seed, bool shuffle);
        float set_firing_rate(Poisson *inputs, float target_firing_rate, int seed);
        Eigen::MatrixXf dvdws();
        float dvdb();
        float dvdomega();
        float dvdv_reset();
        float dvdi_reset();
        void training_step(int error_signal, bool ws_, bool b_, bool omega_, bool v_reset_, bool i_reset_, bool use_obf, float surrogate_beta);
        
        // Properties
        int excitatory_n;
        int inhibitory_n;
        float v_reset;
        float i_reset;
        float v_thr;
        float b;
        float omega;
        Eigen::MatrixXf ws;
        
        bool spiked;
        float beta;
        
        float v;
        Eigen::MatrixXf synaptic_currents;
        float reset_current;
        
        float m;
        float exponential;
};




