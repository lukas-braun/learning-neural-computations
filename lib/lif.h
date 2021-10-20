#pragma once

////////////////
// Include system libraries

#include <eigen3/Eigen/Dense>
#include "adam.h"
#include "inputs.h"

class Lif {
    private:
        Eigen::MatrixXf f1;
        Eigen::MatrixXf f2;
        float f3;
        
        Eigen::MatrixXf tis;
        Eigen::MatrixXf c1;
        Eigen::MatrixXf c2;
        Eigen::MatrixXf c4;
        Eigen::MatrixXf c5;
        
        float tj;
        float c3;
        float c6;
        
        unsigned long t; 
        unsigned long updated;
        
        bool use_adam;
        float lr;
        
    public:
        // Constructor
        Lif(int excitatory_n, int inhibitory_n, float v_0, float v_reset, float v_thr, float tau_s, float tau_m, bool use_adam, float learning_rate);
        
        // Methods
        void reset();
        bool forward(Eigen::VectorXf input);
        void init_synaptic_weights(float beta, int seed, bool shuffle);
        float set_firing_rate(Poisson *inputs, float target_firing_rate, int seed);
        Eigen::MatrixXf dvdws();
        float dvdtau_s();
        float dvdtau_m();
        float dvdv_reset();
        void training_step(int error_signal, bool ws_, bool tau_s_, bool tau_m_, bool v_reset_, bool use_obf, float surrogate);
        
        // Properties
        int excitatory_n;
        int inhibitory_n;
        float v_0;
        float v_reset;
        float v_thr;
        float tau_s;
        float tau_m;
        Eigen::MatrixXf ws;
        bool spiked;
        
        float beta;
        
        float v;
        Eigen::MatrixXf synaptic_currents;
        float reset_current;
        
        float last_v_reset;
        Eigen::MatrixXf last_ws;
        
        Adam* adam;
};




