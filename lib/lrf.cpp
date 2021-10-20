////////////////
// Include project headers

#include "lrf.h"
#include "adam.h"
#include "inputs.h"


////////////////
// Include libraries

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <random>


////////////////
// Constructor

Lrf::Lrf(int excitatory_n, int inhibitory_n, float v_reset, float i_reset, float v_thr, float b, float omega, bool use_adam, float learning_rate) {
    int rows = excitatory_n + inhibitory_n ;
    int cols = 1;
    
    v = 0.;
    
    tis = Eigen::MatrixXf::Zero(rows, cols);
    tis = tis.array() + float(99999.);
    a = Eigen::MatrixXf::Ones(rows, cols);
    c1 = Eigen::MatrixXf::Zero(rows, cols);
    d1 = Eigen::MatrixXf::Zero(rows, cols);
    c2 = Eigen::MatrixXf::Zero(rows, cols);
    d2 = Eigen::MatrixXf::Zero(rows, cols);
    
    tj = 99999.;
    
    this->excitatory_n = excitatory_n;
    this->inhibitory_n = inhibitory_n;
    this->v_reset = v_reset;
    this->i_reset = i_reset;
    this->v_thr = v_thr;
    this->b = b / 1000.;
    this->omega = omega / 1000.;
    this->updated = 9999;
    this->spiked = false;
    
    this->beta = 1.;
    
    ws = Eigen::MatrixXf::Zero(rows, cols);
    
    synaptic_currents = Eigen::MatrixXf::Zero(rows, cols);
    reset_current = 0;
    
    this->use_adam = use_adam;
    if (use_adam) {
        adam = new Adam(learning_rate, 0.9, 0.999, 1e-8);
    }
}


////////////////
// Methods

// Fully reset membrane potential and synapses
void Lrf::reset() {
    v = 0.;
    tis = this->tis.array() * 0. + float(99999.);
    a = a.array() * 0. + 1.;
    c1 = c1.array() * 0.;
    d1 = d1.array() * 0.;
    c2 = c2.array() * 0.;
    d2 = d2.array() * 0.;
    tj = 99999.;
    spiked = false;
    
    synaptic_currents = synaptic_currents.array() * 0.;
    reset_current = 0.;
}


// Perform a simulation step of 1ms forward in time
bool Lrf::forward(Eigen::VectorXf input) {
    // Advance time
    tis.array() += 1.;
    tj += 1.;
    updated += 1;
    
    // Update event-based parameters
    for (size_t i = 0, size = input.size(); i < size; i++) {
        if ((*(input.data() + i)) == 1) {
            a(i, 0) = exp(tis(i, 0) * b);
            c1(i, 0) = a(i, 0) * c1(i, 0) + cos(tj * omega);
            d1(i, 0) = a(i, 0) * d1(i, 0) + sin(tj * omega);
            c2(i, 0) = a(i, 0) * c2(i, 0) + cos(tj * omega) * tj;
            d2(i, 0) = a(i, 0) * d2(i, 0) + sin(tj * omega) * tj;
            
            tis(i, 0) = 0.;
        }
    }
    
    if (spiked) {
        a = (a.array() * 0.) + 1.;
        c1 = c1.array() * 0.;
        d1 = d1.array() * 0.;
        c2 = c2.array() * 0.;
        d2 = d2.array() * 0.;
        tj = 0;
    }
    
    // Calculate synaptic current
    synaptic_currents = exp(tis.array() * b) * (c1.array() * sin(tj * omega) - d1.array() * cos(tj * omega));
    
    // Calculate spike reset current
    reset_current = exp(tj * b) * (v_reset * cos(tj * omega) + i_reset * sin(tj * omega));
    
    // Calculate membrane potential
    v = (ws.array() * synaptic_currents.array()).sum() + reset_current;
    
    // Spike detection
    spiked = v >= v_thr;
    
    return spiked;
}


// Sample random excitatory and inhibitory synapses for a given beta value
void Lrf::init_synaptic_weights(float beta, int seed, bool shuffle) {
    int weights_n = excitatory_n + inhibitory_n;
    
    this->beta = beta;
    float x_hat = -atan(omega / b) / omega;
    float kappa = 1. / (exp(x_hat * b) * sin(x_hat * omega));

    std::default_random_engine rng(seed);
    
    float mean_psp = 1. / 20.;
    float variance_psp = pow(1. / 25., 2);
    float mean_corrected = log(pow(mean_psp, 2) / sqrt(pow(mean_psp, 2) + variance_psp));
    float var_corrected = log(1. + variance_psp / pow(mean_psp, 2));
    
    std::lognormal_distribution<float> log_normal(mean_corrected, sqrt(var_corrected));
    
    std::vector<int> idx(weights_n, 0);
    for(int i = 0; i < weights_n; i++) {
        idx[i] = i;
    }
    if (shuffle) std::shuffle(idx.begin(), idx.end(), rng);
    
    float sign = 1.;
    float w = 0.;
    for (int row = 0; row < weights_n; ++row) {
        if (row == excitatory_n) {
            sign = -1.;
            beta = 1.;
        }
        do {
            w = beta * log_normal(rng);
        } while(w > 0.3);
        ws(idx[row], 0) = w * kappa * sign;
    }
}


// Try to find a beta value, such that a target firing rate is reached
float Lrf::set_firing_rate(Poisson *inputs, float target_firing_rate, int seed) {
    float beta_ = 1.;
    float firing_rate = -99999.;
    int last_update_direction = 0;
    float delta_beta = 1.;
    float neuron_out = 0.;
    unsigned int spikes_n = 0;
    
    while (target_firing_rate - firing_rate < -0.05 or target_firing_rate - firing_rate > 0.05) {
        if (firing_rate < target_firing_rate) {
            if (last_update_direction != 1) delta_beta = delta_beta / 2.;
            beta_ = beta_ + delta_beta;
            last_update_direction = 1;
        } else {
            if (last_update_direction != -1) delta_beta = delta_beta / 2.;
            beta_ = beta_ - delta_beta;
            last_update_direction = -1;
        }
        
        this->init_synaptic_weights(beta_, seed, false);
        spikes_n = 0;
        for (unsigned int i = 0; i < 1000 * 1000; i++) {
            neuron_out = float(this->forward(inputs->sample()));
            if (neuron_out) spikes_n += 1;
        }
        firing_rate = float(spikes_n) / 1000.;
        if (this->beta > 3.5) break;
        if (abs(delta_beta) < 0.0001) break;
    }
    return firing_rate;
}


// Derivative with respect to the synaptic weights
Eigen::MatrixXf Lrf::dvdws() {
    return synaptic_currents;
}


// Derivative with respect to the damping factor
float Lrf::dvdb() {
    Eigen::MatrixXf pre_tis = Eigen::exp(tis.array() * b);
    Eigen::MatrixXf part1 = tj * synaptic_currents.array() - pre_tis.array() * (c2.array() * sin(tj * omega) - d2.array() * cos(tj * omega));
    float part2 = (ws.array() * part1.array()).sum();
    float part3 = tj * reset_current;
    return part2 + part3;
}


// Derivative with respect to the frequency of the subthreshold oscillations
float Lrf::dvdomega() {
    Eigen::MatrixXf pre_tis = Eigen::exp(tis.array() * b);
    Eigen::MatrixXf part1 = tj * pre_tis.array() * (c1.array() * cos(tj * omega) + d1.array() * sin(tj * omega)) - pre_tis.array() * (c2.array() * cos(tj * omega) + d2.array() * sin(tj * omega));
    float part2 = (ws.array() * part1.array()).sum();
    float part3 = tj * exp(tj * b) * (i_reset * cos(tj * omega) - v_reset * sin(tj * omega));
    return part2 + part3;
}


// Derivative with respect to the voltage reset
float Lrf::dvdv_reset() {
    return exp(tj * b) * cos(tj * omega);
}


// Derivative with respect to the current reset
float Lrf::dvdi_reset() {
    return exp(tj * b) * sin(tj * omega);
}


// Perform a graident update step upon the occurrence of an error
void Lrf::training_step(int error_signal, bool ws_, bool b_, bool omega_, bool v_reset_, bool i_reset_, bool use_obf, float surrogate_beta) {
    float obf = 1.;
    // Event-dependent scaling factor
    if (use_obf) {
       if (updated > 75) updated = 75;
       obf = 1000 * (1. - exp(log(0.5) * pow(updated / 500., 4.)));
    }
    // Surrogate gradient
    else if (surrogate_beta > 0.) {
        if (error_signal == 1.) obf = 1.;
        else obf = 1. / pow((surrogate_beta * abs(v - 1.)) + 1., 2);
    }
    // Vanilla
    else {
        obf = 1.;
    }
    
    updated = 0;
    
    // Init adam
    if (use_adam) {
        if (adam->t == 1) {
            if (ws_) adam->add(&this->ws);
            if (b_) adam->add(&this->b);
            if (omega_) adam->add(&this->omega);
            if (v_reset_) adam->add(&this->v_reset);
            if (i_reset_) adam->add(&this->i_reset);
        }
    }
    
    // Calculate gradients
    Eigen::MatrixXf dvdws_ = obf * error_signal * dvdws();
    float dvdb_ = obf * error_signal * dvdb();
    float dvdomega_ = obf * error_signal * dvdomega();
    float dvdv_reset_ = obf * error_signal * dvdv_reset();
    float dvdi_reset_ = obf * error_signal * dvdi_reset();
    
    // Apply gradients
    if (ws_) {
        if (use_adam) adam->train(&ws, dvdws_, .8);
        else ws -= 0.5 * 1e-4 * dvdws_;
    }
    
    if (b_) {
        if (use_adam) adam->train(&b, dvdb_, 150./1000.);
        else b -= 2.5 * 1e-4 * dvdb_;
    }
    if (b > -0.01 / 1000.) b = -0.001 / 1000.;
    
    if (omega_) {
        if (use_adam) adam->train(&omega, dvdomega_, 33./1000);
        else omega -= 5. * 1e-4 * dvdomega_;
    }
    
    if (v_reset_) {
        if (use_adam) adam->train(&v_reset, dvdv_reset_, .8);
        else v_reset -= 1. * 1e-4 * dvdv_reset_;
    }
    if (v_reset > 0.99) v_reset = 0.99; 
    
    if (i_reset_) {
        if (use_adam) adam->train(&i_reset, dvdi_reset_, .8);
        else i_reset -= 1. * 1e-4 * dvdi_reset_;
    }
}





