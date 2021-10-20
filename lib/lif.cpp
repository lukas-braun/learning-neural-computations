////////////////
// Include project headers

#include "lif.h"
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

Lif::Lif(int excitatory_n, int inhibitory_n, float v_0, float v_reset, float v_thr, float tau_s, float tau_m, bool use_adam, float learning_rate) {
    int rows = excitatory_n + inhibitory_n ;
    int cols = 1;
    
    v = v_0;
    
    f1 = Eigen::MatrixXf::Zero(rows, cols);
    f2 = Eigen::MatrixXf::Zero(rows, cols);
    f3 = 0.;
    
    tis = Eigen::MatrixXf::Zero(rows, cols);
    tis = tis.array() + float(99999.);
    c1 = Eigen::MatrixXf::Ones(rows, cols);
    c2 = Eigen::MatrixXf::Ones(rows, cols);
    c3 = 1.;
    c4 = Eigen::MatrixXf::Ones(rows, cols);
    c5 = Eigen::MatrixXf::Ones(rows, cols);
    c6 = 1.;
    
    tj = 99999.;
    
    this->excitatory_n = excitatory_n;
    this->inhibitory_n = inhibitory_n;
    this->v_0 = v_0;
    this->v_reset = v_reset;
    this->v_thr = v_thr;
    this->tau_s = tau_s;
    this->tau_m = tau_m;
    this->t = 0;
    this->updated = 9999;
    this->spiked = false;
    
    this->beta = 1.;
    
    ws = Eigen::MatrixXf::Zero(rows, cols);
    
    synaptic_currents = Eigen::MatrixXf::Zero(rows, cols);
    reset_current = 0;
    
    this->lr = learning_rate;
    this->use_adam = use_adam;
    if (use_adam) {
        adam = new Adam(learning_rate, 0.9, 0.999, 1e-8);
    }
}


////////////////
// Methods

// Fully reset membrane potential and synapses
void Lif::reset() {
    v = 0.;
    f1 = f1.array() * 0.;
    f2 = f2.array() * 0.;
    f3 = 0.;
    tis = tis.array() * 0. + float(99999.);
    c1 = c1.array() * 0. + 1.;
    c2 = c2.array() * 0. + 1.;
    c3 = 1.;
    c4 = c4.array() * 0. + 1.;
    c5 = c5.array() * 0. + 1.;
    c6 = 1.;
    tj = 99999.;
    spiked = false;
    
    synaptic_currents = synaptic_currents.array() * 0.;
    reset_current = 0.;
}


// Perform a simulation step of 1ms forward in time
bool Lif::forward(Eigen::VectorXf input) {
    // Advance time
    tis.array() += 1.;
    tj += 1.;
    updated += 1;
    
    f1 = Eigen::exp(-1 * tis.array() / tau_m);
    f2 = Eigen::exp(-1 * tis.array() / tau_s);
    
    for (size_t i = 0, size = input.size(); i < size; i++) {
        if ((*(input.data() + i)) == 1) {
            // For synaptic currents
            c1(i, 0) = f1(i, 0) * c1(i, 0) + 1.;
            c2(i, 0) = f2(i, 0) * c2(i, 0) + 1.;
            
            // For tau_s gradients
            c4(i, 0) = f2(i, 0) * c4(i, 0) + t;
            
            // For tau_m gradients
            c5(i, 0) = f1(i, 0) * c5(i, 0) + t;
            
            tis(i, 0) = 0.;
            f1(i, 0) = 1.;
            f2(i, 0) = 1.;
        }
    }
    
    if (spiked) {
        // For reset current
        c3 = f3 * c3 + 1.;
        
        // For tau_m gradients
        c6 = f3 * c6 + t;
        
        tj = 0;
    }
    
    // Update f3
    f3 = exp(-1 * tj / tau_m);
    
    // Calculate synaptic current
    synaptic_currents = f1.array() * c1.array() - f2.array() * c2.array();
    
    // Calculate spike reset current
    reset_current = (v_reset - v_thr) * f3 * c3;
    
    // Calculate membrane potential
    v = v_0 + (ws.array() * synaptic_currents.array()).sum() + reset_current;
    
    // Spike detection
    spiked = v >= v_thr;
    
    t += 1;
    
    return spiked;
}


// Sample random excitatory and inhibitory synapses for a given beta value
void Lif::init_synaptic_weights(float beta, int seed, bool shuffle) {
    int weights_n = excitatory_n + inhibitory_n;
    
    this->beta = beta;
    float x_hat = log(tau_s / tau_m) * (tau_s * tau_m) / (tau_s - tau_m);
    float kappa = 1. / (exp(-x_hat / tau_m) - exp(-x_hat / tau_s));

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
    for (int row = 0; row < (excitatory_n + inhibitory_n); ++row) {
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
float Lif::set_firing_rate(Poisson *inputs, float target_firing_rate, int seed) {
    float beta_ = 1.;
    float firing_rate = -1;
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
Eigen::MatrixXf Lif::dvdws() {
    return synaptic_currents;
}


// Derivative with respect to the synaptic time constant
float Lif::dvdtau_s() {
    Eigen::MatrixXf a = t * f2.array() * c2.array();
    Eigen::MatrixXf b = f2.array() * c4.array();
    return -1. / pow(tau_s, 2) * (ws.array() * (a.array() - b.array())).sum();
}


// Derivative with respect to the membrane time constant
float Lif::dvdtau_m() {
    Eigen::MatrixXf c = t * f1.array() * c1.array();
    Eigen::MatrixXf d = f1.array() * c5.array();
    float syn = (ws.array() * (c.array() - d.array())).sum();
    float e = t * f3 * c3;
    float f = f3 * c6;
    float res = (v_reset - v_thr) * (e - f);
    return 1. / pow(tau_m, 2) * (syn + res);
}


// Derivative with respect to the reset potential
float Lif::dvdv_reset() {
    return f3 * c3;
}


// Perform a graident update step upon the occurrence of an error
void Lif::training_step(int error_signal, bool ws_, bool tau_s_, bool tau_m_, bool v_reset_, bool use_obf, float surrogate_beta) {
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
            if (tau_s_) adam->add(&this->tau_s);
            if (tau_m_) adam->add(&this->tau_m);
            if (v_reset_) adam->add(&this->v_reset);
        }
    }
    
    // Calculate Gradients
    Eigen::MatrixXf dvdws_;
    float dvdtau_s_;
    float dvdtau_m_;
    float dvdv_reset_;
    
    if (ws_) dvdws_ = obf * error_signal * dvdws().array();
    if (tau_s_) dvdtau_s_ = obf * error_signal * dvdtau_s();
    if (tau_m_) dvdtau_m_ = obf * error_signal * dvdtau_m();
    if (v_reset_) dvdv_reset_ = obf * error_signal * dvdv_reset();
    
    if (ws_) {
        if (use_adam) adam->train(&ws, dvdws_, .35);
        else ws -= this->lr * 1e-4 * dvdws_;
    }
    
    if (tau_s_) {
        if (use_adam) adam->train(&tau_s, dvdtau_s_, 7.);
        else tau_s -= this->lr * 1e-4 * dvdtau_s_;
    }
    
    if (tau_m_) {
        if (use_adam) adam->train(&tau_m, dvdtau_m_, 28.);
        else tau_m -= this->lr * 1e-4 * dvdtau_m_;
    }
    
    if (v_reset_) {
        if (use_adam) adam->train(&v_reset, dvdv_reset_, 0.7);
        else v_reset -= this->lr * 1e-4 * dvdv_reset_;
    }
    
    if (v_reset > 0.98) v_reset = 0.98;
}



