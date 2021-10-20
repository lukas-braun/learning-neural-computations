////////////////
// Include project headers

#include "adam.h"


////////////////
// Include libraries

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <vector>


////////////////
// Constructor

Adam::Adam(float learning_rate, float beta_1, float beta_2, float epsilon) {
    this->learning_rate = learning_rate;
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
    this->epsilon = epsilon;
    this->t = 1;
    
    alpha = learning_rate * sqrt(1. - pow(beta_2, t)) / (1. - pow(beta_1, t));
    epsilon_ = epsilon * sqrt(1 - pow(beta_2, t));
    
    float_params = {};
    float_mt = {};
    float_vt = {};
    matrix_params = {};
    matrix_mt = {};
    matrix_vt = {};
    
    float_updated = {};
    matrix_updated = {};
}


////////////////
// Methods

// Add a floating parameter to be traced
void Adam::add(float* parameter) {
    float_params.push_back(parameter);
    float_mt.push_back(0.);
    float_vt.push_back(0.);
    float_updated.push_back(false);
}

// Add a matrix parameter to be traced
void Adam::add(Eigen::MatrixXf* parameter) {
    matrix_params.push_back(parameter);
    Eigen::MatrixXf mt = Eigen::MatrixXf::Zero(parameter->rows(), parameter->cols());
    Eigen::MatrixXf vt = Eigen::MatrixXf::Zero(parameter->rows(), parameter->cols());
    matrix_mt.push_back(mt);
    matrix_vt.push_back(vt);
    matrix_updated.push_back(false);
}

// Perform an optimisation step
void Adam::train(float* parameter, float gradient, float learning_rate_scalar) {
    unsigned int i = 0;
    for (auto p : float_params) {
        if (parameter == p) {
            float_mt[i] = beta_1 * float_mt[i] + (1. - beta_1) * gradient;
            float_vt[i] = beta_2 * float_vt[i] + (1. - beta_2) * pow(gradient, 2);
            (*p) -= learning_rate_scalar * alpha * (float_mt[i] / (sqrt(float_vt[i]) + epsilon_));
            if (float_updated[i]) std::cout << "Warning: Parameter updated a second time without counter being advanced." << std::endl;
            float_updated[i] = true;
            break;
        }
        i++;
    }
    advance_counter();
}

// Perform an optimisation step
void Adam::train(Eigen::MatrixXf* parameter, Eigen::MatrixXf gradient, float learning_rate_scalar) {
    unsigned int i = 0;
    for (auto p : matrix_params) {
        if (parameter == p) {
            matrix_mt[i] = beta_1 * matrix_mt[i].array() + (1. - beta_1) * gradient.array();
            matrix_vt[i] = beta_2 * matrix_vt[i].array() + (1. - beta_2) * gradient.array() * gradient.array(); //pow(gradient.array(), 2);
            (*p) = (*p).array() - learning_rate_scalar * alpha * (matrix_mt[i].array() / (sqrt(matrix_vt[i].array()) + epsilon_));
            if (matrix_updated[i]) std::cout << "Warning: Parameter updated a second time without counter being advanced." << std::endl;
            matrix_updated[i] = true;
            break;
        }
        i++;
    }
    advance_counter();
}

// If all traced parameters are updated, calculate new alpha and epsilon values
void Adam::advance_counter() {
    unsigned int updated = 0;
    for (bool u : float_updated) {
        if (u) updated++;
    }
    for (bool u : matrix_updated) {
        if (u) updated++;
    }
    if (updated == float_updated.size() + matrix_updated.size()) {
        t++;
        float b2 = sqrt(1. - pow(beta_2, t));
        alpha = learning_rate * b2 / (1. - pow(beta_1, t));
        epsilon_ = epsilon * b2;
        std::fill(float_updated.begin(), float_updated.end(), false);
        std::fill(matrix_updated.begin(), matrix_updated.end(), false);
    }
}


