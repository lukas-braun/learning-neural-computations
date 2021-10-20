#pragma once

////////////////
// Include system libraries

#include <eigen3/Eigen/Dense>
#include <vector>

class Adam {
    private:
        float alpha;
        float epsilon_;
        
        std::vector<float*> float_params;
        std::vector<float> float_mt;
        std::vector<float> float_vt;
        std::vector<Eigen::MatrixXf*> matrix_params;
        std::vector<Eigen::MatrixXf> matrix_mt;
        std::vector<Eigen::MatrixXf> matrix_vt;
        
        std::vector<bool> float_updated;
        std::vector<bool> matrix_updated;
        
        void advance_counter();
    
    public:
        // Constructor
        Adam(float learning_rate, float beta_1, float beta_2, float epsilon);
        
        // Methods
        void add(float* parameter);
        void add(Eigen::MatrixXf* parameter);
        void train(float* parameter, float gradient, float learning_rate_scalar);
        void train(Eigen::MatrixXf* parameter, Eigen::MatrixXf gradient, float learning_rate_scalar);
        
        // Properties
        unsigned long t;
        float learning_rate;
        float beta_1;
        float beta_2;
        float epsilon;
};

