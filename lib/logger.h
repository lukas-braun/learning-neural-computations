#pragma once

////////////////
// Include system libraries

#include <eigen3/Eigen/Dense>

class LifLogger {
    private:
        Eigen::MatrixXf ws_cache;
        Eigen::MatrixXf tau_s_cache;
        Eigen::MatrixXf tau_m_cache;
        Eigen::MatrixXf v_reset_cache;
        Eigen::MatrixXf firing_rate_cache;
        
        Eigen::MatrixXf teacher_vs;
        Eigen::MatrixXf student_vs;
        
        Eigen::MatrixXf ws_target;
        Eigen::MatrixXf tau_s_target;
        Eigen::MatrixXf tau_m_target;
        Eigen::MatrixXf v_reset_target;
        Eigen::MatrixXf firing_rate_target;
        
        unsigned int calls;
        
    public:
        // Constructor
        LifLogger(int input_size, long training_steps, long log_intervals, std::string prefix, std::string k);
        
        // Methods
        void add_targets(Lif *teacher, float firing_rate);
        void log(Lif *teacher, Lif *student, float firing_rate);
        void save();
        
        // Properties
        std::string prefix;
        std::string k;
};


class LrfLogger {
    private:
        Eigen::MatrixXf ws_cache;
        Eigen::MatrixXf b_cache;
        Eigen::MatrixXf omega_cache;
        Eigen::MatrixXf v_reset_cache;
        Eigen::MatrixXf i_reset_cache;
        Eigen::MatrixXf firing_rate_cache;
        
        Eigen::MatrixXf teacher_vs;
        Eigen::MatrixXf student_vs;
        
        Eigen::MatrixXf ws_target;
        Eigen::MatrixXf b_target;
        Eigen::MatrixXf omega_target;
        Eigen::MatrixXf v_reset_target;
        Eigen::MatrixXf i_reset_target;
        Eigen::MatrixXf firing_rate_target;
        
        unsigned int calls;
        
    public:
        // Constructor
        LrfLogger(int input_size, long training_steps, long log_intervals, std::string prefix, std::string pid);
        
        // Methods
        void add_targets(Lrf *teacher, float firing_rate);
        void log(Lrf *teacher, Lrf *student, float firing_rate);
        void save();
        
        // Properties
        std::string prefix;
        std::string k;
};



