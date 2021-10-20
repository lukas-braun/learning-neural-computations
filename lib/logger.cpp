////////////////
// Include project headers

#include "../lib/lif.h"
#include "../lib/lrf.h"
#include "../lib/logger.h"


////////////////
// Include libraries

#include <eigen3/Eigen/Dense>
#include <fstream>

////////////////
// Constructor

LifLogger::LifLogger(int input_size, long training_steps, long log_intervals, std::string prefix, std::string k) {
    ws_cache = Eigen::MatrixXf(input_size, long(training_steps / log_intervals));
    tau_s_cache = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    tau_m_cache = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    v_reset_cache = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    firing_rate_cache = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    
    teacher_vs = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    student_vs = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    
    ws_target = Eigen::MatrixXf(input_size, 1);
    tau_s_target = Eigen::MatrixXf(1, 1);
    tau_m_target = Eigen::MatrixXf(1, 1);
    v_reset_target = Eigen::MatrixXf(1, 1);
    firing_rate_target = Eigen::MatrixXf(1, 1);
    
    this->prefix = prefix;
    this->k = k;
    this->calls = 0;
}

////////////////
// Methods

// Add to be reached target values
void LifLogger::add_targets(Lif *teacher, float firing_rate) {
    ws_target.col(0) = teacher->ws.col(0);
    tau_s_target(0) = teacher->tau_s;
    tau_m_target(0) = teacher->tau_m;
    v_reset_target(0) = teacher->v_reset;
    firing_rate_target(0) = firing_rate;
}

// Log current state of student
void LifLogger::log(Lif *teacher, Lif *student, float firing_rate) {
    ws_cache.col(calls) =  student->ws.col(0);
    tau_s_cache(0, calls) = student->tau_s;
    tau_m_cache(0, calls) = student->tau_m;
    v_reset_cache(0, calls) = student->v_reset;
    firing_rate_cache(0, calls) = firing_rate;
    
    teacher_vs(0, calls) = teacher->v;
    student_vs(0, calls) = student->v;
    
    calls++;
}

// Store all logs
void LifLogger::save() {
    std::string file_names[] = {
        prefix + "_lif_k_" + k + "_ws.csv",
        prefix + "_lif_k_" + k + "_ws_target.csv",
        prefix + "_lif_k_" + k + "_tau_s.csv",
        prefix + "_lif_k_" + k + "_tau_s_target.csv",
        prefix + "_lif_k_" + k + "_tau_m.csv",
        prefix + "_lif_k_" + k + "_tau_m_target.csv",
        prefix + "_lif_k_" + k + "_v_reset.csv",
        prefix + "_lif_k_" + k + "_v_reset_target.csv",
        prefix + "_lif_k_" + k + "_firing_rate.csv",
        prefix + "_lif_k_" + k + "_firing_rate_target.csv",
        prefix + "_lif_k_" + k + "_teacher_vs.csv",
        prefix + "_lif_k_" + k + "_student_vs.csv"
    };
    
    Eigen::MatrixXf cache_data[] = {
        ws_cache,
        ws_target,
        tau_s_cache,
        tau_s_target,
        tau_m_cache,
        tau_m_target,
        v_reset_cache,
        v_reset_target,
        firing_rate_cache,
        firing_rate_target,
        teacher_vs,
        student_vs
    };
    
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream fd;
    unsigned int i = 0;
    
    for(const std::string &file_name : file_names) {
        fd.open(file_name);
        fd << cache_data[i++].format(CSVFormat);
        fd.close();
        fd.clear();
    }
}


////////////////
// Constructor

LrfLogger::LrfLogger(int input_size, long training_steps, long log_intervals, std::string prefix, std::string k) {
    ws_cache = Eigen::MatrixXf(input_size, long(training_steps / log_intervals));
    b_cache = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    omega_cache = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    v_reset_cache = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    i_reset_cache = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    firing_rate_cache = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    
    teacher_vs = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    student_vs = Eigen::MatrixXf(1, long(training_steps / log_intervals));
    
    ws_target = Eigen::MatrixXf(input_size, 1);
    b_target = Eigen::MatrixXf(1, 1);
    omega_target = Eigen::MatrixXf(1, 1);
    v_reset_target = Eigen::MatrixXf(1, 1);
    i_reset_target = Eigen::MatrixXf(1, 1);
    firing_rate_target = Eigen::MatrixXf(1, 1);
    
    this->prefix = prefix;
    this->k = k;
    this->calls = 0;
}

////////////////
// Methods

// Add to be reached target values
void LrfLogger::add_targets(Lrf *teacher, float firing_rate) {
    ws_target.col(0) = teacher->ws.col(0);
    b_target(0) = teacher->b;
    omega_target(0) = teacher->omega;
    v_reset_target(0) = teacher->v_reset;
    i_reset_target(0) = teacher->i_reset;
    firing_rate_target(0) = firing_rate;
}

// Log current state of student
void LrfLogger::log(Lrf *teacher, Lrf *student, float firing_rate) {
    ws_cache.col(calls) = student->ws.col(0);
    b_cache(0, calls) = student->b;
    omega_cache(0, calls) = student->omega;
    v_reset_cache(0, calls) = student->v_reset;
    i_reset_cache(0, calls) = student->i_reset;
    firing_rate_cache(0, calls) = firing_rate;
    
    teacher_vs(0, calls) = teacher->v;
    student_vs(0, calls) = student->v;
    
    calls++;
}

// Store all logs
void LrfLogger::save() {
    std::string file_names[] = {
        prefix + "_lrf_k_" + k + "_ws.csv",
        prefix + "_lrf_k_" + k + "_ws_target.csv",
        prefix + "_lrf_k_" + k + "_b.csv",
        prefix + "_lrf_k_" + k + "_b_target.csv",
        prefix + "_lrf_k_" + k + "_omega.csv",
        prefix + "_lrf_k_" + k + "_omega_target.csv",
        prefix + "_lrf_k_" + k + "_v_reset.csv",
        prefix + "_lrf_k_" + k + "_v_reset_target.csv",
        prefix + "_lrf_k_" + k + "_i_reset.csv",
        prefix + "_lrf_k_" + k + "_i_reset_target.csv",
        prefix + "_lrf_k_" + k + "_firing_rate.csv",
        prefix + "_lrf_k_" + k + "_firing_rate_target.csv",
        prefix + "_lrf_k_" + k + "_teacher_vs.csv",
        prefix + "_lrf_k_" + k + "_student_vs.csv"
    };
    
    Eigen::MatrixXf cache_data[] = {
        ws_cache,
        ws_target,
        b_cache,
        b_target,
        omega_cache,
        omega_target,
        v_reset_cache,
        v_reset_target,
        i_reset_cache,
        i_reset_target,
        firing_rate_cache,
        firing_rate_target,
        teacher_vs,
        student_vs
    };
    
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream fd;
    unsigned int i = 0;
    
    for(const std::string &file_name : file_names) {
        fd.open(file_name);
        fd << cache_data[i++].format(CSVFormat);
        fd.close();
        fd.clear();
    }
}






