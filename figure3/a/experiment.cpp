#include "../../lib/lif.h"
#include "../../lib/lrf.h"
#include "../../lib/inputs.h"
#include "../../lib/logger.h"

#include <eigen3/Eigen/Dense>

#include <iostream>
#include <ctime>
#include <chrono>
#include <list>
#include <fstream>
#include <random>
#include <unistd.h>
#include <sys/wait.h>

#define SEEDS_N 30

using namespace Eigen;

int main(int argc, char** argv) {
    std::string file_prefix = "./results/adam";
    long steps = 4L * 3000L * 60L * 1000L;
    
    long log_interval = 600000;
    
    int excitatory_n = 80;
    int inhibitory_n = 20;
    int input_size = excitatory_n + inhibitory_n;
    bool use_adam = true;
    
    pid_t pid, wait_pid;
    
    for (int k = 0; k < SEEDS_N; k++) {
        if ((pid = fork()) == 0) {            
            std::cout << "Started pid " << getpid() << " k " << k << std::endl;
            int seed = k;
            std::mt19937 gen(seed++);
            
            // Init Inputs
            Poisson *inputs;
            
            // Init Teacher
            Lif *teacher = new Lif(excitatory_n, inhibitory_n, 0., 0., 1., 0., 0., false, 0.);
            std::uniform_real_distribution<float> v_reset_gen(-1.5, 0.9);
            std::uniform_real_distribution<float> tau_m_gen(10., 60.);
            std::uniform_real_distribution<float> target_firing_rate_gen(1., 50.);
            float target_firing_rate = -1.;
            do {
                inputs = new Poisson(excitatory_n, inhibitory_n, 10., 40., seed);
                teacher->reset();
                teacher->tau_m = tau_m_gen(gen);
                teacher->tau_s = teacher->tau_m / 4.;
                teacher->v_reset = v_reset_gen(gen);
                target_firing_rate = teacher->set_firing_rate(inputs, target_firing_rate_gen(gen), seed);
            } while (teacher->beta > 2.5);
            inputs = new Poisson(excitatory_n, inhibitory_n, 10., 40., seed);
            teacher->reset();
            
            // Init Student
            float tau_m_init = tau_m_gen(gen);
            float v_reset_init = v_reset_gen(gen);
            Lif *student = new Lif(excitatory_n, inhibitory_n, 0., v_reset_init, 1., tau_m_init / 4., tau_m_init, use_adam, 1e-4);
            student->init_synaptic_weights(1., ++seed, true);
            
            LifLogger *logger = new LifLogger(input_size, steps, log_interval, file_prefix, std::to_string(k));
            logger->add_targets(teacher, target_firing_rate);
            
            std::cout << "Initialised " << "(k: " << k << ", pid: " << getpid() << ")" << " (Firing rate: " << target_firing_rate << ", Beta: " << teacher->beta << ")" << std::endl;
            
            // Time
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            
            // Simulate
            unsigned int teacher_spikes_n = 0;
            unsigned int student_spikes_n = 0;
            float teacher_out = 0.;
            float student_out = 0.;
            for (long i = 0; i < steps; i++) {
                // Cache
                if (i % log_interval == 0) {
                    logger->log(teacher, student, student_spikes_n / (float(log_interval) / 1000.));
                    student_spikes_n = 0;
                }
                
                MatrixXf x = inputs->sample();
                teacher_out = float(teacher->forward(x));
                student_out = float(student->forward(x));
                if (teacher_out != student_out) {
                    student->training_step(student_out - teacher_out, true, true, true, true, true, 0.);
                }
                
                if (teacher_out) teacher_spikes_n += 1;
                if (student_out) student_spikes_n += 1;
                
                // Evaluate
                if (i == int(steps / 1000) or i == int(steps / 100) or i == int(steps / 10) or i == int(steps / 5) or i == int(steps / 2) or i+1 == steps) {
                    int percentage = 0;
                    if (i == int(steps / 1000)) percentage = 0;
                    else if (i == int(steps / 100)) percentage = 1;
                    else if (i == int(steps / 10)) percentage = 10;
                    else if (i == int(steps / 5)) percentage = 20;
                    else if (i == int(steps / 2)) percentage = 50;
                    else percentage = 100;
                    
                    Poisson *eval_inputs = new Poisson(excitatory_n, inhibitory_n, 10., 40., seed);
                    std::list<float> teacher_spike_times = {};
                    std::list<float> student_spike_times = {};
                    for (float j = 0.; j < 1000. * 1000.; j++) {
                        MatrixXf x = eval_inputs->sample();
                        teacher_out = float(teacher->forward(x));
                        student_out = float(student->forward(x));
                        if (teacher_out) teacher_spike_times.push_back(j);
                        if (student_out) student_spike_times.push_back(j);
                    }
                    
                    // Convert to eigen matrix
                    MatrixXf teacher_times = MatrixXf(1, teacher_spike_times.size());
                    unsigned int l = 0;
                    for (float spike_time : teacher_spike_times) {
                        teacher_times(0, l++) = spike_time;
                    }
                    
                    MatrixXf student_times = MatrixXf(1, student_spike_times.size());
                    l = 0;
                    for (float spike_time : student_spike_times) {
                        student_times(0, l++) = spike_time;
                    }
                    
                    // Save
                    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
                    std::ofstream fd1;
                    std::ofstream fd2;
                    
                    fd1.open(file_prefix + "_lif_k_" + std::to_string(k) + "_eval_" + std::to_string(percentage) + "_teacher.csv");
                    fd1 << teacher_times.format(CSVFormat);
                    fd1.close();
                    fd1.clear();
                    
                    fd2.open(file_prefix + "_lif_k_" + std::to_string(k) + "_eval_" + std::to_string(percentage) + "_student.csv");
                    fd2 << student_times.format(CSVFormat);
                    fd2.close();
                    fd2.clear();
                }
                
                // Report
                if (k == 0 and i % int(steps / 10) == 0) {
                    std::cout << int(float(i) / float(steps) * 100) << " % \r" << std::flush;
                }
            }
            
            // Store cache
            logger->save();
            
            // Report Runtime
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_span = t2 - t1;
            
            std::cout << "Process " << "(" << k << ", " << getpid() << ")" << " done. Runtime: " << time_span.count() << " seconds." << std::endl;
            exit(0);
        }
    }
    
    // Wait for all simulations to finish
    int status = 0;
    while ((wait_pid = wait(&status)) > 0) {
        continue;
    }
    
    return 0;
}







