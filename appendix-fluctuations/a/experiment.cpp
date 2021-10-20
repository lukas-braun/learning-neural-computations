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

#define SEEDS_N 1

using namespace Eigen;

int main(int argc, char** argv) {
    std::string file_prefix_ = "./results/fluctuations";
    long steps = 600L * 60L * 1000L;
    
    long log_interval = 600;
    
    int excitatory_n = 80;
    int inhibitory_n = 20;
    int input_size = excitatory_n + inhibitory_n;
    
    bool use_adam = false;
    bool use_lamda = false;
    
    pid_t pid, wait_pid;
    
    float learning_rates[4] = {1., 5., 10., 20.};
    
    for (float lr : learning_rates) {
        for (int k = 0; k < SEEDS_N; k++) {
            if ((pid = fork()) == 0) {
                std::string file_prefix = file_prefix_ + "_" + std::to_string(lr);
                
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
                Lif *student = new Lif(excitatory_n, inhibitory_n, 0., teacher->v_reset, 1., teacher->tau_s, teacher->tau_m, use_adam, lr);
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
                        student->training_step(student_out - teacher_out, true, false, false, false, use_lamda, 0.);
                    }
                    
                    if (teacher_out) teacher_spikes_n += 1;
                    if (student_out) student_spikes_n += 1;
                    
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
    }
    
    return 0;
}







