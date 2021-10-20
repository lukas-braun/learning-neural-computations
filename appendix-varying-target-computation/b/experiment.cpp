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
    std::string file_prefix = "./results/vary_target_computation";
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
            
            // Init teacher
            Lrf *teacher;
            int target_n = 0;
            
            float x_hat = 0.;
            float kappa = 0;
            
            std::uniform_real_distribution<float> b_gen(-20., -120.);
            std::uniform_real_distribution<float> omega_gen(2., 25.);
            std::uniform_real_distribution<float> v_reset_gen(-0.8, 0.8);
            std::uniform_real_distribution<float> i_reset_gen(-0.8, 0.8);
            std::uniform_real_distribution<float> target_firing_rate_gen(1., 20.);
            
            // Init Inputs
            Poisson *inputs = new Poisson(excitatory_n, inhibitory_n, 10., 40., seed++);;
            Poisson *inputs_;
            
            // Init Student
            Lrf *student = new Lrf(excitatory_n, inhibitory_n, v_reset_gen(gen), i_reset_gen(gen), 1., 0., 0., true, 1e-4);
            do {
                student->b = b_gen(gen) / 1000.;
                student->omega = (omega_gen(gen) * 2. * 3.14159) / 1000.;
                x_hat = -atan(student->omega / student->b) / student->omega;
                kappa = 1. / (exp(x_hat * student->b) * sin(x_hat * student->omega));
            } while(kappa > 4.);
            student->init_synaptic_weights(1., ++seed, true);
            
            // Init Logger
            LrfLogger *logger = new LrfLogger(input_size, steps, log_interval, file_prefix, std::to_string(k));
            
            // Time
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            
            // Simulate
            unsigned int teacher_spikes_n = 0;
            unsigned int student_spikes_n = 0;
            float teacher_out = 0.;
            float student_out = 0.;
            std::vector<int> idx(input_size, 0);
            for(int p = 0; p < input_size; p++) {
                idx[p] = p;
            }
            
            for (long i = 0; i < steps; i++) {
                // Init Teacher
                if (i % (3000L * 60L * 1000L) == 0) {
                    teacher = new Lrf(excitatory_n, inhibitory_n, 0., 0., 1., 0., 0., false, 0.);
                    float target_firing_rate = -1.;
                    do {
                        inputs_ = new Poisson(excitatory_n, inhibitory_n, 10., 40., seed);
                        teacher->reset();
                        do {
                            teacher->b = b_gen(gen) / 1000.;
                            teacher->omega = (omega_gen(gen) * 2. * 3.14159) / 1000.;
                            x_hat = -atan(teacher->omega / teacher->b) / teacher->omega;
                            kappa = 1. / (exp(x_hat * teacher->b) * sin(x_hat * teacher->omega));
                        } while(kappa > 4.);
                        
                        teacher->v_reset = v_reset_gen(gen);
                        teacher->i_reset = i_reset_gen(gen);
                        target_firing_rate = target_firing_rate_gen(gen);
                        
                        target_firing_rate = teacher->set_firing_rate(inputs, target_firing_rate, seed);
                    } while (teacher->beta > 2.5);
                    teacher->reset();
                    
                    // Shuffle position of excitatory and inhibitory inputs
                    std::shuffle(idx.begin(), idx.end(), gen);
                    MatrixXf ws_copy = teacher->ws;
                    for (int row = 0; row < input_size; ++row) {
                        teacher->ws(row, 0) = ws_copy(idx[row], 0);
                    }
                    
                    logger->add_targets(teacher, target_firing_rate);
                    logger->prefix = file_prefix + "_" + std::to_string(target_n);
                    logger->save();
                    
                    seed++;
                    target_n++;
                    
                    std::cout << "Initialised " << "(k: " << k << ", pid: " << getpid() << ") with target " << target_n << " (Firing rate: " << target_firing_rate << ", Beta: " << teacher->beta << ")" << std::endl;
                }
            
                // Cache
                if (i % log_interval == 0) {
                    logger->log(teacher, student, student_spikes_n / (float(log_interval) / 1000.));
                    student_spikes_n = 0;
                }
                
                MatrixXf x = inputs->sample();
                MatrixXf x_copy = x;
                // Resort x
                for (int row = 0; row < input_size; ++row) {
                    x(row) = x_copy(idx[row]);
                }
                teacher_out = float(teacher->forward(x));
                student_out = float(student->forward(x));
                if (teacher_out != student_out) {
                    student->training_step(student_out - teacher_out, true, true, true, true, true, true, 0.);
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
    
    return 0;
}







