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

std::vector<std::vector<bool>>combinations = {
    {true, false, false, false, false},
    {false, true, false, false, false},
    //{false, false, true, false, false},
    {false, false, false, true, false},
    {false, false, false, false, true},
    {true, true, false, false, false},
    {true, false, true, false, false},
    {true, false, false, true, false},
    {true, false, false, false, true},
    {false, true, true, false, false},
    {false, true, false, true, false},
    {false, true, false, false, true},
    {false, false, true, true, false},
    {false, false, true, false, true},
    {false, false, false, true, true},
    {true, true, true, false, false},
    {true, true, false, true, false},
    {true, true, false, false, true},
    {true, false, true, true, false},
    {true, false, true, false, true},
    {true, false, false, true, true},
    {false, true, true, true, false},
    {false, true, true, false, true},
    {false, true, false, true, true},
    {false, false, true, true, true},
    {true, true, true, true, false},
    {true, true, true, false, true},
    {true, true, false, true, true},
    {true, false, true, true, true},
    /*{false, true, true, true, true},
    {true, true, true, true, true},*/
};


int main(int argc, char** argv) {
    long steps = 4L * 3000L * 60L * 1000L;
    long log_interval = 600000;
    
    int excitatory_n = 80;
    int inhibitory_n = 20;
    int input_size = excitatory_n + inhibitory_n;
    
    bool use_adam = true;
    
    pid_t pid, wait_pid;
    
    float surrogate_betas[3] = {0., 0.25, 0.};
    int run = 0;
    
    for (bool use_lambda : {false, false, true}) {   
        for (int k = 0; k < SEEDS_N; k++) {
            if ((pid = fork()) == 0) {
                std::cout << "Started pid " << getpid() << " k " << k << std::endl;
                int seed = k;
                std::mt19937 gen(seed++);
                
                // Init Inputs
                Poisson *inputs;
                
                 // Init Teacher
                Lrf *teacher = new Lrf(excitatory_n, inhibitory_n, 0., 0., 1., 0., 0., false, 0.);
                std::uniform_real_distribution<float> b_gen(-20., -120.);
                std::uniform_real_distribution<float> omega_gen(2., 25.);
                std::uniform_real_distribution<float> v_reset_gen(-0.8, 0.8);
                std::uniform_real_distribution<float> i_reset_gen(-0.8, 0.8);
                std::uniform_real_distribution<float> target_firing_rate_gen(1., 20.);
                float target_firing_rate = -1;
                
                float x_hat = 0.;
                float kappa = 0;
                
                do {
                    inputs = new Poisson(excitatory_n, inhibitory_n, 10., 40., seed);
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
                inputs = new Poisson(excitatory_n, inhibitory_n, 10., 40., ++seed);
                teacher->reset();
                
                // Init Student
                float b_init = 0;
                float omega_init = 0;
                do {
                    b_init = b_gen(gen);
                    omega_init = omega_gen(gen) * 2. * 3.14159;
                    x_hat = -atan(omega_init / b_init) / omega_init;
                    kappa = 1. / (exp(x_hat * b_init) * sin(x_hat * omega_init));
                } while(kappa > 4.);
                float v_reset_init = v_reset_gen(gen);
                float i_reset_init = i_reset_gen(gen);
                
                Lrf *student = new Lrf(excitatory_n, inhibitory_n, v_reset_init, i_reset_init, 1., b_init, omega_init, use_adam, 0.);
                student->init_synaptic_weights(1., ++seed, true);
                Eigen::MatrixXf ws_copy = student->ws;
                
                std::cout << "Initialised " << "(k: " << k << ", pid: " << getpid() << ")" << " (Firing rate: " << target_firing_rate << ", Beta: " << teacher->beta << ")" << std::endl;
                
                for(std::vector<bool> comb : combinations) {                
                    std::string file_prefix = "./results/mode_" + std::to_string(run) + "_" + std::to_string(comb[0]) + std::to_string(comb[1]) + std::to_string(comb[2]) + std::to_string(comb[3]) + std::to_string(comb[4]);
                    std::cout << "k " << k << ": Now running " << file_prefix << std::endl;
                    
                    // Init inputs
                    inputs = new Poisson(excitatory_n, inhibitory_n, 10., 40., k+1);
                    
                    // Init Student
                    float b_init_ = comb[1] ? b_init : teacher->b * 1000.;
                    float omega_init_ = comb[2] ? omega_init : teacher->omega * 1000.;
                    float v_reset_init_ = comb[3] ? v_reset_init : teacher->v_reset;
                    float i_reset_init_ = comb[4] ? i_reset_init : teacher->i_reset;
                    
                    student = new Lrf(excitatory_n, inhibitory_n, v_reset_init_, i_reset_init_, 1., b_init_, omega_init_, use_adam, 1e-4);
                    if (comb[0]) {
                        student->ws = ws_copy;
                    } else {
                        student->ws = teacher->ws;
                    }
                    
                    LrfLogger *logger = new LrfLogger(input_size, steps, log_interval, file_prefix, std::to_string(k));
                    logger->add_targets(teacher, target_firing_rate);
                    
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
                            student->training_step(student_out - teacher_out, comb[0], comb[1], comb[2], comb[3], comb[4], use_lambda, surrogate_betas[run]);
                        }
                        
                        if (teacher_out) teacher_spikes_n += 1;
                        if (student_out) student_spikes_n += 1;
                        
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
                }
                exit(0);
            }
        }

        // Wait for all simulations to finish
        int status = 0;
        while ((wait_pid = wait(&status)) > 0) {
            continue;
        }
        
        run += 1;
    }
    
    return 0;
}







