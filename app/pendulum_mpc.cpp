// app/pendulum_mpc.cpp
// Simple pendulum MPC test to validate iLQR optimization

#include "robot_utils.hpp"
#include "ilqr.hpp"
#include "mpc.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

// Generate sinusoidal reference trajectory for pendulum
void generateRefs(int num_steps, double dt,
                  std::vector<std::vector<double>>& q_ref,
                  std::vector<std::vector<double>>& v_ref,
                  bool use_sinusoidal = true) {
    q_ref.clear();
    v_ref.clear();
    
    double amplitude = M_PI / 4.0; // 45 degrees
    double frequency = 0.5; // 0.5 Hz
    
    for (int i = 0; i < num_steps; ++i) {
        double t = i * dt;
        double angle, velocity;
        
        if (use_sinusoidal) {
            angle = amplitude * sin(2.0 * M_PI * frequency * t);
            velocity = amplitude * 2.0 * M_PI * frequency * cos(2.0 * M_PI * frequency * t);
        } else {
            // CONSTANT reference at 45 degrees (not zero!)
            angle = amplitude;  // M_PI/4 radians = 45 degrees
            velocity = 0.0;     // Zero velocity for static target
        }
        std::vector<double> q_step = {angle};
        std::vector<double> v_step = {velocity};
        
        q_ref.push_back(q_step);
        v_ref.push_back(v_step);
    }
}

// Save reference to CSV files
void saveReference(const std::string& q_path, const std::string& v_path,
                   const std::vector<std::vector<double>>& q_ref,
                   const std::vector<std::vector<double>>& v_ref) {
    
    // Save q_ref
    std::ofstream q_file(q_path);
    for (const auto& row : q_ref) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) q_file << ",";
            q_file << row[i];
        }
        q_file << "\n";
    }
    q_file.close();
    
    // Save v_ref (control reference)
    std::ofstream v_file(v_path);
    for (const auto& row : v_ref) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) v_file << ",";
            v_file << row[i];
        }
        v_file << "\n";
    }
    v_file.close();
    
    std::cout << "Reference saved to: " << q_path << " and " << v_path << "\n";
}

int main() {
    std::cout << "Starting Pendulum MPC Test with iLQR...\n";

    // Configuration (matching proven working implementation)
    const double dt = 0.02;          // 20Hz MPC (exactly matches previous)
    const int N = 20;                // 1.0s horizon (exactly matches previous)
    const int sim_steps = 200;       // 10 seconds simulation (matches previous)
    const double physics_dt = 0.01;  // Match pendulum model timestep
    
    // Pendulum physical parameters (matching previous implementation)
    const double m = 1.0;            // Pendulum mass
    const double L = 1.0;            // Pendulum length
    
    // Initialize robot with pendulum model
    RobotUtils robot;
    if (!robot.loadModel("/home/prem/mujoco_mpc/pendulum/pendulum.xml")) {
        std::cerr << "Failed to load pendulum model\n";
        return 1;
    }
    
    robot.setTimeStep(physics_dt);
    
    std::cout << "Pendulum model loaded: nx=" << robot.nx() << ", nu=" << robot.nu() << "\n";
    
    // Generate and save sinusoidal reference
    std::vector<std::vector<double>> q_ref, v_ref;
    generateRefs(sim_steps + N + 10, dt, q_ref, v_ref, true);
    saveReference("/home/prem/mujoco_mpc/pendulum/q_ref.csv",
                  "/home/prem/mujoco_mpc/pendulum/v_ref.csv", q_ref, v_ref);
    
    // Set cost weights (exactly matching previous working implementation)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(robot.nu(), robot.nu());
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());
    
    Q(0, 0) = 500.0;  
    Q(1, 1) = 0.1;  
    R(0, 0) = 0.01;    
    Qf(0, 0) = 10000.0;
    Qf(1, 1) = 10.0; 
    
    robot.setCostWeights(Q, R, Qf);
    std::cout << "Cost weights set for pendulum tracking\n";
    
    // Load reference trajectories
    if (!robot.loadReferences("/home/prem/mujoco_mpc/pendulum/q_ref.csv", 
                             "/home/prem/mujoco_mpc/pendulum/v_ref.csv")) {
        std::cerr << "Failed to load reference trajectories\n";
        return 1;
    }
    
    // Initialize MPC with improved iLQR settings (matching previous implementation)
    MPC mpc(robot, N, dt);
    
    // Configure iLQR solver with parameters from previous working implementation
    // Previous used: max_iter=20, tolerance=1e-3, regularization=1e-2
    auto& ilqr_solver = const_cast<iLQR&>(mpc.solver());
    ilqr_solver.setMaxIterations(1);        // Match previous max_iter
    ilqr_solver.setTolerance(1e-4);          // Match previous tolerance=1e-4  
    ilqr_solver.setRegularization(1e-6);     // Match previous regularization=1e-2
    
    // Enable trajectory logging
    mpc.enableOptimalTrajectoryLogging("/home/prem/mujoco_mpc/results");
    
    std::cout << "Starting pendulum MPC simulation...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize pendulum state (matching previous: hanging down at pi/4 = 45 degrees)
    Eigen::VectorXd x_current(robot.nx());
    x_current(0) = 0.0;  // 45 degrees initial angle (matches previous x0)
    x_current(1) = 0.0;         // Zero initial velocity
    robot.setState(x_current);
    
    std::cout << "Initial state: angle=" << x_current(0) << " rad (" 
              << x_current(0) * 180.0 / M_PI << " deg), velocity=" << x_current(1) << " rad/s\n";
    
    for (int step = 0; step < sim_steps; ++step) {
        // Get current state
        robot.getState(x_current);
        
        // MPC step with cost tracking
        Eigen::VectorXd u_apply(robot.nu());
        bool success = mpc.stepOnce(x_current, u_apply);
        
        // Report MPC step cost (this is what we want to see!)
        double current_mpc_cost = mpc.getLastSolveCost();
        
        if (!success) {
            std::cerr << "MPC failed at step " << step << "\n";
            u_apply.setZero();
        }
        
        // Apply control and simulate
        robot.setControl(u_apply);
        robot.step();
        
        // Progress info with MPC cost progression
        if (step % 25 == 0 || step < 5) {  // More frequent output initially
            robot.getState(x_current);
            std::cout << "MPC Step " << step << "/" << sim_steps 
                      << " - Cost: " << current_mpc_cost
                      << " - Angle: " << x_current(0) * 180.0 / M_PI << " deg"
                      << " - Control: " << u_apply(0) << " Nm" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Finalize trajectory logging
    mpc.finalizeOptimalTrajectoryLog();
    
    std::cout << "Pendulum simulation completed in " << duration.count() << " ms\n";
    std::cout << "Average step time: " << duration.count() / (double)sim_steps << " ms\n";
    std::cout << "Results saved to: /home/prem/mujoco_mpc/results/\n";
    
    return 0;
}
