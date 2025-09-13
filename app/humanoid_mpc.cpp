// app/humanoid_mpc.cpp
#include "robot_utils.hpp"
#include "ilqr.hpp"
#include "mpc.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

int main() {
    // Configuration (matching paper with improved stability)
    const double dt = 0.05;          // 50Hz MPC (paper)
    const int N = 25;                // 0.5s horizon (paper)
    const int sim_steps = 50;       // 4 seconds simulation
    
    // Use smaller physics timestep for stability
    const double physics_dt = 0.05; 
    
    // Initialize robot
    RobotUtils robot;
    if (!robot.loadModel("/home/prem/mujoco_mpc/robot/h1_description/mjcf/scene.xml")) {
        std::cerr << "Failed to load robot model\n";
        return 1;
    }
    
    // Set paper's key parameters
    robot.setContactImpratio(500.0);  // Paper's approach
    robot.setTimeStep(physics_dt);    // Use fine physics timestep
    // ZERO GRAVITY TEST - ELIMINATE GRAVITY COMPLETELY
    robot.setGravity(0.0, 0.0, -0.0025);  // No gravity in any direction
    // robot.scaleRobotMass(0.3);  // Now robot weighs ~5kg instead of 51kg
    
    // Initialize robot to stable standing pose
    robot.initializeStandingPose();
    
    std::cout << "Model loaded: nx=" << robot.nx() << ", nu=" << robot.nu() << "\n";
    
    // Set cost weights (hierarchical as per paper)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(robot.nu(), robot.nu());
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());
    
    // ZERO GRAVITY OPTIMIZED COST WEIGHTS
    Q(0,0) = 10.0;   // X position - very loose tracking
    Q(1,1) = 10.0;   // Y position - very loose tracking  
    Q(2,2) = 50.0;   // Z position 

    // BASE ORIENTATION: Keep moderate
    Q(3,3) = 100.0;  // qw - more important in zero-g
    Q(4,4) = 100.0;  // qx (roll)
    Q(5,5) = 100.0;  // qy (pitch) 
    Q(6,6) = 50.0;   // qz (yaw)

    for (int i = 7; i < robot.nq(); ++i) {
        Q(i, i) = 50.0;  // Joint positions - increased for stability
    }

    // VELOCITIES: Critical for zero-gravity stability
    int nq = robot.nq();
    Q(nq + 0, nq + 0) = 100.0;  // X velocity damping
    Q(nq + 1, nq + 1) = 100.0;  // Y velocity damping  
    Q(nq + 2, nq + 2) = 300.0;  // Z velocity 
    Q(nq + 3, nq + 3) = 50.0;   // Angular velocity damping
    Q(nq + 4, nq + 4) = 50.0;   
    Q(nq + 5, nq + 5) = 50.0;   

    // Joint velocities: Very important for smooth motion
    for (int i = nq + 6; i < robot.nx(); ++i) {
        Q(i, i) = 100.0;  // INCREASED joint velocity damping
    }

    R *= 0.1;         // INCREASED control effort penalty 
    Qf = Q*10.0;      // Terminal
    Qf(nq + 2, nq + 2)  *= 10.0;
    
    robot.setCostWeights(Q, R, Qf);
    
    // Set constraint weights for joint and control limits
    robot.setConstraintWeights(0,      // w_joint_limits 
                              0);      // w_control_limits
    
    // Load reference trajectories
    if (!robot.loadReferences("/home/prem/mujoco_mpc/data/q_standing.csv", "/home/prem/mujoco_mpc/data/v_standing.csv")) {
        std::cerr << "Failed to load reference trajectories\n";
        return 1;
    }    
    // Initialize MPC
    MPC mpc(robot, N, dt);
    
    // Simulation loop with sub-stepping for stability
    // std::cout << "Starting MPC simulation loop with " << dt/physics_dt << "x physics sub-stepping...\n";
    
    // Enable optimal trajectory logging to results folder
    mpc.enableOptimalTrajectoryLogging("/home/prem/mujoco_mpc/results");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int physics_steps_per_mpc = (int)(dt / physics_dt);
    
    for (int step = 0; step < sim_steps; ++step) {
        // ENHANCED NUMERICAL STABILITY MONITORING
        // Check system stability before proceeding
        if (!robot.checkNumericalStability()) {
            std::cout << "INSTABILITY DETECTED at step " << step << "!" << std::endl;
            robot.logNumericalState();
            
            // Attempt recovery
            if (robot.recoverFromInstability()) {
                std::cout << "Recovery successful, continuing simulation..." << std::endl;
            } else {
                std::cout << "Recovery failed, terminating simulation." << std::endl;
                break;
            }
        }
        
        // Get current state (from simulation - later from sensors)
        Eigen::VectorXd x_current(robot.nx());
        robot.getState(x_current);
        
        // Check for NaN values in state
        if (!x_current.allFinite()) {
            std::cerr << "NaN detected in state at step " << step << ", attempting recovery..." << std::endl;
            
            if (robot.recoverFromInstability()) {
                robot.getState(x_current);  // Get recovered state
                if (!x_current.allFinite()) {
                    std::cerr << "Recovery failed, breaking simulation" << std::endl;
                    break;
                }
            } else {
                break;
            }
        }
        
        // MPC step
        Eigen::VectorXd u_apply(robot.nu());
        bool success = mpc.stepOnce(x_current, u_apply);
        
        if (!success) {
            std::cerr << "MPC failed at step " << step << std::endl;
            // Continue with zero control instead of breaking
            u_apply.setZero();
            if (step > 10) {  // Allow some initial failures
                // Log state before potential termination
                robot.logNumericalState();
                break;
            }
        }
        
        // Check for NaN values in control
        if (!u_apply.allFinite()) {
            std::cerr << "NaN detected in control at step " << step << ", using zero control" << std::endl;
            u_apply.setZero();
        }
        
        // Apply control and sub-step physics for stability
        robot.setControl(u_apply);
        // CRITICAL: Recompute contacts after control change
        mj_forward(robot.model(), robot.data());
        
        // Enhanced sub-stepping with stability monitoring
        for (int sub_step = 0; sub_step < physics_steps_per_mpc; ++sub_step) {
            // Check stability before each physics step
            if (sub_step > 0 && !robot.checkNumericalStability()) {
                std::cout << "Instability detected during sub-step " << sub_step 
                          << " of step " << step << std::endl;
                robot.recoverFromInstability();
            }
            
            robot.step();
        }
        
        // Enhanced progress reporting every 10 steps
        if (step % 1 == 0 || step == sim_steps - 1) {
            // Get MPC solve cost
            double current_cost = mpc.getLastSolveCost();
            
            // Get current z position (height)
            double z_position = x_current(2);  // Z is the 3rd element (index 2)
            double x_position = x_current(0);
            double y_position = x_current(1);
            // Calculate control range
            double u_min = u_apply.minCoeff();
            double u_max = u_apply.maxCoeff();
            
            std::cout << "Step " << step << "/" << sim_steps 
                      << " | Cost: " << current_cost
                      << " | (X,Y,Z): " << "(" << x_position<< "," << y_position << "," << z_position << ")" << "m"
                      << " | Control range: [" << u_min << ", " << u_max << "]";
            
            // Add stability indicator
            if (robot.checkNumericalStability()) {
                std::cout << " | Status: STABLE";
            } else {
                std::cout << " | Status: UNSTABLE";
            }
            std::cout << std::endl;
            
            // Log detailed diagnostics every 10 steps or if unstable
            if (step % 10 == 0 || !robot.checkNumericalStability()) {
                robot.logNumericalState();
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Finalize optimal trajectory logging
    mpc.finalizeOptimalTrajectoryLog();
    
    std::cout << "Simulation completed in " << duration.count() << " ms\n";
    std::cout << "Average step time: " << duration.count() / (double)sim_steps << " ms\n";
    
    // Save results
    // std::cout << "Applied optimal trajectories saved to: /home/prem/mujoco_mpc/results/q_optimal.csv and u_optimal.csv\n";
    
    return 0;
}
