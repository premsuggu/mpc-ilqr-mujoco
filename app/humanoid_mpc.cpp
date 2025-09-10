// app/humanoid_mpc.cpp
#include "robot_utils.hpp"
#include "ilqr.hpp"
#include "mpc.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

int main() {
    // Configuration (matching paper with improved stability)
    const double dt = 0.02;          // 50Hz MPC (paper)
    const int N = 25;                // 0.5s horizon (paper)
    const int sim_steps = 100;       // 4 seconds simulation
    
    // Use smaller physics timestep for stability
    const double physics_dt = 0.02; 
    
    // Initialize robot
    RobotUtils robot;
    if (!robot.loadModel("/home/prem/mujoco_mpc/robot/h1_description/mjcf/scene.xml")) {
        std::cerr << "Failed to load robot model\n";
        return 1;
    }
    
    // Set paper's key parameters
    robot.setContactImpratio(100.0);  // Paper's approach
    robot.setTimeStep(physics_dt);    // Use fine physics timestep
    
    // Initialize robot to stable standing pose
    robot.initializeStandingPose();
    
    std::cout << "Model loaded: nx=" << robot.nx() << ", nu=" << robot.nu() << "\n";
    
    // Set cost weights (hierarchical as per paper)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(robot.nu(), robot.nu());
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());
    
    // BASE POSITION: Set to ZERO (don't track absolute position)
    Q(0,0) = 50.0;   // X position - let it drift
    Q(1,1) = 50.0;   // Y position - let it drift  
    Q(2,2) = 1000.0;   // Z position - CRITICAL: don't fight gravity directly
    
    // BASE ORIENTATION: Moderate tracking (keep upright)
    Q(3,3) = 50.0;  // qw (scalar part of quaternion)
    Q(4,4) = 50.0;  // qx (roll - very important for balance)  
    Q(5,5) = 50.0;  // qy (pitch - very important for balance)
    Q(6,6) = 50.0;   // qz (yaw - less critical)

    for (int i = 7; i < robot.nq(); ++i) {
        Q(i, i) = 10.0;  // Joint positions
    }
    // VELOCITIES: This is where stability comes from!
    int nq = robot.nq();
    Q(nq + 0, nq + 0) = 10.0;      // X velocity damping
    Q(nq + 1, nq + 1) = 10.0;      // Y velocity damping  
    Q(nq + 2, nq + 2) = 500.0;    // Z velocity damping - CRITICAL for height stability
    Q(nq + 3, nq + 3) = 10.0;     // Angular velocity damping (roll)
    Q(nq + 4, nq + 4) = 10.0;     // Angular velocity damping (pitch)
    Q(nq + 5, nq + 5) = 10.0;      // Angular velocity damping (yaw)

    // Joint velocities: Strong damping for smooth motion
    for (int i = nq + 6; i < robot.nx(); ++i) {
        Q(i, i) = 0.1;  // Joint velocity damping
    }

    R *= 0.001;        // Control effort  
    Qf = Q*10.0;      // Terminal
    
    robot.setCostWeights(Q, R, Qf);
    
    // Set constraint weights for joint and control limits
    robot.setConstraintWeights(1e3,      // w_joint_limits 
                              1e3);      // w_control_limits
    
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
        // Get current state (from simulation - later from sensors)
        Eigen::VectorXd x_current(robot.nx());
        robot.getState(x_current);
        
        // Check for NaN values in state
        if (!x_current.allFinite()) {
            std::cerr << "NaN detected in state at step " << step << ", breaking simulation" << std::endl;
            break;
        }
        
        // MPC step
        Eigen::VectorXd u_apply(robot.nu());
        bool success = mpc.stepOnce(x_current, u_apply);
        
        if (!success) {
            std::cerr << "MPC failed at step " << step << "\n";
            // Continue with zero control instead of breaking
            u_apply.setZero();
            if (step > 10) {  // Allow some initial failures
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
        for (int sub_step = 0; sub_step < physics_steps_per_mpc; ++sub_step) {
            robot.step();
        }
        
        // Enhanced progress reporting every 10 steps
        if (step % 25 == 0 || step == sim_steps - 1) {
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
                      << " | Control range: [" << u_min << ", " << u_max << "]"
                      << " | Imp"
                      << std::endl;
            
            robot.diagnoseContactForces();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Finalize optimal trajectory logging
    mpc.finalizeOptimalTrajectoryLog();
    
    std::cout << "Simulation completed in " << duration.count() << " ms\n";
    std::cout << "Average step time: " << duration.count() / (double)sim_steps << " ms\n";
    
    // Save results
    std::cout << "Applied optimal trajectories saved to: /home/prem/mujoco_mpc/results/q_optimal.csv and u_optimal.csv\n";
    
    return 0;
}
