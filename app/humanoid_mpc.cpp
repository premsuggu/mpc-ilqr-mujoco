// app/humanoid_mpc.cpp
#include "robot_utils.hpp"
#include "ilqr.hpp"
#include "mpc.hpp"
#include <iostream>
#include <fstream>
#include <chrono>


int main() {
    // --- Simulation setup ---
    const double dt = 0.02;          // MPC runs at 50Hz
    const int N = 25;                // 0.5s prediction horizon
    const int sim_steps = 50;        // Simulate for 1.0 second
    const double physics_dt = 0.02;  // Physics engine step size

    // Fire up the robot model
    RobotUtils robot;
    if (!robot.loadModel("/home/prem/mujoco_mpc/robot/h1_description/mjcf/scene.xml")) {
        std::cerr << "Failed to load robot model\n";
        return 1;
    }

    // Enhanced simulation parameters for stability at challenging gravity values
    robot.setContactImpratio(100.0);               // Increased contact stiffness for better ground contact
    robot.setTimeStep(physics_dt);
    robot.setGravity(0.0, 0.0, -5.0);              // World gravity
    // robot.scaleRobotMass(0.3); 

    // Start in a nice, stable standing pose
    robot.initializeStandingPose();

    std::cout << "Model loaded: nx=" << robot.nx() << ", nu=" << robot.nu() << "\n";

    // State deviation and Control effort weights
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(robot.nu(), robot.nu());
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(robot.nx(), robot.nx());

    // Position Specific
    Q(0,0) = 100.0;   // X - heavily penalize forward/backward lean
    Q(1,1) = 100.0;   // Y - heavily penalize lateral lean  
    Q(2,2) = 2000.0;  // Z - critical to maintain height

    // Orienration Specific
    Q(3,3) = 500.0;  // qw - quaternion real part
    Q(4,4) = 500.0;  // qx - roll stability
    Q(5,5) = 500.0;  // qy - pitch stability (critical for forward/back)
    Q(6,6) = 300.0;  // qz - yaw stability

    // Joint positions 
    for (int i = 7; i < robot.nq(); ++i) {
        Q(i, i) = 80.0;  // Increased from 50 for better tracking
    }

    // CRITICAL: Velocity penalties to prevent oscillations
    int nq = robot.nq();
    Q(nq + 0, nq + 0) = 800.0;   // X velocity - prevent forward/back motion
    Q(nq + 1, nq + 1) = 800.0;   // Y velocity - prevent lateral motion
    Q(nq + 2, nq + 2) = 5000.0;  // Z velocity - critical for vertical stability
    Q(nq + 3, nq + 3) = 200.0;   // Angular velocities - prevent spinning
    Q(nq + 4, nq + 4) = 200.0;
    Q(nq + 5, nq + 5) = 200.0;

    // Joint velocities - prevent rapid joint motion
    for (int i = nq + 6; i < robot.nx(); ++i) {
        Q(i, i) = 150.0;  // Increased from 100
    }

    // Control Effort
    R *= 0.01;
    
    // Terminal cost - heavily weight final state
    Qf = Q * 15.0;      // Final Terminal weight
    Qf(0,0) *= 2.0;     // Final X position
    Qf(1,1) *= 2.0;     // Final Y position
    Qf(2,2) *= 3.0;     // final Z position
    Qf(nq + 2, nq + 2) *= 5.0;  // Critical final Z velocity

    robot.setCostWeights(Q, R, Qf);  // Basic MPC with state tracking + control regularization

    // Soft Constraint Penalties
    robot.setConstraintWeights(5000.0, 5000.0); 

    // Load reference trajectories (standing still)
    if (!robot.loadReferences("/home/prem/mujoco_mpc/data/q_standing.csv", "/home/prem/mujoco_mpc/data/v_standing.csv")) {
        std::cerr << "Failed to load reference trajectories\n";
        return 1;
    }

    // Set up the MPC controller
    MPC mpc(robot, N, dt);

    // Log optimal trajectories for later analysis
    mpc.enableOptimalTrajectoryLogging("/home/prem/mujoco_mpc/results");

    auto start_time = std::chrono::high_resolution_clock::now();

    int physics_steps_per_mpc = (int)(dt / physics_dt);
    for (int step = 0; step < sim_steps; ++step) {
        // Grab the current state
        Eigen::VectorXd x_current(robot.nx());
        robot.getState(x_current);

        // Enhanced stability checks
        if (!x_current.allFinite()) {
            std::cerr << "NaN detected in state at step " << step << ", breaking simulation" << std::endl;
            break;
        }

        // Run one step of MPC
        Eigen::VectorXd u_apply(robot.nu());
        bool success = mpc.stepOnce(x_current, u_apply);

        if (!success) {
            std::cerr << "MPC failed at step " << step << ", using gravity compensation only" << std::endl;
            // Use gravity compensation as fallback instead of zero control
            mj_forward(robot.model(), robot.data());
            for (int i = 0; i < robot.nu(); ++i) {
                int joint_id = robot.model()->actuator_trnid[i * 2];
                int qpos_addr = robot.model()->jnt_qposadr[joint_id];
                u_apply(i) = robot.data()->qfrc_bias[qpos_addr] * 0.5;  // Conservative gravity comp
            }
            
            if (step > 15) {  // Give more attempts before giving up
                break;
            }
        }

        // If control is NaN, zero it out
        if (!u_apply.allFinite()) {
            std::cerr << "NaN detected in control at step " << step << ", using zero control" << std::endl;
            u_apply.setZero();
        }

        robot.setControl(u_apply);       // Without the Gravity Compensation
        // Recompute contacts after changing control
        mj_forward(robot.model(), robot.data());
        for (int sub_step = 0; sub_step < physics_steps_per_mpc; ++sub_step) {
            robot.step();
        }

        // Print progress every step
        if (step % 1 == 0 || step == sim_steps - 1) {
            double current_cost = mpc.getLastSolveCost();
            double z_position = x_current(2);
            double x_position = x_current(0);
            double y_position = x_current(1);
            double u_min = u_apply.minCoeff();
            double u_max = u_apply.maxCoeff();

            std::cout << "Step " << step << "/" << sim_steps
                      << " | Cost: " << current_cost
                      << " | (X,Y,Z): (" << x_position << "," << y_position << "," << z_position << ") m"
                      << " | Control range: [" << u_min << ", " << u_max << "]"
                      << std::endl;
            // robot.debugContactSolver();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Wrap up logging
    mpc.finalizeOptimalTrajectoryLog();

    std::cout << "Simulation completed in " << duration.count() << " ms\n";
    std::cout << "Average step time: " << duration.count() / (double)sim_steps << " ms\n";

    return 0;
}