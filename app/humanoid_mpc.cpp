// app/humanoid_mpc.cpp
#include "robot_utils.hpp"
#include "ilqr.hpp"
#include "mpc.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <chrono>


int main() {
    
    Config config = loadConfigFromFile("config.yaml");
    std::cout << "Configuration loaded from config.yaml" << std::endl;

    RobotUtils robot;
    if (!robot.loadModel(config.model_path)) {
        std::cerr << "Failed to load robot model from: " << config.model_path << std::endl;
        return 1;
    }

    // Set simulation parameters
    robot.setContactImpratio(config.mpc.contact_impratio);
    robot.setTimeStep(config.mpc.physics_dt);
    robot.setGravity(config.mpc.gravity[0], config.mpc.gravity[1], config.mpc.gravity[2]);
    
    // Start in standing pose
    robot.initializeStandingPose();

    std::cout << "Model loaded: nx=" << robot.nx() << ", nu=" << robot.nu() << "\n";

    config.buildCostMatrices(robot.nx(), robot.nu(), robot.nq());
    
    robot.setCostWeights(config.Q, config.R, config.Qf);
    robot.setCoMWeight(config.mpc.costs.W_com);
    robot.setEEPosWeight(config.mpc.costs.W_foot);
    robot.setConstraintWeights(config.mpc.joint_limit_weight, config.mpc.torque_limit_weight);

    // Load reference trajectories
    if (!robot.loadReferences(config.q_ref_path, config.v_ref_path)) {
        std::cerr << "Failed to load reference trajectories\n";
        return 1;
    }

    MPC mpc(robot, config.mpc.horizon, config.mpc.dt, config.urdf_path);

    // Enable trajectory logging if configured
    if (config.save_trajectories) {
        mpc.enableOptimalTrajectoryLogging(config.results_path);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    int physics_steps_per_mpc = (int)(config.mpc.dt / config.mpc.physics_dt);
    for (int step = 0; step < config.mpc.sim_steps; ++step) {
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

        robot.setControl(u_apply);
        // Recompute contacts after changing control
        mj_forward(robot.model(), robot.data());
        for (int sub_step = 0; sub_step < physics_steps_per_mpc; ++sub_step) {
            robot.step();
        }

        // Print progress (if verbose mode enabled)
        if (config.verbose && (step % 1 == 0 || step == config.mpc.sim_steps - 1)) {
            double current_cost = mpc.getLastSolveCost();
            double z_position = x_current(2);
            double x_position = x_current(0);
            double y_position = x_current(1);
            double u_min = u_apply.minCoeff();
            double u_max = u_apply.maxCoeff();

            std::cout << "Step " << step << "/" << config.mpc.sim_steps
                      << " | Cost: " << current_cost
                      << " | (X,Y,Z): (" << x_position << "," << y_position << "," << z_position << ") m"
                      << " | Control range: [" << u_min << ", " << u_max << "]"
                      << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Finalize trajectory logging if enabled
    if (config.save_trajectories) {
        mpc.finalizeOptimalTrajectoryLog();
    }

    std::cout << "Simulation completed in " << duration.count() << " ms\n";
    std::cout << "Average step time: " << duration.count() / (double)config.mpc.sim_steps << " ms\n";

    return 0;
}
