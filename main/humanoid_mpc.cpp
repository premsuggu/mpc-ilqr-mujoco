#include "ilqr/robot_utils.hpp"
#include "ilqr/ilqr.hpp"
#include "ilqr/mpc.hpp"
#include "ilqr/config.hpp"
#include "rerun/rerun_logger.hpp"
#include <iostream>
#include <chrono>
#include <map>
#include <vector>
#include <cmath>

#ifdef ENABLE_PROFILING
#include <fstream>
#include <sstream>
#include <iomanip>

// Platform-specific includes for memory usage
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// Simple profiling data structure and global map
struct ProfileData {
    std::vector<double> times;
};
std::map<std::string, ProfileData> prof_data;
double mem_peak = 0.0; // Global variable to track peak memory

double getCurrentMemoryMB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0); // Convert bytes to MB
    }
    return 0.0;
#else // for Linux and other Unix-like systems
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label;
            size_t value;
            iss >> label >> value;
            return value / 1024.0; // Convert KB to MB
        }
    }
    return 0.0;
#endif
}
#endif

// FUNCTION PROTOTYPES

void setupSimulation(RobotUtils& robot, Config& config);
void runSimulation(RobotUtils& robot, MPC& mpc, const Config& config, RerunLogger* rerun_logger);
#ifdef ENABLE_PROFILING
void printProfilingResults();
#endif

 
// MAIN FUNCTION
int main(int argc, char** argv) {
    std::string config_path = "config.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }

    Config config = loadConfigFromFile(config_path);
    std::cout << "Configuration loaded successfully from " << config_path << std::endl;

    RobotUtils robot;
    setupSimulation(robot, config);
    MPC mpc(robot, config.mpc.horizon, config.mpc.dt, config.urdf_path);
    
    // Set gravity magnitude for balance cost computation
    double g_magnitude = std::sqrt(config.mpc.gravity[0] * config.mpc.gravity[0] + 
                                   config.mpc.gravity[1] * config.mpc.gravity[1] + 
                                   config.mpc.gravity[2] * config.mpc.gravity[2]);
    mpc.setGravity(g_magnitude);
    std::cout << "Gravity magnitude set to: " << g_magnitude << " m/s^2" << std::endl;
    
    // Set balance time constant (omega_0) for capture point computation
    mpc.setBalanceTimeConstant(config.mpc.costs.balance_time_constant);
    std::cout << "Balance time constant set to: " << config.mpc.costs.balance_time_constant << " s" << std::endl;
    
    // Configure norm parameters for all cost terms
    mpc.configureNorms(config.norm_params);
    
    // Configure iLQR solver settings
    mpc.configureSolver(
        config.mpc.ilqr_settings.reg_min,
        config.mpc.ilqr_settings.reg_max,
        config.mpc.ilqr_settings.reg_increase_factor,
        config.mpc.ilqr_settings.reg_decrease_factor,
        config.mpc.ilqr_settings.trust_region_good,
        config.mpc.ilqr_settings.trust_region_poor,
        config.mpc.ilqr_settings.num_line_search_steps,
        config.mpc.ilqr_settings.min_linesearch_step,
        config.mpc.ilqr_settings.line_search_tolerance,
        config.mpc.ilqr_settings.quu_regularization,
        config.mpc.ilqr_settings.convergence_threshold
    );
    
    // Configure finite difference parameters (DeepMind MJPC compatible)
    mpc.setFiniteDiffParams(config.mpc.ilqr_settings.fd_tolerance, 
                            config.mpc.ilqr_settings.fd_mode);
    std::cout << "FD parameters set: tolerance=" << config.mpc.ilqr_settings.fd_tolerance 
              << ", mode=" << (config.mpc.ilqr_settings.fd_mode == 0 ? "forward" : "centered") << std::endl;

    // Initialize Rerun visualization if enabled
    RerunLogger* rerun_logger = nullptr;
    if (config.enable_rerun) {
        rerun_logger = new RerunLogger("H1_Humanoid_MPC");
        if (rerun_logger->initialize()) {
            std::cout << "[Rerun] Visualization initialized successfully" << std::endl;
        } else {
            std::cerr << "[Rerun] Failed to initialize, continuing without visualization" << std::endl;
            delete rerun_logger;
            rerun_logger = nullptr;
        }
    }

    #ifdef ENABLE_PROFILING
        double mem_initial = getCurrentMemoryMB();
        mem_peak = mem_initial; // Initialize peak memory
        std::cout << "=== Profiling ENABLED ===" << std::endl;
        std::cout << "Initial memory: " << std::fixed << std::setprecision(2) << mem_initial << " MB" << std::endl;
    #endif

    runSimulation(robot, mpc, config, rerun_logger);
    
    // Cleanup Rerun
    if (rerun_logger) {
        delete rerun_logger;
    }

    #ifdef ENABLE_PROFILING
        double mem_final = getCurrentMemoryMB();
        printProfilingResults();
        std::cout << "\n--- Memory Summary ---" << std::endl;
        std::cout << "Initial:  " << std::fixed << std::setprecision(2) << mem_initial << " MB" << std::endl;
        std::cout << "Peak:     " << std::fixed << std::setprecision(2) << mem_peak << " MB" << std::endl;
        std::cout << "Final:    " << std::fixed << std::setprecision(2) << mem_final << " MB" << std::endl;
        std::cout << "==========================" << std::endl;
    #endif

    return 0;
}

 
// SETUP FUNCTION
void setupSimulation(RobotUtils& robot, Config& config) {
    if (!robot.loadModel(config.model_path, 
                         config.left_foot_body_name, 
                         config.right_foot_body_name)) {
        throw std::runtime_error("Failed to load robot model from: " + config.model_path);
    }
    robot.setContactImpratio(config.mpc.contact_impratio);
    robot.setTimeStep(config.mpc.physics_dt);
    robot.setGravity(config.mpc.gravity[0], config.mpc.gravity[1], config.mpc.gravity[2]);
    std::cout << "Model loaded: nx=" << robot.nx() << ", nu=" << robot.nu() << std::endl;
    config.buildCostMatrices(robot.nx(), robot.nu(), robot.nq());
    robot.setCostWeights(config.Q, config.R, config.Qf);
    robot.setHeightWeight(config.mpc.costs.W_height);
    robot.setVelocityWeight(config.mpc.costs.W_vel);
    robot.setUprightWeight(config.mpc.costs.W_upright);
    robot.setBalanceWeight(config.mpc.costs.w_balance);
    robot.setPelvisFeetWeight(config.mpc.costs.W_pelvis_feet);
    robot.setWalkWeight(config.mpc.costs.W_walk);
    robot.setSpeedGoal(config.mpc.costs.speed_goal);
    robot.setJointVelWeight(config.mpc.costs.W_joint_vel);
    // Body names for costs (must be set before MPC construction so iLQR picks them up)
    robot.setLeftFootBodyName(config.left_foot_body_name);
    robot.setRightFootBodyName(config.right_foot_body_name);
    robot.setPelvisBodyName(config.pelvis_body_name);
    robot.setTorsoBodyName(config.torso_body_name);
    robot.setWaistLowerBodyName(config.waist_lower_body_name);
    robot.setConstraintWeights(config.mpc.joint_limit_weight, config.mpc.torque_limit_weight);
    robot.configureInstabilityDebug(
        config.mpc.ilqr_settings.debug_qacc_enable,
        config.mpc.ilqr_settings.debug_qacc_threshold,
        config.mpc.ilqr_settings.debug_qacc_max_logs
    );
    if (!robot.loadReferences(config.q_ref_path, config.v_ref_path)) {
        throw std::runtime_error("Failed to load reference trajectories.");
    }
    // Initialize robot state from first reference state (correct height for each model)
    if (!robot.x_ref_full_.empty()) {
        robot.setState(robot.x_ref_full_[0]);
        std::cout << "Initialized robot state from reference (Z = " 
                  << robot.x_ref_full_[0](2) << " m)" << std::endl;
    } else {
        std::cerr << "WARNING: No reference trajectory loaded, using default pose" << std::endl;
        robot.initializeStandingPose();
    }
    if (!robot.loadContactSchedule(config.contact_schedule_path)) {
        std::cerr << "Warning: Failed to load contact schedule, continuing without it." << std::endl;
    }
}

 
// SIMULATION LOOP FUNCTION
void runSimulation(RobotUtils& robot, MPC& mpc, const Config& config, RerunLogger* rerun_logger) {
    if (config.save_trajectories) {
        mpc.enableOptimalTrajectoryLogging(config.results_path);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    int physics_steps_per_mpc = static_cast<int>(config.mpc.dt / config.mpc.physics_dt);

    for (int step = 0; step < config.mpc.sim_steps; ++step) {
        Eigen::VectorXd x_current(robot.nx());
        robot.getState(x_current);

        if (!x_current.allFinite()) {
            std::cerr << "NaN detected in state at step " << step << ", breaking." << std::endl;
            break;
        }

        // Log current state to Rerun
        if (rerun_logger) {
            double sim_time = step * config.mpc.dt;
            rerun_logger->setTime(step, sim_time);
            
            // Get reference at current timestep (if available)
            const Eigen::VectorXd* x_ref = nullptr;
            const Eigen::Vector3d* height_ref = nullptr;
            const Eigen::Vector3d* com_vel_ref = nullptr;
            const std::vector<Eigen::Vector3d>* ee_pos_ref = nullptr;
            
            // Access reference data from robot (public members)
            if (step < robot.x_ref_full_.size()) {
                x_ref = &robot.x_ref_full_[step];
            }
            if (step < robot.height_ref_full_.size()) {
                height_ref = &robot.height_ref_full_[step];
            }
            if (step < robot.com_vel_ref_full_.size()) {
                com_vel_ref = &robot.com_vel_ref_full_[step];
            }
            if (step < robot.ee_pos_ref_full_.size()) {
                ee_pos_ref = &robot.ee_pos_ref_full_[step];
            }
            
            // Log base state (position & velocity)
            rerun_logger->logBaseState(x_current, x_ref, robot.nq());
            
            // Log joints
            Eigen::VectorXd q = x_current.head(robot.nq());
            Eigen::VectorXd* q_ref_ptr = nullptr;
            if (x_ref) {
                static Eigen::VectorXd q_ref_temp;
                q_ref_temp = x_ref->head(robot.nq());
                q_ref_ptr = &q_ref_temp;
            }
            auto joint_lower = robot.getJointLowerLimits();
            auto joint_upper = robot.getJointUpperLimits();
            auto joint_names = robot.getJointNames();
            rerun_logger->logJoints(q, joint_lower, joint_upper, joint_names, q_ref_ptr);
            
            // Log CoM
            Eigen::Vector3d com_pos = robot.computeCoM(x_current);
            Eigen::Vector3d com_vel = robot.computeCoMVelocity(x_current);
            rerun_logger->logCoM(com_pos, com_vel, height_ref, com_vel_ref);
            
            // Log end effectors
            auto ee_positions = robot.getEndEffectorPositions();
            auto contact_states = robot.getContactStates(step);
            rerun_logger->logEndEffectors(ee_positions, contact_states, ee_pos_ref);
        }

        Eigen::VectorXd u_apply(robot.nu());
        #ifdef ENABLE_PROFILING
            auto t_mpc_start = std::chrono::steady_clock::now();
        #endif
        bool success = mpc.stepOnce(x_current, u_apply);
        #ifdef ENABLE_PROFILING
            auto t_mpc_end = std::chrono::steady_clock::now();
            prof_data["MPC_stepOnce"].times.push_back(std::chrono::duration<double, std::milli>(t_mpc_end - t_mpc_start).count());
            
            // Track peak memory within the loop
            double mem_current = getCurrentMemoryMB();
            if (mem_current > mem_peak) mem_peak = mem_current;
        #endif

        if (!success) {
            std::cerr << "MPC failed at step " << step << ", using gravity compensation." << std::endl;
            mj_forward(robot.model(), robot.data());
            for (int i = 0; i < robot.nu(); ++i) {
                u_apply(i) = robot.data()->qfrc_bias[i + 6];
            }
            if (step > 15) break;
        }

        if (!u_apply.allFinite()) {
            std::cerr << "NaN in control at step " << step << ", using zero control." << std::endl;
            u_apply.setZero();
        }

        // Log torques to Rerun
        if (rerun_logger) {
            auto torque_limits = robot.getTorqueLimits();
            auto joint_names = robot.getJointNames();
            rerun_logger->logTorques(u_apply, torque_limits, joint_names);
        }

        robot.setControl(u_apply);
        for (int sub_step = 0; sub_step < physics_steps_per_mpc; ++sub_step) {
            robot.step();
        }

        if (config.verbose) {
            double cost = mpc.getLastSolveCost();
            std::cout << "Step " << step << "/" << config.mpc.sim_steps
                      << " | Cost: " << cost
                      << " | (X,Y,Z): (" << x_current(0) << "," << x_current(1) << "," << x_current(2) << ") m"
                      << " | Control range: [" << u_apply.minCoeff() << ", " << u_apply.maxCoeff() << "]" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (config.save_trajectories) {
        mpc.finalizeOptimalTrajectoryLog();
    }

    std::cout << "Simulation completed in " << duration.count() << " ms\n";
    std::cout << "Average step time: " << duration.count() / static_cast<double>(config.mpc.sim_steps) << " ms\n";
}


// PROFILING RESULTS FUNCTION
#ifdef ENABLE_PROFILING
void printProfilingResults() {
    std::cout << "\n=== Performance Profiling ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- Timing Summary ---" << std::endl;
    std::cout << std::left << std::setw(20) << "Function"
              << std::right << std::setw(8) << "Calls"
              << std::setw(12) << "Total(ms)"
              << std::setw(12) << "Avg(ms)"
              << std::setw(12) << "Min(ms)"
              << std::setw(12) << "Max(ms)" << std::endl;
    std::cout << std::string(76, '-') << std::endl;

    for (const auto& entry : prof_data) {
        const auto& times = entry.second.times;
        if (times.empty()) continue;

        double total = 0.0, min_t = times[0], max_t = times[0];
        for (double t : times) {
            total += t;
            if (t < min_t) min_t = t;
            if (t > max_t) max_t = t;
        }
        double avg = total / times.size();

        std::cout << std::left << std::setw(20) << entry.first
                  << std::right << std::setw(8) << times.size()
                  << std::setw(12) << total
                  << std::setw(12) << avg
                  << std::setw(12) << min_t
                  << std::setw(12) << max_t << std::endl;
    }
}
#endif