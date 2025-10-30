#include "common/robot_utils.hpp"
#include "ilqr/ilqr.hpp"
#include "ilqr/mpc.hpp"
#include "common/config.hpp"
#include <iostream>
#include <chrono>
#include <map>
#include <vector>

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
void runSimulation(RobotUtils& robot, MPC& mpc, const Config& config);
#ifdef ENABLE_PROFILING
void printProfilingResults();
#endif

 
// MAIN FUNCTION
int main() {
    Config config = loadConfigFromFile("config.yaml");
    std::cout << "Configuration loaded successfully from config.yaml" << std::endl;

    RobotUtils robot;
    setupSimulation(robot, config);
    MPC mpc(robot, config.mpc.horizon, config.mpc.dt, config.urdf_path);

    #ifdef ENABLE_PROFILING
        double mem_initial = getCurrentMemoryMB();
        mem_peak = mem_initial; // Initialize peak memory
        std::cout << "=== Profiling ENABLED ===" << std::endl;
        std::cout << "Initial memory: " << std::fixed << std::setprecision(2) << mem_initial << " MB" << std::endl;
    #endif

    runSimulation(robot, mpc, config);

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
    if (!robot.loadModel(config.model_path)) {
        throw std::runtime_error("Failed to load robot model from: " + config.model_path);
    }
    robot.setContactImpratio(config.mpc.contact_impratio);
    robot.setTimeStep(config.mpc.physics_dt);
    robot.setGravity(config.mpc.gravity[0], config.mpc.gravity[1], config.mpc.gravity[2]);
    robot.initializeStandingPose();
    std::cout << "Model loaded: nx=" << robot.nx() << ", nu=" << robot.nu() << std::endl;
    config.buildCostMatrices(robot.nx(), robot.nu(), robot.nq());
    robot.setCostWeights(config.Q, config.R, config.Qf);
    robot.setCoMWeight(config.mpc.costs.W_com);
    robot.setCoMVelWeight(config.mpc.costs.W_com_vel);
    robot.setEEPosWeight(config.mpc.costs.W_foot); 
    robot.setEEVelWeight(config.mpc.costs.W_foot_vel);
    robot.setUprightWeight(config.mpc.costs.W_upright);
    robot.setBalanceWeight(config.mpc.costs.w_balance);
    robot.setConstraintWeights(config.mpc.joint_limit_weight, config.mpc.torque_limit_weight);
    if (!robot.loadReferences(config.q_ref_path, config.v_ref_path)) {
        throw std::runtime_error("Failed to load reference trajectories.");
    }
    if (!robot.loadContactSchedule(config.contact_schedule_path)) {
        std::cerr << "Warning: Failed to load contact schedule, continuing without it." << std::endl;
    }
}

 
// SIMULATION LOOP FUNCTION
void runSimulation(RobotUtils& robot, MPC& mpc, const Config& config) {
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