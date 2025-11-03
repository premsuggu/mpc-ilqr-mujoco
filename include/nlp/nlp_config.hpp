#pragma once

#include <string>
#include <vector>

namespace nlp {

/**
 * @brief Cost weight configuration
 */
struct CostWeights {
    double w_q = 10.0;           // Position tracking weight
    double w_v = 1.0;            // Velocity tracking weight
    double w_com = 100.0;        // CoM tracking weight
    double w_ee_pos = 400.0;     // End-effector position tracking (swing)
    double w_ee_vel = 0.0;       // End-effector velocity tracking (stance)
    double w_upright = 20.0;     // Upright posture penalty
    double w_a = 0.01;           // Acceleration regularization
    double w_f = 0.001;          // Force regularization
    double terminal_multiplier = 5.0;  // Terminal cost multiplier
};

/**
 * @brief IPOPT solver options
 */
struct SolverOptions {
    int max_iter = 500;
    double tol = 1e-4;
    double acceptable_tol = 1e-3;
    double mu_init = 1e-3;
    int print_level = 3;
    bool use_hsl = true;        // Try to use HSL solvers if available
    std::string linear_solver = "ma57";  // HSL solver (if available)
};

/**
 * @brief NLP MPC configuration
 */
struct NLPConfig {
    // Horizon parameters
    int N = 15;                  // Horizon length
    double dt = 0.05;            // Time step (s)
    
    // Robot configuration
    std::string urdf_path = "robots/h1_description/urdf/h1.urdf";
    std::vector<std::string> ee_names = {"left_ankle_link", "right_ankle_link"};
    
    // Reference trajectories
    std::string q_ref_path = "data/q_standing.csv";
    std::string v_ref_path = "data/v_standing.csv";
    
    // Cost weights
    CostWeights weights;
    
    // Solver options
    SolverOptions solver_options;
    
    // MPC execution
    int num_mpc_steps = 50;      // Number of MPC iterations to run
    
    // Output paths
    std::string output_dir = "results/";
};

} // namespace nlp
