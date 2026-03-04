#pragma once

#include <string>
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include "ilqr/norm.hpp"

// Struct to hold cost function weights
struct CostWeights {
    // Posture cost: penalises joint angles qpos[7:nq] from reference (Quadratic)
    double W_posture;           // Running cost weight
    double W_posture_terminal;  // Terminal cost weight

    // Control regularization (R matrix, Quadratic)
    double R_control;

    // Task-specific weights (used by addXxxCostDerivatives in ilqr.cpp)
    double W_height, W_vel;
    double W_upright, w_balance;
    double W_pelvis_feet;        // DeepMind "Pelvis/Feet" = 1.0
    double W_walk;               // DeepMind "Walk" = 1.0
    double speed_goal;           // Target forward speed (m/s); 0.0 = standing task
};

// Struct to hold MPC parameters
struct MpcParams {
    int horizon;
    double dt, physics_dt;
    std::vector<double> gravity;
    int sim_steps;
    double contact_impratio;
    CostWeights costs;
    double joint_limit_weight;
    double torque_limit_weight;
    
    // iLQR solver settings
    struct ILQRSettings {
        double initial_regularization;
        int max_iterations;
        double tolerance;
        double reg_min;
        double reg_max;
        double reg_increase_factor;
        double reg_decrease_factor;
        double trust_region_good;
        double trust_region_poor;
        int num_line_search_steps;        // Number of line search candidates
        double min_linesearch_step;       // Minimum line search step size
        double line_search_tolerance;
        double quu_regularization;
        double convergence_threshold;
    } ilqr_settings;
};

// Main Config struct to hold everything
struct Config {
    std::string model_path;
    std::string urdf_path;
    std::string q_ref_path;
    std::string v_ref_path;
    std::string contact_schedule_path;
    std::string results_path;
    bool verbose;
    bool save_trajectories;
    bool enable_rerun;  // Enable Rerun visualization
    
    // End-effector body names (loaded from config.yaml)
    std::string left_foot_body_name;
    std::string right_foot_body_name;
    std::string pelvis_body_name;       // Body for Pelvis/Feet cost z-position (e.g. "pelvis")
    std::string torso_body_name;        // Floating-base body for Walk forward direction (e.g. "torso")
    std::string waist_lower_body_name;  // Body for Walk com_vel: 0.5*(subtreeLinVel+torso_vel)
    
    MpcParams mpc;
    
    // Pre-built cost matrices (constructed after loading robot dimensions)
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd Qf;
    
    // Norm parameters for each cost term
    std::map<std::string, ilqr::NormParams> norm_params;
    
    // Build cost matrices based on robot dimensions
    void buildCostMatrices(int nx, int nu, int nq);
};

// Function declaration for loading the config
Config loadConfigFromFile(const std::string& filepath);