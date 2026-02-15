#pragma once

#include <string>
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include "ilqr/norm.hpp"

// Struct to hold cost function weights
struct CostWeights {
    double Q_position_x, Q_position_y, Q_position_z, Q_quat_w;
    std::vector<double> Q_quat_xyz;
    double Q_joint_pos, Q_vel_x, Q_vel_y, Q_vel_z, Q_ang_vel, Q_joint_vel;
    double R_control;
    double Qf_multiplier, Qf_position_x, Qf_position_y, Qf_position_z, Qf_vel_z;
    double W_com, W_com_vel, W_foot, W_foot_vel;
    double W_upright;
    double w_balance;
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