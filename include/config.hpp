#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

// Struct to hold cost function weights
struct CostWeights {
    double Q_position_xy, Q_position_z, Q_quat_w;
    std::vector<double> Q_quat_xyz;
    double Q_joint_pos, Q_vel_xy, Q_vel_z, Q_ang_vel, Q_joint_vel;
    double R_control;
    double Qf_multiplier, Qf_position_xy, Qf_position_z, Qf_vel_z;
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
    MpcParams mpc;
    
    // Pre-built cost matrices (constructed after loading robot dimensions)
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd Qf;
    
    // Build cost matrices based on robot dimensions
    void buildCostMatrices(int nx, int nu, int nq);
};

// Function declaration for loading the config
Config loadConfigFromFile(const std::string& filepath);