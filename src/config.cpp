#include "config.hpp"
#include <iostream>

Config loadConfigFromFile(const std::string& filepath) {
    Config config;
    try {
        YAML::Node yaml_node = YAML::LoadFile(filepath);

        // Load top-level and robot parameters
        config.model_path = yaml_node["robot"]["model_path"].as<std::string>();
        config.urdf_path = yaml_node["robot"]["urdf_path"].as<std::string>();
        config.q_ref_path = yaml_node["reference_trajectory"]["q_ref"].as<std::string>();
        config.v_ref_path = yaml_node["reference_trajectory"]["v_ref"].as<std::string>();
        config.contact_schedule_path = yaml_node["reference_trajectory"]["contact_schedule"].as<std::string>();
        config.results_path = yaml_node["logging"]["results_path"].as<std::string>();
        config.verbose = yaml_node["logging"]["verbose"].as<bool>();
        config.save_trajectories = yaml_node["logging"]["save_trajectories"].as<bool>();

        // Load MPC parameters
        auto mpc_node = yaml_node["mpc"];
        config.mpc.horizon = mpc_node["horizon"].as<int>();
        config.mpc.dt = mpc_node["dt"].as<double>();
        config.mpc.physics_dt = mpc_node["physics_dt"].as<double>();
        config.mpc.gravity = mpc_node["gravity"].as<std::vector<double>>();
        config.mpc.sim_steps = mpc_node["sim_steps"].as<int>();
        config.mpc.contact_impratio = mpc_node["contact_impratio"].as<double>();

        // Load cost weights
        auto costs_node = mpc_node["cost_weights"];
        config.mpc.costs.Q_position_xy = costs_node["Q_position_xy"].as<double>();
        config.mpc.costs.Q_position_z = costs_node["Q_position_z"].as<double>();
        config.mpc.costs.Q_quat_w = costs_node["Q_quat_w"].as<double>();
        config.mpc.costs.Q_quat_xyz = costs_node["Q_quat_xyz"].as<std::vector<double>>();
        config.mpc.costs.Q_joint_pos = costs_node["Q_joint_pos"].as<double>();
        config.mpc.costs.Q_vel_xy = costs_node["Q_vel_xy"].as<double>();
        config.mpc.costs.Q_vel_z = costs_node["Q_vel_z"].as<double>();
        config.mpc.costs.Q_ang_vel = costs_node["Q_ang_vel"].as<double>();
        config.mpc.costs.Q_joint_vel = costs_node["Q_joint_vel"].as<double>();
        config.mpc.costs.R_control = costs_node["R_control"].as<double>();
        config.mpc.costs.Qf_multiplier = costs_node["Qf_multiplier"].as<double>();
        config.mpc.costs.Qf_position_xy = costs_node["Qf_position_xy"].as<double>();
        config.mpc.costs.Qf_position_z = costs_node["Qf_position_z"].as<double>();
        config.mpc.costs.Qf_vel_z = costs_node["Qf_vel_z"].as<double>();
        config.mpc.costs.W_com = costs_node["W_com"].as<double>();
        config.mpc.costs.W_foot = costs_node["W_foot"].as<double>();
        config.mpc.costs.W_foot_vel = costs_node["W_foot_vel"].as<double>();
        // Load constraints
        auto constraints_node = mpc_node["constraints"];
        config.mpc.joint_limit_weight = constraints_node["joint_limit_weight"].as<double>();
        config.mpc.torque_limit_weight = constraints_node["torque_limit_weight"].as<double>();

    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load or parse config.yaml: " << e.what() << std::endl;
        exit(1);
    }
    return config;
}

void Config::buildCostMatrices(int nx, int nu, int nq) {
    // Initialize all matrices as identity (all diagonal elements = 1.0)
    Q = Eigen::MatrixXd::Identity(nx, nx);
    R = Eigen::MatrixXd::Identity(nu, nu);
    Qf = Eigen::MatrixXd::Identity(nx, nx);
    
    // Build Q matrix (state deviation weights)
    
    // Position tracking weights (CoM position in world frame)
    Q(0, 0) = mpc.costs.Q_position_xy;   // X position
    Q(1, 1) = mpc.costs.Q_position_xy;   // Y position
    Q(2, 2) = mpc.costs.Q_position_z;    // Z position (critical)
    
    // Orientation tracking weights (quaternion representation)
    Q(3, 3) = mpc.costs.Q_quat_w;        // quat w (real part)
    Q(4, 4) = mpc.costs.Q_quat_xyz[0];   // quat x (roll)
    Q(5, 5) = mpc.costs.Q_quat_xyz[1];   // quat y (pitch)
    Q(6, 6) = mpc.costs.Q_quat_xyz[2];   // quat z (yaw)
    
    // Joint position tracking weights (actuated joints)
    for (int i = 7; i < nq; i++) {
        Q(i, i) = mpc.costs.Q_joint_pos;
    }
    
    // Velocity tracking weights (linear velocities)
    Q(nq + 0, nq + 0) = mpc.costs.Q_vel_xy;    // vx
    Q(nq + 1, nq + 1) = mpc.costs.Q_vel_xy;    // vy
    Q(nq + 2, nq + 2) = mpc.costs.Q_vel_z;     // vz (critical)
    
    // Angular velocity tracking weights
    Q(nq + 3, nq + 3) = mpc.costs.Q_ang_vel;   // omega_x
    Q(nq + 4, nq + 4) = mpc.costs.Q_ang_vel;   // omega_y
    Q(nq + 5, nq + 5) = mpc.costs.Q_ang_vel;   // omega_z
    
    // Joint velocity tracking weights
    for (int i = nq + 6; i < nx; i++) {
        Q(i, i) = mpc.costs.Q_joint_vel;
    }
    
    // Build R matrix (control effort regularization)
    R *= mpc.costs.R_control;
    
    // Build Qf matrix (terminal cost weights)
    
    // Start with Q scaled by multiplier
    Qf = Q * mpc.costs.Qf_multiplier;
    
    // Apply additional terminal weight multipliers to specific states
    Qf(0, 0) *= mpc.costs.Qf_position_xy;      // Final X position
    Qf(1, 1) *= mpc.costs.Qf_position_xy;      // Final Y position
    Qf(2, 2) *= mpc.costs.Qf_position_z;       // Final Z position
    Qf(nq + 2, nq + 2) *= mpc.costs.Qf_vel_z;  // Final Z velocity (critical)
    
    std::cout << "Cost matrices built: Q(" << nx << "x" << nx 
              << "), R(" << nu << "x" << nu 
              << "), Qf(" << nx << "x" << nx << ")" << std::endl;
}