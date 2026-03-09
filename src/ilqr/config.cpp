#include "ilqr/config.hpp"
#include <iostream>

Config loadConfigFromFile(const std::string& filepath) {
    Config config;
    try {
        YAML::Node yaml_node = YAML::LoadFile(filepath);

        // Load top-level and robot parameters
        config.model_path = yaml_node["robot"]["model_path"].as<std::string>();
        config.urdf_path = yaml_node["robot"]["urdf_path"].as<std::string>();
        
        // Load end-effector body names
        if (!yaml_node["robot"]["ee_feet"]) {
            std::cerr << "ERROR: config.yaml missing 'robot.ee_feet' section!" << std::endl;
            std::cerr << "Required format:" << std::endl;
            std::cerr << "  ee_feet:" << std::endl;
            std::cerr << "    left_feet_ee: \"foot_left\"" << std::endl;
            std::cerr << "    right_feet_ee: \"foot_right\"" << std::endl;
            exit(1);
        }
        auto ee_feet_node = yaml_node["robot"]["ee_feet"];
        config.left_foot_body_name = ee_feet_node["left_feet_ee"].as<std::string>();
        config.right_foot_body_name = ee_feet_node["right_feet_ee"].as<std::string>();
        // Pelvis body name for Pelvis/Feet cost
        if (yaml_node["robot"]["pelvis_body_name"]) {
            config.pelvis_body_name = yaml_node["robot"]["pelvis_body_name"].as<std::string>();
        } else {
            config.pelvis_body_name = "pelvis";
        }
        if (yaml_node["robot"]["torso_body_name"]) {
            config.torso_body_name = yaml_node["robot"]["torso_body_name"].as<std::string>();
        } else {
            config.torso_body_name = "torso";   // Default for DM humanoid
        }
        if (yaml_node["robot"]["waist_lower_body_name"]) {
            config.waist_lower_body_name = yaml_node["robot"]["waist_lower_body_name"].as<std::string>();
        } else {
            config.waist_lower_body_name = "waist_lower";  // Default for DM humanoid
        }
        
        config.q_ref_path = yaml_node["reference_trajectory"]["q_ref"].as<std::string>();
        config.v_ref_path = yaml_node["reference_trajectory"]["v_ref"].as<std::string>();
        config.contact_schedule_path = yaml_node["reference_trajectory"]["contact_schedule"].as<std::string>();
        config.results_path = yaml_node["logging"]["results_path"].as<std::string>();
        config.verbose = yaml_node["logging"]["verbose"].as<bool>();
        config.save_trajectories = yaml_node["logging"]["save_trajectories"].as<bool>();
        
        // Load visualization parameters
        config.enable_rerun = yaml_node["visualization"]["enable_rerun"].as<bool>(false);  // Default to false

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

        // Posture cost (joint angles [7:nq], Quadratic norm)
        config.mpc.costs.W_posture          = costs_node["posture"]["weight"].as<double>();
        config.mpc.costs.W_posture_terminal = costs_node["posture"]["terminal_weight"].as<double>();

        // Control regularization
        config.mpc.costs.R_control = costs_node["control"]["R_control"].as<double>();

        // Task-specific weights
        config.mpc.costs.W_height   = costs_node["W_height"].as<double>();
        config.mpc.costs.W_vel      = costs_node["W_vel"].as<double>();
        config.mpc.costs.W_joint_vel = costs_node["W_joint_vel"].as<double>();
        config.mpc.costs.W_upright     = costs_node["W_upright"].as<double>();
        config.mpc.costs.w_balance     = costs_node["w_balance"].as<double>();
        config.mpc.costs.W_pelvis_feet = costs_node["W_pelvis_feet"].as<double>(1.0);
        config.mpc.costs.W_walk        = costs_node["W_walk"].as<double>(1.0);
        config.mpc.costs.speed_goal    = costs_node["speed_goal"].as<double>(0.0);
        
        // Load constraints
        auto constraints_node = mpc_node["constraints"];
        config.mpc.joint_limit_weight = constraints_node["joint_limit_weight"].as<double>();
        config.mpc.torque_limit_weight = constraints_node["torque_limit_weight"].as<double>();
        
        // Load iLQR solver settings
        if (mpc_node["ilqr_settings"]) {
            auto solver_node = mpc_node["ilqr_settings"];
            config.mpc.ilqr_settings.initial_regularization = solver_node["initial_regularization"].as<double>();
            config.mpc.ilqr_settings.max_iterations = solver_node["max_iterations"].as<int>();
            config.mpc.ilqr_settings.tolerance = solver_node["tolerance"].as<double>();
            config.mpc.ilqr_settings.reg_min = solver_node["reg_min"].as<double>();
            config.mpc.ilqr_settings.reg_max = solver_node["reg_max"].as<double>();
            config.mpc.ilqr_settings.reg_increase_factor = solver_node["reg_increase_factor"].as<double>();
            config.mpc.ilqr_settings.reg_decrease_factor = solver_node["reg_decrease_factor"].as<double>();
            config.mpc.ilqr_settings.trust_region_good = solver_node["trust_region_good"].as<double>();
            config.mpc.ilqr_settings.trust_region_poor = solver_node["trust_region_poor"].as<double>();
            config.mpc.ilqr_settings.num_line_search_steps = solver_node["num_line_search_steps"].as<int>();
            config.mpc.ilqr_settings.min_linesearch_step = solver_node["min_linesearch_step"].as<double>();
            config.mpc.ilqr_settings.line_search_tolerance = solver_node["line_search_tolerance"].as<double>();
            config.mpc.ilqr_settings.quu_regularization = solver_node["quu_regularization"].as<double>();
            config.mpc.ilqr_settings.convergence_threshold = solver_node["convergence_threshold"].as<double>();
        } else {
            // Default values if not specified
            config.mpc.ilqr_settings.initial_regularization = 1e-6;
            config.mpc.ilqr_settings.max_iterations = 10;
            config.mpc.ilqr_settings.tolerance = 1e-4;
            config.mpc.ilqr_settings.reg_min = 1e-6;
            config.mpc.ilqr_settings.reg_max = 100.0;
            config.mpc.ilqr_settings.reg_increase_factor = 10.0;
            config.mpc.ilqr_settings.reg_decrease_factor = 10.0;
            config.mpc.ilqr_settings.trust_region_good = 0.75;
            config.mpc.ilqr_settings.trust_region_poor = 0.25;
            config.mpc.ilqr_settings.num_line_search_steps = 10;
            config.mpc.ilqr_settings.min_linesearch_step = 1e-3;
            config.mpc.ilqr_settings.line_search_tolerance = 1e-6;
            config.mpc.ilqr_settings.quu_regularization = 1e-4;
            config.mpc.ilqr_settings.convergence_threshold = 1e-8;
        }
        
        // Load norm types for cost terms
        if (mpc_node["norm_types"]) {
            const YAML::Node& norm_types = mpc_node["norm_types"];
            
            for (YAML::const_iterator it = norm_types.begin(); it != norm_types.end(); ++it) {
                std::string cost_name = it->first.as<std::string>();
                int type = it->second["type"].as<int>();
                double p = it->second["p"].as<double>();
                double q = it->second["q"].as<double>();
                
                config.norm_params[cost_name] = ilqr::NormParams{
                    static_cast<ilqr::NormType>(type), p, q
                };
            }
        }

    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load or parse config.yaml: " << e.what() << std::endl;
        exit(1);
    }
    return config;
}

void Config::buildCostMatrices(int nx, int nu, int nq) {
    // --- Q: posture cost ---
    // Only joint angles qpos[7:nq] are penalised (DeepMind "Posture", Quadratic).
    // Base DOF (indices 0-6: xyz + quaternion) and all velocity rows stay zero.
    Q = Eigen::MatrixXd::Zero(nx, nx);
    for (int i = 7; i < nq; ++i)
        Q(i, i) = mpc.costs.W_posture;

    // --- R: control regularization (uniform across all actuators) ---
    R = Eigen::MatrixXd::Identity(nu, nu) * mpc.costs.R_control;

    // --- Qf: terminal posture cost (same structure, independent weight) ---
    Qf = Eigen::MatrixXd::Zero(nx, nx);
    for (int i = 7; i < nq; ++i)
        Qf(i, i) = mpc.costs.W_posture_terminal;

    std::cout << "Cost matrices built: Q(" << nx << "x" << nx
              << "), R(" << nu << "x" << nu
              << "), Qf(" << nx << "x" << nx << ")" << std::endl;
}