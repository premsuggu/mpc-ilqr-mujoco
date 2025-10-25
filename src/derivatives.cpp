#include "derivatives.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <iostream>

namespace derivatives {

// Convert MuJoCo state to Pinocchio-compatible state (fix quaternion ordering)
Eigen::VectorXd convertMuJoCoToPinocchio(const Eigen::VectorXd& mujoco_state, int nq) {
    Eigen::VectorXd pinocchio_state = mujoco_state;  // Copy all first
    
    // Convert quaternion: MuJoCo [qw,qx,qy,qz] -> Pinocchio [qx,qy,qz,qw]
    if (nq >= 7) {  // Has floating base with quaternion
        pinocchio_state(3) = mujoco_state(4);  // qx
        pinocchio_state(4) = mujoco_state(5);  // qy  
        pinocchio_state(5) = mujoco_state(6);  // qz
        pinocchio_state(6) = mujoco_state(3);  // qw
    }
    
    return pinocchio_state;
}

EEDerivatives::EEDerivatives(const std::string& urdf_path, bool floating_base) {
    // Load Pinocchio model
    if (floating_base) {
        pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_);
    } else {
        pinocchio::urdf::buildModel(urdf_path, model_);
    }
    data_ = pinocchio::Data(model_);
    
    std::cout << "Loaded robot: " << model_.nq << " DOF" << std::endl;
    
    // Build symbolic computation framework once
    buildSymbolicFunctions();
}

void EEDerivatives::buildSymbolicFunctions() {
    // Create symbolic full state vector [q, v]
    int nx = model_.nq + model_.nv;  // Full state size
    nx_ = nx;  // Cache for later use
    x_sym_ = ::casadi::SX::sym("x", nx);
    
    // Create CasADi-compatible model for symbolic computations
    ad_model_ = model_.template cast<ADScalar>();
    ad_data_ = pinocchio::DataTpl<ADScalar>(ad_model_);
    
    // Initialize CoM functions flag
    com_functions_built_ = false;
    upright_functions_built_ = false;
    balance_functions_built_ = false;
    
    std::cout << "Built symbolic computation framework for state size " << nx 
              << " (nq=" << model_.nq << ", nv=" << model_.nv << ")" << std::endl;
}

void EEDerivatives::buildEEFunctions(const std::string& frame_name) {
    // Get frame ID
    pinocchio::FrameIndex frame_id = getFrameId(frame_name);
    
    // Create symbolic input parameters
    ::casadi::SX target_sym = ::casadi::SX::sym("target", 3);
    ::casadi::SX weight_sym = ::casadi::SX::sym("weight");
    
    // Set up symbolic configuration for kinematics
    // Extract q from full state x = [q, v]
    typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> ConfigVector;
    ConfigVector q_ad(model_.nq);
    for (int i = 0; i < model_.nq; i++) {
        q_ad[i] = x_sym_(i);  // First nq elements of full state
    }
    
    // Symbolic forward kinematics
    pinocchio::forwardKinematics(ad_model_, ad_data_, q_ad);
    pinocchio::updateFramePlacements(ad_model_, ad_data_);
    
    // Extract end-effector position
    auto ee_transform = ad_data_.oMf[frame_id];
    ::casadi::SX ee_pos = ::casadi::SX::vertcat({
        ee_transform.translation()[0],
        ee_transform.translation()[1], 
        ee_transform.translation()[2]
    });
    
    // Position error and cost
    ::casadi::SX pos_error = ee_pos - target_sym;
    ::casadi::SX cost = weight_sym * ::casadi::SX::dot(pos_error, pos_error);
    
    // Build cached functions
    // 1. Position function (for debugging/validation) - only depends on q part
    ::casadi::SX q_only = x_sym_(::casadi::Slice(0, model_.nq));
    ee_pos_fns_[frame_name] = ::casadi::Function(
        "ee_pos_" + frame_name, 
        {q_only}, {ee_pos}
    );
    
    // 2. Gradient function w.r.t. full state [q, v]
    ::casadi::SX grad = ::casadi::SX::gradient(cost, x_sym_);
    ee_grad_fns_[frame_name] = ::casadi::Function(
        "ee_grad_" + frame_name,
        {x_sym_, target_sym, weight_sym}, {grad}
    );
    
    // 3. Hessian function w.r.t. full state [q, v]
    ::casadi::SX hess = ::casadi::SX::jacobian(grad, x_sym_);
    ee_hess_fns_[frame_name] = ::casadi::Function(
        "ee_hess_" + frame_name,
        {x_sym_, target_sym, weight_sym}, {hess}
    );
    
    std::cout << "Built cached EE position functions for frame: " << frame_name << std::endl;
}

void EEDerivatives::buildEEVelFunctions(const std::string& frame_name) {
    // Get frame ID
    pinocchio::FrameIndex frame_id = getFrameId(frame_name);
    
    // Create symbolic input parameters
    ::casadi::SX target_vel_sym = ::casadi::SX::sym("target_vel", 3);
    ::casadi::SX weight_sym = ::casadi::SX::sym("weight");
    
    // Extract q and v from full state x = [q, v]
    typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> ConfigVector;
    ConfigVector q_ad(model_.nq), v_ad(model_.nv);
    for (int i = 0; i < model_.nq; i++) {
        q_ad[i] = x_sym_(i);
    }
    for (int i = 0; i < model_.nv; i++) {
        v_ad[i] = x_sym_(model_.nq + i);
    }
    
    // Create a LOCAL ad_data object to avoid corrupting the shared one
    pinocchio::DataTpl<ADScalar> local_ad_data(ad_model_);
    
    // Forward kinematics with velocities (using local data)
    pinocchio::forwardKinematics(ad_model_, local_ad_data, q_ad, v_ad);
    pinocchio::updateFramePlacements(ad_model_, local_ad_data);
    
    // Get end-effector velocity (linear part only, in world frame)
    auto frame_vel = pinocchio::getFrameVelocity(ad_model_, local_ad_data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    ::casadi::SX ee_vel = ::casadi::SX::vertcat({
        frame_vel.linear()[0],
        frame_vel.linear()[1],
        frame_vel.linear()[2]
    });
    
    // Velocity error and cost
    ::casadi::SX vel_error = ee_vel - target_vel_sym;
    ::casadi::SX vel_cost = weight_sym * ::casadi::SX::dot(vel_error, vel_error);
    
    // Build cached functions
    ::casadi::SX vel_grad = ::casadi::SX::gradient(vel_cost, x_sym_);
    ee_vel_grad_fns_[frame_name] = ::casadi::Function(
        "ee_vel_grad_" + frame_name,
        {x_sym_, target_vel_sym, weight_sym}, {vel_grad}
    );
    
    ::casadi::SX vel_hess = ::casadi::SX::jacobian(vel_grad, x_sym_);
    ee_vel_hess_fns_[frame_name] = ::casadi::Function(
        "ee_vel_hess_" + frame_name,
        {x_sym_, target_vel_sym, weight_sym}, {vel_hess}
    );
    
    std::cout << "Built cached EE velocity functions for frame: " << frame_name << std::endl;
}

void EEDerivatives::buildCoMFunctions() {
    // Create symbolic input parameters
    ::casadi::SX target_com_sym = ::casadi::SX::sym("target_com", 3);
    ::casadi::SX weight_sym = ::casadi::SX::sym("weight");
    
    // Set up symbolic configuration for kinematics
    typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> ConfigVector;
    ConfigVector q_ad(model_.nq);
    for (int i = 0; i < model_.nq; i++) {
        q_ad[i] = x_sym_(i);  // First nq elements of full state
    }
    
    // Symbolic forward kinematics and CoM computation
    pinocchio::forwardKinematics(ad_model_, ad_data_, q_ad);
    pinocchio::centerOfMass(ad_model_, ad_data_, q_ad);  // Compute CoM
    
    // Extract symbolic CoM position components
    ::casadi::SX com_x = ad_data_.com[0][0];  // x component
    ::casadi::SX com_y = ad_data_.com[0][1];  // y component  
    ::casadi::SX com_z = ad_data_.com[0][2];  // z component
    
    std::vector<::casadi::SX> com_components;
    com_components.push_back(com_x);
    com_components.push_back(com_y);
    com_components.push_back(com_z);
    ::casadi::SX com_pos = ::casadi::SX::vertcat(com_components);
    
    // Position error and cost
    ::casadi::SX com_error = com_pos - target_com_sym;
    ::casadi::SX com_cost = weight_sym * ::casadi::SX::dot(com_error, com_error);
    
    // Build cached CoM functions
    ::casadi::SX com_grad = ::casadi::SX::gradient(com_cost, x_sym_);
    com_grad_fn_ = ::casadi::Function(
        "com_grad",
        {x_sym_, target_com_sym, weight_sym}, {com_grad}
    );
    
    ::casadi::SX com_hess = ::casadi::SX::jacobian(com_grad, x_sym_);
    com_hess_fn_ = ::casadi::Function(
        "com_hess", 
        {x_sym_, target_com_sym, weight_sym}, {com_hess}
    );
    
    com_functions_built_ = true;
    std::cout << "Built cached CoM functions" << std::endl;
}

// For keeping the robot upright
void EEDerivatives::buildUprightFunctions() {
    // Extract quaternion from state: x = [q, v] where q = [pos(3), quat(4), joints...]
    // Quaternion indices in q: [3, 4, 5, 6] for [qw, qx, qy, qz]
    casadi::SX qw = x_sym_(3);
    casadi::SX qx = x_sym_(4);
    casadi::SX qy = x_sym_(5);
    casadi::SX qz = x_sym_(6);
    
    // Compute torso z-axis in world frame (3rd column of rotation matrix)
    casadi::SX z_torso_x = 2*(qx*qz + qw*qy);
    casadi::SX z_torso_y = 2*(qy*qz - qw*qx);
    casadi::SX z_torso_z = 1 - 2*(qx*qx + qy*qy);
    
    // World z-axis target
    casadi::SX z_world_x = 0.0;
    casadi::SX z_world_y = 0.0;
    casadi::SX z_world_z = 1.0;
    
    // Residual: r = z_torso - z_world
    casadi::SX rx = z_torso_x - z_world_x;
    casadi::SX ry = z_torso_y - z_world_y;
    casadi::SX rz = z_torso_z - z_world_z;
    
    // Weight parameter
    casadi::SX w_upright = casadi::SX::sym("w_upright");
    
    // Cost: L = 0.5 * w * ||r||²
    casadi::SX cost = 0.5 * w_upright * (rx*rx + ry*ry + rz*rz);
    
    // Compute gradient: ∂L/∂x (automatic differentiation!)
    casadi::SX grad = casadi::SX::gradient(cost, x_sym_);
    
    // Compute Hessian: ∂²L/∂x² (automatic differentiation!)
    casadi::SX hess = casadi::SX::hessian(cost, x_sym_);
    
    // Create CasADi functions
    upright_grad_fn_ = casadi::Function("upright_grad", 
                                        {x_sym_, w_upright}, 
                                        {grad});
    
    upright_hess_fn_ = casadi::Function("upright_hess", 
                                        {x_sym_, w_upright}, 
                                        {hess});
}

void EEDerivatives::prepareFrame(const std::string& frame_name) {
    // Build position functions if not exist
    if (ee_grad_fns_.find(frame_name) == ee_grad_fns_.end()) {
        buildEEFunctions(frame_name);
    }
    // NOTE: Velocity functions are built on-demand in EEvelGrad/EEvelHess
    // to avoid any potential interference with position functions
}

Eigen::VectorXd EEDerivatives::EEposGrad(const Eigen::VectorXd& x, 
                                         const Eigen::Vector3d& target_pos,
                                         const std::string& frame_name,
                                         double weight) {
    
    // Ensure functions are built for this frame
    prepareFrame(frame_name);
    
    // Convert MuJoCo state to Pinocchio state (fix quaternion ordering)
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM target_dm = ::casadi::DM({target_pos[0], target_pos[1], target_pos[2]});
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM grad_dm = ee_grad_fns_[frame_name](::casadi::DMVector{x_dm, target_dm, weight_dm})[0];
    
    // Convert back to Eigen (full state size)
    int nx = model_.nq + model_.nv;
    Eigen::VectorXd gradient(nx);
    for (int i = 0; i < nx; i++) {
        gradient(i) = double(grad_dm(i));
    }
    
    return gradient;
}

Eigen::MatrixXd EEDerivatives::EEposHess(const Eigen::VectorXd& x,
                                         const Eigen::Vector3d& target_pos,
                                         const std::string& frame_name,
                                         double weight) {
    
    // Ensure functions are built for this frame
    prepareFrame(frame_name);
    
    // Convert MuJoCo state to Pinocchio state (fix quaternion ordering)
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM target_dm = ::casadi::DM({target_pos[0], target_pos[1], target_pos[2]});
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM hess_dm = ee_hess_fns_[frame_name](::casadi::DMVector{x_dm, target_dm, weight_dm})[0];
    
    // Convert back to Eigen (full state size)
    int nx = model_.nq + model_.nv;
    Eigen::MatrixXd hessian(nx, nx);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nx; j++) {
            hessian(i, j) = double(hess_dm(i, j));
        }
    }
    
    return hessian;
}

Eigen::VectorXd EEDerivatives::CoMGrad(const Eigen::VectorXd& x,
                                       const Eigen::Vector3d& target_com,
                                       double weight) {
    
    // Ensure CoM functions are built
    if (!com_functions_built_) {
        buildCoMFunctions();
    }
    
    // Convert MuJoCo state to Pinocchio state (fix quaternion ordering)
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM target_dm = ::casadi::DM({target_com[0], target_com[1], target_com[2]});
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM grad_dm = com_grad_fn_(::casadi::DMVector{x_dm, target_dm, weight_dm})[0];
    
    // Convert back to Eigen (full state size)
    int nx = model_.nq + model_.nv;
    Eigen::VectorXd gradient(nx);
    for (int i = 0; i < nx; i++) {
        gradient(i) = double(grad_dm(i));
    }
    
    return gradient;
}

Eigen::MatrixXd EEDerivatives::CoMHess(const Eigen::VectorXd& x,
                                       const Eigen::Vector3d& target_com,
                                       double weight) {
    
    // Ensure CoM functions are built
    if (!com_functions_built_) {
        buildCoMFunctions();
    }
    
    // Convert MuJoCo state to Pinocchio state (fix quaternion ordering)
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM target_dm = ::casadi::DM({target_com[0], target_com[1], target_com[2]});
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM hess_dm = com_hess_fn_(::casadi::DMVector{x_dm, target_dm, weight_dm})[0];
    
    // Convert back to Eigen (full state size)
    int nx = model_.nq + model_.nv;
    Eigen::MatrixXd hessian(nx, nx);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nx; j++) {
            hessian(i, j) = double(hess_dm(i, j));
        }
    }
    
    return hessian;
}

Eigen::VectorXd EEDerivatives::EEvelGrad(const Eigen::VectorXd& x,
                                         const Eigen::Vector3d& target_vel,
                                         const std::string& frame_name,
                                         double weight) {
    
    // Build velocity functions on-demand
    if (ee_vel_grad_fns_.find(frame_name) == ee_vel_grad_fns_.end()) {
        buildEEVelFunctions(frame_name);
    }
    
    // Convert MuJoCo state to Pinocchio state (fix quaternion ordering)
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM target_dm = ::casadi::DM({target_vel[0], target_vel[1], target_vel[2]});
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM grad_dm = ee_vel_grad_fns_[frame_name](::casadi::DMVector{x_dm, target_dm, weight_dm})[0];
    
    // Convert back to Eigen (full state size)
    int nx = model_.nq + model_.nv;
    Eigen::VectorXd gradient(nx);
    for (int i = 0; i < nx; i++) {
        gradient(i) = double(grad_dm(i));
    }
    
    return gradient;
}

Eigen::MatrixXd EEDerivatives::EEvelHess(const Eigen::VectorXd& x,
                                         const Eigen::Vector3d& target_vel,
                                         const std::string& frame_name,
                                         double weight) {
    
    // Build velocity functions on-demand
    if (ee_vel_hess_fns_.find(frame_name) == ee_vel_hess_fns_.end()) {
        buildEEVelFunctions(frame_name);
    }
    
    // Convert MuJoCo state to Pinocchio state (fix quaternion ordering)
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM target_dm = ::casadi::DM({target_vel[0], target_vel[1], target_vel[2]});
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM hess_dm = ee_vel_hess_fns_[frame_name](::casadi::DMVector{x_dm, target_dm, weight_dm})[0];
    
    // Convert back to Eigen (full state size)
    int nx = model_.nq + model_.nv;
    Eigen::MatrixXd hessian(nx, nx);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nx; j++) {
            hessian(i, j) = double(hess_dm(i, j));
        }
    }
    
    return hessian;
}

// Upright posture derivatives
Eigen::VectorXd EEDerivatives::UprightGrad(const Eigen::VectorXd& x, double w_upright) {
    // Build functions if not yet built
    if (!upright_functions_built_) {
        buildUprightFunctions();
        upright_functions_built_ = true;
    }
    
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM w_dm = ::casadi::DM(w_upright);
    
    // Evaluate function using proper CasADi vector syntax
    ::casadi::DM grad_dm = upright_grad_fn_(::casadi::DMVector{x_dm, w_dm})[0];
    
    // Convert back to Eigen
    std::vector<double> grad_vec = grad_dm.get_elements();
    return Eigen::Map<Eigen::VectorXd>(grad_vec.data(), grad_vec.size());
}

Eigen::MatrixXd EEDerivatives::UprightHess(const Eigen::VectorXd& x, double w_upright) {
    // Build functions if not yet built
    if (!upright_functions_built_) {
        buildUprightFunctions();
        upright_functions_built_ = true;
    }
    
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM w_dm = ::casadi::DM(w_upright);
    
    // Evaluate function using proper CasADi vector syntax
    ::casadi::DM hess_dm = upright_hess_fn_(::casadi::DMVector{x_dm, w_dm})[0];
    
    // Convert back to Eigen
    std::vector<double> hess_vec = hess_dm.get_elements();
    Eigen::MatrixXd hess = Eigen::Map<Eigen::MatrixXd>(hess_vec.data(), nx_, nx_);
    
    // Ensure symmetry (CasADi Hessian should already be symmetric)
    return 0.5 * (hess + hess.transpose());
}


pinocchio::FrameIndex EEDerivatives::getFrameId(const std::string& frame_name) {
    if (!model_.existFrame(frame_name)) {
        throw std::runtime_error("Frame '" + frame_name + "' not found");
    }
    return model_.getFrameId(frame_name);
}

void EEDerivatives::buildBalanceFunctions() {
    // Extract q and v from symbolic state
    typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> ConfigVector;
    ConfigVector q_ad(model_.nq), v_ad(model_.nv);
    for (int i = 0; i < model_.nq; i++) q_ad[i] = x_sym_(i);
    for (int i = 0; i < model_.nv; i++) v_ad[i] = x_sym_(model_.nq + i);
    
    // Compute center of mass and its velocity symbolically
    pinocchio::forwardKinematics(ad_model_, ad_data_, q_ad, v_ad);
    pinocchio::centerOfMass(ad_model_, ad_data_, q_ad, v_ad);
    
    // Extract CoM position and velocity
    casadi::SX pcom_x = ad_data_.com[0][0];
    casadi::SX pcom_y = ad_data_.com[0][1];
    casadi::SX pcom_z = ad_data_.com[0][2];
    
    casadi::SX vcom_x = ad_data_.vcom[0][0];
    casadi::SX vcom_y = ad_data_.vcom[0][1];
    
    casadi::SX h_com = pcom_z;  // CoM height
    
    // Capture point computation: p_cp = p_com_xy + v_com_xy * sqrt(h_com / g)
    double g = 9.81;
    casadi::SX omega_0 = casadi::SX::sqrt(h_com / g);
    
    std::vector<casadi::SX> p_com_xy = {pcom_x, pcom_y};
    std::vector<casadi::SX> v_com_xy = {vcom_x, vcom_y};
    casadi::SX p_com_2d = casadi::SX::vertcat(p_com_xy);
    casadi::SX v_com_2d = casadi::SX::vertcat(v_com_xy);
    
    casadi::SX p_cp = p_com_2d + v_com_2d * omega_0;
    
    // Symbolic parameters
    casadi::SX p_support = casadi::SX::sym("p_support", 2);
    casadi::SX w_balance = casadi::SX::sym("w_balance");
    
    // Balance cost: 0.5 * w * ||p_cp - p_support||²
    casadi::SX residual = p_cp - p_support;
    casadi::SX cost = 0.5 * w_balance * casadi::SX::dot(residual, residual);
    
    // Compute gradient and Hessian
    casadi::SX grad = casadi::SX::gradient(cost, x_sym_);
    casadi::SX hess = casadi::SX::jacobian(grad, x_sym_);
    
    // Create CasADi functions
    balance_grad_fn_ = casadi::Function("balance_grad",
                                        {x_sym_, p_support, w_balance},
                                        {grad});
    
    balance_hess_fn_ = casadi::Function("balance_hess",
                                        {x_sym_, p_support, w_balance},
                                        {hess});
}

Eigen::VectorXd EEDerivatives::BalanceGrad(const Eigen::VectorXd& x,
                                            const Eigen::Vector2d& p_support,
                                            double w_balance) {
    // Build functions if not yet built
    if (!balance_functions_built_) {
        buildBalanceFunctions();
        balance_functions_built_ = true;
    }
    
    // Early exit if weight is zero
    if (w_balance == 0.0) {
        return Eigen::VectorXd::Zero(nx_);
    }
    
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM p_support_dm = ::casadi::DM({p_support[0], p_support[1]});
    ::casadi::DM w_dm = ::casadi::DM(w_balance);
    
    // Evaluate function
    ::casadi::DM grad_dm = balance_grad_fn_(::casadi::DMVector{x_dm, p_support_dm, w_dm})[0];
    
    // Convert back to Eigen
    std::vector<double> grad_vec = grad_dm.get_elements();
    return Eigen::Map<Eigen::VectorXd>(grad_vec.data(), grad_vec.size());
}

Eigen::MatrixXd EEDerivatives::BalanceHess(const Eigen::VectorXd& x,
                                            const Eigen::Vector2d& p_support,
                                            double w_balance) {
    // Build functions if not yet built
    if (!balance_functions_built_) {
        buildBalanceFunctions();
        balance_functions_built_ = true;
    }
    
    // Early exit if weight is zero
    if (w_balance == 0.0) {
        return Eigen::MatrixXd::Zero(nx_, nx_);
    }
    
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM p_support_dm = ::casadi::DM({p_support[0], p_support[1]});
    ::casadi::DM w_dm = ::casadi::DM(w_balance);
    
    // Evaluate function
    ::casadi::DM hess_dm = balance_hess_fn_(::casadi::DMVector{x_dm, p_support_dm, w_dm})[0];
    
    // Convert back to Eigen
    std::vector<double> hess_vec = hess_dm.get_elements();
    Eigen::MatrixXd hess = Eigen::Map<Eigen::MatrixXd>(hess_vec.data(), nx_, nx_);
    
    // Ensure symmetry
    return 0.5 * (hess + hess.transpose());
}

double validateGrad(EEDerivatives& ee_deriv,
                    const Eigen::VectorXd& x,
                    const Eigen::Vector3d& target,
                    const std::string& frame_name,
                    double weight,
                    double eps) {
    
    // Get analytical gradient (now w.r.t. full state [q, v])
    Eigen::VectorXd grad_analytical = ee_deriv.EEposGrad(x, target, frame_name, weight);
    
    // Compute numerical gradient using simple forward differences
    Eigen::VectorXd grad_numerical(x.size());
    
    // Base cost (extract q from full state)
    Eigen::VectorXd q = x.head(ee_deriv.model_.nq);
    double cost_base = 0.0;
    {
        // Evaluate cost at base point
        pinocchio::forwardKinematics(ee_deriv.model_, ee_deriv.data_, q);
        pinocchio::updateFramePlacements(ee_deriv.model_, ee_deriv.data_);
        
        pinocchio::FrameIndex frame_id = ee_deriv.getFrameId(frame_name);
        Eigen::Vector3d ee_pos = ee_deriv.data_.oMf[frame_id].translation();
        Eigen::Vector3d error = ee_pos - target;
        cost_base = weight * error.squaredNorm();
    }
    
    // Finite differences w.r.t. full state
    for (int i = 0; i < x.size(); i++) {
        Eigen::VectorXd x_plus = x;
        x_plus(i) += eps;
        
        double cost_plus = cost_base;  // Default: no change for velocity derivatives
        
        // Only position derivatives affect end-effector cost
        if (i < ee_deriv.model_.nq) {
            Eigen::VectorXd q_plus = x_plus.head(ee_deriv.model_.nq);
            
            // Cost at perturbed point
            pinocchio::forwardKinematics(ee_deriv.model_, ee_deriv.data_, q_plus);
            pinocchio::updateFramePlacements(ee_deriv.model_, ee_deriv.data_);
            
            pinocchio::FrameIndex frame_id = ee_deriv.getFrameId(frame_name);
            Eigen::Vector3d ee_pos = ee_deriv.data_.oMf[frame_id].translation();
            Eigen::Vector3d error = ee_pos - target;
            cost_plus = weight * error.squaredNorm();
        }
        // Velocity derivatives should be zero for position-only cost
        
        grad_numerical(i) = (cost_plus - cost_base) / eps;
    }
    
    // Compute maximum error
    return (grad_analytical - grad_numerical).cwiseAbs().maxCoeff();
}

} // namespace derivatives