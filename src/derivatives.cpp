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
    x_sym_ = ::casadi::SX::sym("x", nx);
    
    // Create CasADi-compatible model for symbolic computations
    ad_model_ = model_.template cast<ADScalar>();
    ad_data_ = pinocchio::DataTpl<ADScalar>(ad_model_);
    
    // Initialize CoM functions flag
    com_functions_built_ = false;
    
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

pinocchio::FrameIndex EEDerivatives::getFrameId(const std::string& frame_name) {
    if (!model_.existFrame(frame_name)) {
        throw std::runtime_error("Frame '" + frame_name + "' not found");
    }
    return model_.getFrameId(frame_name);
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