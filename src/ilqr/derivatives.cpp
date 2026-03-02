#include "ilqr/derivatives.hpp"
#include "ilqr/cost.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/mjcf.hpp>
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

symDerivatives::symDerivatives(const std::string& model_path, bool floating_base) {
    // Detect file type by extension
    bool is_mjcf = (model_path.size() >= 4 && 
                    model_path.substr(model_path.size() - 4) == ".xml");
    
    if (is_mjcf) {
        // Load MJCF (MuJoCo XML) - freejoint in XML automatically creates floating base
        pinocchio::mjcf::buildModel(model_path, model_);
        std::cout << "Loaded robot from MJCF: " << model_path << std::endl;
    } else {
        // Load URDF
        if (floating_base) {
            pinocchio::urdf::buildModel(model_path, pinocchio::JointModelFreeFlyer(), model_);
        } else {
            pinocchio::urdf::buildModel(model_path, model_);
        }
        std::cout << "Loaded robot from URDF: " << model_path << std::endl;
    }
    data_ = pinocchio::Data(model_);
    
    std::cout << "Robot: nq=" << model_.nq << ", nv=" << model_.nv << std::endl;
    
    // Build symbolic computation framework once
    buildSymbolicFunctions();
}

void symDerivatives::buildSymbolicFunctions() {
    // Create symbolic full state vector [q, v]
    int nx = model_.nq + model_.nv;  // Full state size
    nx_ = nx;  // Cache for later use
    x_sym_ = ::casadi::SX::sym("x", nx);
    
    // Create CasADi-compatible model for symbolic computations
    ad_model_ = model_.template cast<ADScalar>();
    ad_data_ = pinocchio::DataTpl<ADScalar>(ad_model_);
    
    // Initialize gravity with default value (will be overridden from config)
    gravity_ = 9.81;
    
    // Initialize CoM functions flag
    height_functions_built_ = false;
    vel_functions_built_ = false;
    upright_functions_built_ = false;
    balance_functions_built_ = false;
    
    std::cout << "Built symbolic computation framework for state size " << nx 
              << " (nq=" << model_.nq << ", nv=" << model_.nv << ")" << std::endl;
}

void symDerivatives::buildHeightFunctions() {
    // Create symbolic input parameters (scalar target height)
    ::casadi::SX target_z_sym = ::casadi::SX::sym("target_z");
    ::casadi::SX weight_sym = ::casadi::SX::sym("weight");
    
    // Use symbolic cost helper
    ::casadi::SX height_cost = symHeight(target_z_sym, weight_sym);
    
    // Build gradient and Hessian functions
    ::casadi::SX height_grad = ::casadi::SX::gradient(height_cost, x_sym_);
    height_grad_fn_ = ::casadi::Function(
        "height_grad",
        {x_sym_, target_z_sym, weight_sym}, {height_grad}
    );
    
    ::casadi::SX height_hess = ::casadi::SX::jacobian(height_grad, x_sym_);
    height_hess_fn_ = ::casadi::Function(
        "height_hess", 
        {x_sym_, target_z_sym, weight_sym}, {height_hess}
    );
    
    height_functions_built_ = true;
    std::cout << "Built cached height functions" << std::endl;
}

// Build velocity cost functions (world-frame base xy velocity, zero target)
void symDerivatives::buildVelocityFunctions() {
    // No target parameter — residual is raw v_base_xy (zero target, DeepMind "Velocity")
    ::casadi::SX weight_sym = ::casadi::SX::sym("weight");
    
    // Use symbolic cost helper
    ::casadi::SX vel_cost = symVelocity(weight_sym);
    
    // Build gradient and Hessian functions
    ::casadi::SX vel_grad = ::casadi::SX::gradient(vel_cost, x_sym_);
    vel_grad_fn_ = ::casadi::Function(
        "vel_grad",
        {x_sym_, weight_sym}, {vel_grad}
    );
    
    ::casadi::SX vel_hess = ::casadi::SX::jacobian(vel_grad, x_sym_);
    vel_hess_fn_ = ::casadi::Function(
        "vel_hess", 
        {x_sym_, weight_sym}, {vel_hess}
    );
    
    vel_functions_built_ = true;
    std::cout << "Built cached velocity (2D xy base) functions" << std::endl;
}

// For keeping the robot upright
void symDerivatives::buildUprightFunctions() {
    // Weight parameter
    casadi::SX w_upright = casadi::SX::sym("w_upright");
    
    // Use symbolic cost helper
    casadi::SX cost = symUpright(w_upright);
    
    // Compute gradient and Hessian
    casadi::SX grad = casadi::SX::gradient(cost, x_sym_);
    casadi::SX hess = casadi::SX::hessian(cost, x_sym_);
    
    // Create CasADi functions
    upright_grad_fn_ = casadi::Function("upright_grad", 
                                        {x_sym_, w_upright}, 
                                        {grad});
    
    upright_hess_fn_ = casadi::Function("upright_hess", 
                                        {x_sym_, w_upright}, 
                                        {hess});
}

Eigen::VectorXd symDerivatives::HeightGrad(const Eigen::VectorXd& x,
                                       double goal_z,
                                       double weight) {
    
    // Ensure height functions are built
    if (!height_functions_built_) {
        buildHeightFunctions();
    }
    
    // Convert MuJoCo state to Pinocchio state (fix quaternion ordering)
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM target_dm = ::casadi::DM(goal_z);
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM grad_dm = height_grad_fn_(::casadi::DMVector{x_dm, target_dm, weight_dm})[0];
    
    // Convert back to Eigen (full state size)
    int nx = model_.nq + model_.nv;
    Eigen::VectorXd gradient(nx);
    for (int i = 0; i < nx; i++) {
        gradient(i) = double(grad_dm(i));
    }
    
    return gradient;
}

Eigen::MatrixXd symDerivatives::HeightHess(const Eigen::VectorXd& x,
                                       double goal_z,
                                       double weight) {
    
    // Ensure height functions are built
    if (!height_functions_built_) {
        buildHeightFunctions();
    }
    
    // Convert MuJoCo state to Pinocchio state (fix quaternion ordering)
    Eigen::VectorXd x_pinocchio = convertMuJoCoToPinocchio(x, model_.nq);
    
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x_pinocchio.data(), x_pinocchio.data() + x_pinocchio.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM target_dm = ::casadi::DM(goal_z);
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM hess_dm = height_hess_fn_(::casadi::DMVector{x_dm, target_dm, weight_dm})[0];
    
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

Eigen::VectorXd symDerivatives::VelocityGrad(const Eigen::VectorXd& x,
                                             double weight) {
    
    // Ensure velocity functions are built
    if (!vel_functions_built_) {
        buildVelocityFunctions();
    }
    
    // No Pinocchio quaternion conversion needed — only uses velocity states directly
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x.data(), x.data() + x.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM grad_dm = vel_grad_fn_(::casadi::DMVector{x_dm, weight_dm})[0];
    
    // Convert back to Eigen (full state size)
    int nx = model_.nq + model_.nv;
    Eigen::VectorXd gradient(nx);
    for (int i = 0; i < nx; i++) {
        gradient(i) = double(grad_dm(i));
    }
    
    return gradient;
}

// Velocity Hessian (world-frame base xy velocity, zero target)
Eigen::MatrixXd symDerivatives::VelocityHess(const Eigen::VectorXd& x,
                                             double weight) {
    
    // Ensure velocity functions are built
    if (!vel_functions_built_) {
        buildVelocityFunctions();
    }
    
    // No Pinocchio quaternion conversion needed — only uses velocity states directly
    // Convert inputs to CasADi format
    std::vector<double> x_vec(x.data(), x.data() + x.size());
    ::casadi::DM x_dm = ::casadi::DM(x_vec);
    ::casadi::DM weight_dm = ::casadi::DM(weight);
    
    // Evaluate cached function (fast!)
    ::casadi::DM hess_dm = vel_hess_fn_(::casadi::DMVector{x_dm, weight_dm})[0];
    
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
Eigen::VectorXd symDerivatives::UprightGrad(const Eigen::VectorXd& x, double w_upright) {
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

Eigen::MatrixXd symDerivatives::UprightHess(const Eigen::VectorXd& x, double w_upright) {
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


::casadi::SX symDerivatives::symHeight(const ::casadi::SX& target_z,
                                       const ::casadi::SX& weight) {
    // Height cost: residual = torso_z - goal_z  (DeepMind "Height" term)
    // x_sym_(2) is the base/torso z position in Pinocchio convention
    // (indices 0,1,2 are px,py,pz regardless of quaternion convention)
    ::casadi::SX residual = x_sym_(2) - target_z;
    
    // Get norm params from configuration (default to quadratic if not found)
    ilqr::NormParams norm = norm_params_.count("height") > 0 
        ? norm_params_.at("height")
        : ilqr::NormParams{ilqr::NormType::Quadratic, 1.0, 1.0};
    return ilqr::HeightCost(residual, weight, norm);
}

::casadi::SX symDerivatives::symVelocity(const ::casadi::SX& weight) {
    // World-frame base linear velocity is directly in the MuJoCo state at x[nq+0], x[nq+1].
    // This mirrors DeepMind's use of torso_velocity (world frame xy).
    ::casadi::SX vx = x_sym_(model_.nq + 0);
    ::casadi::SX vy = x_sym_(model_.nq + 1);
    ::casadi::SX residual = ::casadi::SX::vertcat({vx, vy});
    
    // Get norm params from configuration
    ilqr::NormParams norm = norm_params_.count("velocity") > 0 
        ? norm_params_.at("velocity")
        : ilqr::NormParams{ilqr::NormType::Quadratic, 1.0, 1.0};
    return ilqr::VelocityCost(residual, weight, norm);
}

::casadi::SX symDerivatives::symUpright(const ::casadi::SX& weight) {
    // Extract quaternion from x_sym_: [pos(3), quat(4), joints...]
    // Quaternion indices: [3, 4, 5, 6] for [qx, qy, qz, qw] in Pinocchio format
    casadi::SX qx = x_sym_(3);
    casadi::SX qy = x_sym_(4);
    casadi::SX qz = x_sym_(5);
    casadi::SX qw = x_sym_(6);
    
    // Compute torso z-axis in world frame (3rd column of rotation matrix)
    casadi::SX z_torso_x = 2 * (qx * qz + qw * qy);
    casadi::SX z_torso_y = 2 * (qy * qz - qw * qx);
    casadi::SX z_torso_z = 1 - 2 * (qx * qx + qy * qy);
    
    // Build torso z-axis vector
    ::casadi::SX torso_z = ::casadi::SX::vertcat({z_torso_x, z_torso_y, z_torso_z});
    ::casadi::SX up = ::casadi::SX::vertcat({0.0, 0.0, 1.0});
    ::casadi::SX residual = torso_z - up;
    
    // Get norm params from configuration
    ilqr::NormParams norm = norm_params_.count("upright") > 0 
        ? norm_params_.at("upright")
        : ilqr::NormParams{ilqr::NormType::Quadratic, 1.0, 1.0};
    return ilqr::uprightCost(residual, weight, norm);
}

::casadi::SX symDerivatives::symBalance(const ::casadi::SX& p_support,
                                        const ::casadi::SX& weight) {
    // Extract q and v from x_sym_
    typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> ConfigVector;
    ConfigVector q_ad(model_.nq);
    
    for (int i = 0; i < model_.nq; i++) {
        q_ad[i] = x_sym_(i);
    }
    
    // World-frame base linear velocity is at x[nq+0], x[nq+1] in both MuJoCo and Pinocchio
    // states (convertMuJoCoToPinocchio only swaps quaternion order, not velocity frame).
    // No rotation matrix or Jacobian needed — numerically stable.
    casadi::SX vcom_x = x_sym_(model_.nq + 0);
    casadi::SX vcom_y = x_sym_(model_.nq + 1);
    
    // CoM position via FK (stable, no Jacobian needed)
    pinocchio::forwardKinematics(ad_model_, ad_data_, q_ad);
    pinocchio::centerOfMass(ad_model_, ad_data_, q_ad, false);
    
    casadi::SX pcom_x = ad_data_.com[0][0];
    casadi::SX pcom_y = ad_data_.com[0][1];
    
    // Fixed CP time constant: 0.3 s (DeepMind walk.cc, not sqrt(h/g))
    casadi::SX omega_0 = 0.3;
    
    std::vector<casadi::SX> p_com_xy = {pcom_x, pcom_y};
    std::vector<casadi::SX> v_com_xy = {vcom_x, vcom_y};
    casadi::SX p_com_2d = casadi::SX::vertcat(p_com_xy);
    casadi::SX v_com_2d = casadi::SX::vertcat(v_com_xy);
    
    casadi::SX p_cp = p_com_2d + v_com_2d * omega_0;
    casadi::SX residual = p_cp - p_support;
    
    // Get norm params from configuration
    ilqr::NormParams norm = norm_params_.count("balance") > 0 
        ? norm_params_.at("balance")
        : ilqr::NormParams{ilqr::NormType::Quadratic, 1.0, 1.0};
    return ilqr::balanceCost(residual, weight, norm);
}

pinocchio::FrameIndex symDerivatives::getFrameId(const std::string& frame_name) {
    if (!model_.existFrame(frame_name)) {
        throw std::runtime_error("Frame '" + frame_name + "' not found");
    }
    return model_.getFrameId(frame_name);
}

void symDerivatives::setNormParams(const std::map<std::string, ilqr::NormParams>& norm_params) {
    norm_params_ = norm_params;
}

void symDerivatives::buildBalanceFunctions() {
    // Symbolic parameters
    casadi::SX p_support = casadi::SX::sym("p_support", 2);
    casadi::SX w_balance = casadi::SX::sym("w_balance");
    
    // Use symbolic cost helper
    casadi::SX cost = symBalance(p_support, w_balance);
    
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

Eigen::VectorXd symDerivatives::BalanceGrad(const Eigen::VectorXd& x,
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

Eigen::MatrixXd symDerivatives::BalanceHess(const Eigen::VectorXd& x,
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
} // namespace derivatives