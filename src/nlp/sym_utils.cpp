#include "nlp/sym_utils.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <iostream>

namespace nlp {

SymUtils::SymUtils(const pinocchio::Model& model, const std::vector<std::string>& ee_names)
    : model_(model), data_(model), ee_names_(ee_names)
{
    nq_ = model_.nq;
    nv_ = model_.nv;
    n_ee_ = ee_names.size();

    // Build CasADi model for symbolic differentiation
    ad_model_ = model_.cast<ADScalar>();
    ad_data_ = ADData(ad_model_);

    // Get frame IDs
    for (const auto& name : ee_names_) {
        if (!model_.existFrame(name)) {
            std::cerr << "[SymUtils] Warning: Frame " << name << " not found in model!" << std::endl;
            continue;
        }
        frame_ids_[name] = model_.getFrameId(name);
    }

    // Build all symbolic functions
    buildDynamicsFunctions();
    buildCoMFunction();
    buildEEPositionFunctions();

    std::cout << "[SymUtils] Initialized with " << n_ee_ << " end-effectors" << std::endl;
}

// ==================== DYNAMICS FUNCTIONS ====================

void SymUtils::buildDynamicsFunctions() {
    casadi::SX cs_q = casadi::SX::sym("q", nq_);
    casadi::SX cs_v = casadi::SX::sym("v", nv_);
    casadi::SX cs_dt = casadi::SX::sym("dt", 1);
    
    typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> ConfigVectorAD;
    typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> TangentVectorAD;
    
    ConfigVectorAD q_ad(nq_);
    TangentVectorAD v_ad(nv_);
    
    for (int i = 0; i < nq_; ++i) q_ad[i] = cs_q(i);
    for (int i = 0; i < nv_; ++i) v_ad[i] = cs_v(i);
    
    // 1. Integration function
    {
        ConfigVectorAD q_next_ad(nq_);
        TangentVectorAD v_dt_ad = v_ad * cs_dt(0);
        pinocchio::integrate(ad_model_, q_ad, v_dt_ad, q_next_ad);
        
        casadi::SX cs_q_next = casadi::SX::zeros(nq_, 1);
        for (int i = 0; i < nq_; ++i) cs_q_next(i) = q_next_ad[i];
        
        integrate_fn_ = casadi::Function("integrate", {cs_q, cs_v, cs_dt}, {cs_q_next});
    }
    
    // 2. Mass matrix function
    {
        pinocchio::crba(ad_model_, ad_data_, q_ad);
        casadi::SX cs_M = casadi::SX::zeros(nv_, nv_);
        for (int i = 0; i < nv_; ++i) {
            for (int j = 0; j < nv_; ++j) {
                cs_M(i, j) = ad_data_.M(i, j);
            }
        }
        mass_matrix_fn_ = casadi::Function("mass_matrix", {cs_q}, {cs_M});
    }
    
    // 3. Coriolis function
    {
        pinocchio::rnea(ad_model_, ad_data_, q_ad, v_ad, TangentVectorAD::Zero(nv_));
        casadi::SX cs_C_v = casadi::SX::zeros(nv_, 1);
        for (int i = 0; i < nv_; ++i) cs_C_v(i) = ad_data_.tau[i];
        
        coriolis_fn_ = casadi::Function("coriolis", {cs_q, cs_v}, {cs_C_v});
    }
    
    // 4. Gravity function
    {
        pinocchio::rnea(ad_model_, ad_data_, q_ad, TangentVectorAD::Zero(nv_), TangentVectorAD::Zero(nv_));
        casadi::SX cs_g = casadi::SX::zeros(nv_, 1);
        for (int i = 0; i < nv_; ++i) cs_g(i) = ad_data_.tau[i];
        
        gravity_fn_ = casadi::Function("gravity", {cs_q}, {cs_g});
    }
    
    // 5. Jacobian functions - COMPUTE ONCE, then extract for each EE
    pinocchio::computeJointJacobians(ad_model_, ad_data_, q_ad);
    pinocchio::updateFramePlacements(ad_model_, ad_data_);
    
    for (const auto& ee_name : ee_names_) {
        pinocchio::FrameIndex frame_id = model_.getFrameId(ee_name);
        
        Eigen::Matrix<ADScalar, 6, Eigen::Dynamic> J_frame(6, nv_);
        J_frame.setZero();
        pinocchio::getFrameJacobian(ad_model_, ad_data_, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J_frame);
        
        casadi::SX cs_J = casadi::SX::zeros(6, nv_);
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < nv_; ++j) {
                cs_J(i, j) = J_frame(i, j);
            }
        }
        jacobian_fns_[ee_name] = casadi::Function("jacobian_" + ee_name, {cs_q}, {cs_J});
    }
    
    std::cout << "[SymUtils] Built dynamics functions (M, C, g, integrate, " 
              << jacobian_fns_.size() << " Jacobians)" << std::endl;
}

// ==================== COST FUNCTIONS ====================

void SymUtils::buildCoMFunction() {
    casadi::SX q_sym = casadi::SX::sym("q", nq_);

    Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> q_ad(nq_);
    for (int i = 0; i < nq_; ++i) {
        q_ad(i) = q_sym(i);
    }

    pinocchio::centerOfMass(ad_model_, ad_data_, q_ad, false);
    
    casadi::SX com_pos(3, 1);
    for (int i = 0; i < 3; ++i) {
        com_pos(i) = ad_data_.com[0](i);
    }

    com_function_ = casadi::Function("com_pos", {q_sym}, {com_pos});
}

void SymUtils::buildEEPositionFunctions() {
    casadi::SX q_sym = casadi::SX::sym("q", nq_);

    Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> q_ad(nq_);
    for (int i = 0; i < nq_; ++i) {
        q_ad(i) = q_sym(i);
    }

    pinocchio::forwardKinematics(ad_model_, ad_data_, q_ad);
    pinocchio::updateFramePlacements(ad_model_, ad_data_);

    for (const auto& name : ee_names_) {
        if (frame_ids_.find(name) == frame_ids_.end()) continue;

        int frame_id = frame_ids_[name];
        
        casadi::SX ee_pos(3, 1);
        for (int i = 0; i < 3; ++i) {
            ee_pos(i) = ad_data_.oMf[frame_id].translation()(i);
        }

        ee_position_functions_[name] = casadi::Function(
            "ee_pos_" + name, {q_sym}, {ee_pos});
    }

    std::cout << "[SymUtils] Built cost functions (CoM, " 
              << ee_position_functions_.size() << " EE positions)" << std::endl;
}

casadi::SX SymUtils::computeStageCost(
    const casadi::SX& q, const casadi::SX& v,
    const casadi::SX& a, const casadi::SX& f,
    const casadi::SX& q_ref, const casadi::SX& v_ref,
    const casadi::SX& com_ref,
    const std::vector<casadi::SX>& ee_pos_ref,
    const std::vector<bool>& contact_schedule,
    const CostWeights& weights)
{
    casadi::SX cost = 0.0;

    // 1. Position tracking
    casadi::SX q_err = q - q_ref;
    cost += 0.5 * weights.w_q * casadi::SX::dot(q_err, q_err);

    // 2. Velocity tracking
    casadi::SX v_err = v - v_ref;
    cost += 0.5 * weights.w_v * casadi::SX::dot(v_err, v_err);

    // 3. Center-of-mass tracking
    if (weights.w_com > 1e-6) {
        casadi::SX com_pos = com_function_(casadi::SXVector{q})[0];
        casadi::SX com_err = com_pos - com_ref;
        cost += 0.5 * weights.w_com * casadi::SX::dot(com_err, com_err);
    }

    // 4. Contact-aware end-effector cost
    if (weights.w_ee_pos > 1e-6) {
        cost += computeContactAwareEECost(q, ee_pos_ref, contact_schedule, weights.w_ee_pos);
    }

    // 5. Upright torso penalty
    if (weights.w_upright > 1e-6) {
        cost += computeUprightCost(q, weights.w_upright);
    }

    // 6. Acceleration regularization
    cost += 0.5 * weights.w_a * casadi::SX::dot(a, a);

    // 7. Force regularization
    cost += 0.5 * weights.w_f * casadi::SX::dot(f, f);

    return cost;
}

casadi::SX SymUtils::computeTerminalCost(
    const casadi::SX& q, const casadi::SX& v,
    const casadi::SX& q_ref, const casadi::SX& v_ref,
    const CostWeights& weights)
{
    casadi::SX cost = 0.0;

    // Terminal position tracking (higher weight)
    casadi::SX q_err = q - q_ref;
    cost += 0.5 * weights.terminal_multiplier * weights.w_q * casadi::SX::dot(q_err, q_err);

    // Terminal velocity tracking
    casadi::SX v_err = v - v_ref;
    cost += 0.5 * weights.terminal_multiplier * weights.w_v * casadi::SX::dot(v_err, v_err);

    return cost;
}

casadi::SX SymUtils::computeContactAwareEECost(
    const casadi::SX& q,
    const std::vector<casadi::SX>& ee_pos_ref,
    const std::vector<bool>& contact_schedule,
    double w_ee_pos)
{
    casadi::SX cost = 0.0;

    for (size_t ee_idx = 0; ee_idx < ee_names_.size(); ++ee_idx) {
        const std::string& ee_name = ee_names_[ee_idx];
        
        if (ee_position_functions_.find(ee_name) == ee_position_functions_.end()) {
            continue;
        }

        casadi::SX ee_pos = ee_position_functions_[ee_name](casadi::SXVector{q})[0];
        casadi::SX ee_pos_ref_val = ee_pos_ref[ee_idx];
        casadi::SX ee_err = ee_pos - ee_pos_ref_val;

        // Contact-aware weighting: stance=OFF, swing=ON
        double k = contact_schedule[ee_idx] ? 1.0 : 0.0;
        double weight = (1.0 - k) * w_ee_pos;

        cost += 0.5 * weight * casadi::SX::dot(ee_err, ee_err);
    }

    return cost;
}

casadi::SX SymUtils::computeUprightCost(const casadi::SX& q, double w_upright) {
    // Quaternion at q[3:7] (free-flyer: x,y,z, qx,qy,qz,qw)
    casadi::SX qx = q(3);
    casadi::SX qy = q(4);
    casadi::SX qz = q(5);
    casadi::SX qw = q(6);

    // Rotate world Z-axis by quaternion to get torso Z-axis
    casadi::SX z_torso_x = 2.0 * (qx * qz + qw * qy);
    casadi::SX z_torso_y = 2.0 * (qy * qz - qw * qx);
    casadi::SX z_torso_z = 1.0 - 2.0 * (qx * qx + qy * qy);

    // Desired: z_torso = [0, 0, 1]
    casadi::SX err_x = z_torso_x - 0.0;
    casadi::SX err_y = z_torso_y - 0.0;
    casadi::SX err_z = z_torso_z - 1.0;

    return 0.5 * w_upright * (err_x * err_x + err_y * err_y + err_z * err_z);
}

// ==================== CONSTRAINT HELPERS ====================

casadi::SX SymUtils::computeTorques(
    const casadi::SX& q,
    const casadi::SX& v,
    const casadi::SX& a,
    const casadi::SX& f,
    const std::vector<std::string>& ee_names)
{
    // τ = M*a + C + g - J^T*f
    casadi::SX M = mass_matrix_fn_(casadi::SXVector{q})[0];
    casadi::SX C_v = coriolis_fn_(casadi::SXVector{q, v})[0];
    casadi::SX g = gravity_fn_(casadi::SXVector{q})[0];
    
    casadi::SX tau = casadi::SX::mtimes(M, a) + C_v + g;
    
    // Subtract contact forces
    for (size_t ee_idx = 0; ee_idx < ee_names.size(); ++ee_idx) {
        const std::string& ee_name = ee_names[ee_idx];
        casadi::SX J = jacobian_fns_[ee_name](casadi::SXVector{q})[0];
        int start_idx = static_cast<int>(ee_idx * 6);
        int end_idx = static_cast<int>((ee_idx + 1) * 6);
        casadi::SX f_ee = f(casadi::Slice(start_idx, end_idx));
        tau -= casadi::SX::mtimes(J.T(), f_ee);
    }
    
    return tau;
}

} // namespace nlp
