#pragma once

#include <casadi/casadi.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>

#include "nlp/nlp_config.hpp"

namespace nlp {

/**
 * @brief Symbolic Utilities for NLP MPC
 * 
 * Centralizes ALL symbolic expression construction including:
 * - Cost functions (tracking, regularization, penalties)
 * - Dynamics functions (M, C, g, integration)
 * - Constraint helpers (torque computation, Jacobians)
 * 
 * This class handles the symbolic math, while the solver handles optimization.
 */
class SymUtils {
public:
    /**
     * @brief Constructor
     * @param model Pinocchio model
     * @param ee_names End-effector frame names
     */
    SymUtils(const pinocchio::Model& model, const std::vector<std::string>& ee_names);
    
    ~SymUtils() = default;

    // ==================== COST FUNCTIONS ====================
    
    /**
     * @brief Compute stage cost for one timestep
     */
    casadi::SX computeStageCost(
        const casadi::SX& q, const casadi::SX& v,
        const casadi::SX& a, const casadi::SX& f,
        const casadi::SX& q_ref, const casadi::SX& v_ref,
        const casadi::SX& com_ref,
        const std::vector<casadi::SX>& ee_pos_ref,
        const std::vector<bool>& contact_schedule,
        const CostWeights& weights);

    /**
     * @brief Compute terminal cost
     */
    casadi::SX computeTerminalCost(
        const casadi::SX& q, const casadi::SX& v,
        const casadi::SX& q_ref, const casadi::SX& v_ref,
        const CostWeights& weights);

    // ==================== DYNAMICS FUNCTIONS ====================
    
    /**
     * @brief Get integration function: q_next = integrate(q, v, dt)
     */
    const casadi::Function& getIntegrateFunction() const { return integrate_fn_; }
    
    /**
     * @brief Get mass matrix function: M(q)
     */
    const casadi::Function& getMassMatrixFunction() const { return mass_matrix_fn_; }
    
    /**
     * @brief Get Coriolis function: C(q, v)
     */
    const casadi::Function& getCoriolisFunction() const { return coriolis_fn_; }
    
    /**
     * @brief Get gravity function: g(q)
     */
    const casadi::Function& getGravityFunction() const { return gravity_fn_; }
    
    /**
     * @brief Get Jacobian function for end-effector
     */
    const casadi::Function& getJacobianFunction(const std::string& ee_name) const {
        return jacobian_fns_.at(ee_name);
    }

    // ==================== CONSTRAINT HELPERS ====================
    
    /**
     * @brief Compute generalized torques: τ = M*a + C + g - J^T*f
     * 
     * @param q Configuration (symbolic)
     * @param v Velocity (symbolic)
     * @param a Acceleration (symbolic)
     * @param f Contact forces (symbolic, stacked [f1; f2; ...])
     * @param ee_names End-effector names
     * @return Generalized torques (size nv)
     */
    casadi::SX computeTorques(
        const casadi::SX& q,
        const casadi::SX& v,
        const casadi::SX& a,
        const casadi::SX& f,
        const std::vector<std::string>& ee_names);

    // ==================== GETTERS ====================
    
    int nq() const { return nq_; }
    int nv() const { return nv_; }
    int n_ee() const { return n_ee_; }

private:
    // Robot model (concrete Pinocchio)
    pinocchio::Model model_;
    pinocchio::Data data_;
    int nq_, nv_, n_ee_;
    std::vector<std::string> ee_names_;
    std::unordered_map<std::string, int> frame_ids_;

    // CasADi model (for symbolic differentiation)
    typedef casadi::SX ADScalar;
    typedef pinocchio::ModelTpl<ADScalar> ADModel;
    typedef pinocchio::DataTpl<ADScalar> ADData;
    
    ADModel ad_model_;
    ADData ad_data_;

    // Pre-built CasADi functions for dynamics
    casadi::Function integrate_fn_;
    casadi::Function mass_matrix_fn_;
    casadi::Function coriolis_fn_;
    casadi::Function gravity_fn_;
    std::unordered_map<std::string, casadi::Function> jacobian_fns_;

    // Pre-built CasADi functions for costs
    casadi::Function com_function_;
    std::unordered_map<std::string, casadi::Function> ee_position_functions_;

    // ==================== INITIALIZATION ====================
    
    /**
     * @brief Build all symbolic dynamics functions
     */
    void buildDynamicsFunctions();
    
    /**
     * @brief Build symbolic COM function
     */
    void buildCoMFunction();

    /**
     * @brief Build symbolic EE position functions
     */
    void buildEEPositionFunctions();

    // ==================== COST COMPONENTS ====================
    
    /**
     * @brief Compute contact-aware EE cost
     * Cost = (1 - k) * w_ee_pos * ||p_ee - p_ref||^2
     * where k = 1 for stance (cost OFF), k = 0 for swing (cost ON)
     */
    casadi::SX computeContactAwareEECost(
        const casadi::SX& q,
        const std::vector<casadi::SX>& ee_pos_ref,
        const std::vector<bool>& contact_schedule,
        double w_ee_pos);

    /**
     * @brief Compute upright torso cost using quaternion
     */
    casadi::SX computeUprightCost(const casadi::SX& q, double w_upright);
};

} // namespace nlp
