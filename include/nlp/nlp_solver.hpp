#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <casadi/casadi.hpp>
#include <vector>
#include <string>

#include "nlp/nlp_config.hpp"
#include "nlp/mpc_utils.hpp"

namespace nlp {

// Forward declaration
class SymUtils;

/**
 * @brief NLP-based Trajectory Optimizer
 * 
 * Thin orchestration layer that:
 * - Sets up the NLP problem structure
 * - Packs/unpacks parameters
 * - Interfaces with IPOPT
 * - Delegates symbolic math to SymUtils
 */
class NLPSolver {
public:
    /**
     * @brief Constructor
     * @param config NLP configuration
     * @param model Pinocchio model
     */
    NLPSolver(const NLPConfig& config, const pinocchio::Model& model);
    
    ~NLPSolver();
    
    /**
     * @brief Solve trajectory optimization problem
     * @param x0 Initial state [q0; v0]
     * @param refs Reference trajectories
     * @param contacts Contact schedule
     * @param W_guess Warm start (optional)
     * @return Decision variables W = [a_0...a_{N-1}; f_0...f_{N-1}]
     */
    Eigen::VectorXd solve(const Eigen::VectorXd& x0,
                          const References& refs,
                          const ContactSchedule& contacts,
                          const Eigen::VectorXd& W_guess = Eigen::VectorXd());
    
    /**
     * @brief Extract first control from solution
     * @param W_sol Solution vector
     * @return Control vector (currently unused, returns zeros)
     */
    Eigen::VectorXd extractFirstControl(const Eigen::VectorXd& W_sol);
    
    /**
     * @brief Create warm start for next iteration
     * @param W_sol Current solution
     * @return Shifted warm start
     */
    Eigen::VectorXd warmStart(const Eigen::VectorXd& W_sol);
    
private:
    const NLPConfig& config_;
    const pinocchio::Model& model_;
    
    int nq_, nv_, nu_, n_ee_;
    
    SymUtils* sym_;  // Symbolic expressions (costs + dynamics + constraints)
    
    // Symbolic variables (decision variables + parameters)
    std::vector<casadi::SX> a_sym_;
    std::vector<casadi::SX> f_sym_;
    std::vector<casadi::SX> q_sym_;
    std::vector<casadi::SX> v_sym_;
    
    casadi::SX q0_param_;
    casadi::SX v0_param_;
    std::vector<casadi::SX> q_ref_param_;
    std::vector<casadi::SX> v_ref_param_;
    std::vector<casadi::SX> com_ref_param_;
    std::vector<std::vector<casadi::SX>> ee_pos_ref_param_;
    
    // IPOPT solver
    casadi::Function solver_;
    casadi::DM lbx_, ubx_, lbg_, ubg_;
    
    // NLP construction
    void createSymbolicVariables();
    void buildIntegrationChain();
    casadi::SX buildCost(const ContactSchedule& contacts);
    casadi::SX buildConstraints(const ContactSchedule& contacts);
    void setupBounds(const ContactSchedule& contacts);
    void setupSolver(const casadi::SX& cost, const casadi::SX& constraints);
};

} // namespace nlp
