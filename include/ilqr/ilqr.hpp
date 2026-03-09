#pragma once

#include "ilqr/robot_utils.hpp"
#include "ilqr/derivatives.hpp"
#include "ilqr/norm.hpp"
#include <vector>
#include <map>
#include <string>

/**
 * @brief Iterative LQR solver for MPC
 * 
 * Implements single-iteration iLQR with MuJoCo dynamics:
 * - Forward rollout with nominal controls
 * - Finite-difference linearization 
 * - Backward pass (Riccati recursion)
 * - Forward line search
 * - Output: nominal trajectories + TV-LQR gains
 */
class iLQR {
public:
    iLQR(RobotUtils& robot, int N, double dt, const std::string& urdf_path);

    // Configuration
    void setRegularization(double lambda);
    void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }
    void setTolerance(double tol) { tolerance_ = tol; }
    
    /**
     * @brief Set gravity magnitude for balance cost computation
     * @param g Gravity magnitude (m/s^2)
     */
    void setGravity(double g) { derivatives_.setGravity(g); }
    void setBalanceTimeConstant(double omega) { derivatives_.setBalanceTimeConstant(omega); }
    /**
     * @brief Set norm parameters for all cost terms
     * @param norm_params Map of cost term names to their norm configurations
     */
    void setNormParams(const std::map<std::string, ilqr::NormParams>& norm_params);
    
    /**
     * @brief Configure iLQR solver settings
     */
    void configureSolver(double reg_min, double reg_max, double reg_increase_factor,
                        double reg_decrease_factor, double trust_region_good,
                        double trust_region_poor, int num_line_search_steps,
                        double min_linesearch_step, double line_search_tolerance,
                        double quu_regularization, double convergence_threshold);

    // solve (multi-iteration iLQR)
    bool solve(const Eigen::VectorXd& x0,
               const std::vector<Eigen::VectorXd>& x_ref,
               const std::vector<Eigen::VectorXd>& u_ref,
               const std::vector<Eigen::Vector3d>& height_ref,
               double& cost_out);

    // Access results
    const std::vector<Eigen::VectorXd>& xbar() const { return xbar_; }
    const std::vector<Eigen::VectorXd>& ubar() const { return ubar_; }
    const std::vector<Eigen::MatrixXd>& gainsK() const { return K_; }
    const std::vector<Eigen::VectorXd>& gainsKff() const { return kff_; }

    // Reference-aware initialization for better cold start
    void initializeWithReference(const Eigen::VectorXd& x0,
                                const std::vector<Eigen::VectorXd>& x_ref,
                                const std::vector<Eigen::VectorXd>& u_ref,
                                const std::vector<Eigen::Vector3d>& height_ref,
                                const std::vector<Eigen::VectorXd>* prev_xbar = nullptr,
                                const std::vector<Eigen::VectorXd>* prev_ubar = nullptr);

private:
    RobotUtils& robot_;
    derivatives::symDerivatives derivatives_;  // Symbolic derivatives system
    int N_;      // Horizon length
    double dt_;

    // Regularization and options
    double reg_lambda_;
    int max_iterations_;
    double tolerance_;
    
    // Solver settings (configurable)
    double reg_min_;
    double reg_max_;
    double reg_increase_factor_;
    double reg_decrease_factor_;
    double trust_region_good_;
    double trust_region_poor_;
    int num_line_search_steps_;       // Number of line search candidates
    double min_linesearch_step_;      // Minimum line search step size
    double line_search_tolerance_;
    double quu_regularization_;
    double convergence_threshold_;
    
    // Cost norm configurations
    std::map<std::string, ilqr::NormParams> norm_params_;

    // Nominal trajectories
    std::vector<Eigen::VectorXd> xbar_, ubar_;

    // TV-LQR gains
    std::vector<Eigen::MatrixXd> K_;     // Feedback gains
    std::vector<Eigen::VectorXd> kff_;   // Feedforward terms

    // Linearizations (A_t, B_t matrices)
    std::vector<Eigen::MatrixXd> A_, B_;

    // Cost quadratics
    std::vector<Eigen::VectorXd> lx_, lu_;          // Gradients
    std::vector<Eigen::MatrixXd> lxx_, luu_, lxu_;  // Hessians

    // Reference storage
    std::vector<Eigen::VectorXd> x_ref_, u_ref_;
    std::vector<Eigen::Vector3d> height_ref_;  // Height (torso z) reference trajectory

    // Value function
    Eigen::VectorXd VxN_;     // Terminal gradient
    Eigen::MatrixXd VxxN_;    // Terminal Hessian
    
    // Expected cost reduction: dV_[0] = linear, dV_[1] = quadratic
    Eigen::Vector2d dV_;

    // iLQR stages
    void forwardRolloutNominal();
    void computeLinearization();
    void computeCostQuadratics(const std::vector<Eigen::VectorXd>& x_ref,
                               const std::vector<Eigen::VectorXd>& u_ref);
    void backwardPass();
    bool forwardPassLineSearch(const Eigen::VectorXd& x0,
                               const std::vector<Eigen::VectorXd>& x_ref,
                               const std::vector<Eigen::VectorXd>& u_ref,
                               double& new_cost);
    
    // Symbolic cost derivatives
    void addHeightCostDerivatives(int t, double goal_z);
    void addVelocityCostDerivatives(int t);  // SEPARATE velocity tracking
    void addJointVelCostDerivatives(int t);  // Joint velocity damping
    void addUprightCostDerivatives(int t);
    void addBalanceCostDerivatives(int t);
    void addPelvisFeetCostDerivatives(int t);
    void addWalkCostDerivatives(int t);

    // Utilities
    double computeTotalCost(const std::vector<Eigen::VectorXd>& x_traj,
                            const std::vector<Eigen::VectorXd>& u_traj,
                            const std::vector<Eigen::VectorXd>& x_ref,
                            const std::vector<Eigen::VectorXd>& u_ref);
};