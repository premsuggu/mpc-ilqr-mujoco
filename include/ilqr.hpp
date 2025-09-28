#pragma once

#include "robot_utils.hpp"
#include <vector>

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
    iLQR(RobotUtils& robot, int N, double dt);

    // Configuration
    void setRegularization(double lambda);
    void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }
    void setTolerance(double tol) { tolerance_ = tol; }

    // solve (multi-iteration iLQR)
    bool solve(const Eigen::VectorXd& x0,
               const std::vector<Eigen::VectorXd>& x_ref,
               const std::vector<Eigen::VectorXd>& u_ref,
               const std::vector<Eigen::Vector3d>& com_ref,
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
                                const std::vector<Eigen::Vector3d>& com_ref,
                                const std::vector<Eigen::VectorXd>* prev_xbar = nullptr,
                                const std::vector<Eigen::VectorXd>* prev_ubar = nullptr);

private:
    RobotUtils& robot_;
    int N_;      // Horizon length
    double dt_;

    // Regularization and options
    double reg_lambda_;
    int max_iterations_;
    double tolerance_;

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
    std::vector<Eigen::Vector3d> com_ref_;  // CoM reference trajectory

    // Value function
    Eigen::VectorXd VxN_;     // Terminal gradient
    Eigen::MatrixXd VxxN_;    // Terminal Hessian

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
    
    // CoM cost derivatives (finite difference)
    void addCoMCostDerivatives(int t, Eigen::VectorXd& lx, Eigen::MatrixXd& lxx);

    // Utilities
    double computeTotalCost(const std::vector<Eigen::VectorXd>& x_traj,
                            const std::vector<Eigen::VectorXd>& u_traj,
                            const std::vector<Eigen::VectorXd>& x_ref,
                            const std::vector<Eigen::VectorXd>& u_ref);
};