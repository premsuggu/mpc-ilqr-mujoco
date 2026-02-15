#include "ilqr/cost.hpp"

namespace ilqr {

// ============================================================================
// Core Cost Functions - Take residuals only
// ============================================================================

// State cost: quadratic only (LQR design)
double StateCost(const Eigen::VectorXd& x_err, const Eigen::MatrixXd& Q) {
    return 0.5 * x_err.transpose() * Q * x_err;
}

// Control cost: quadratic only (LQR design)
double ControlCost(const Eigen::VectorXd& u_err, const Eigen::MatrixXd& R) {
    return 0.5 * u_err.transpose() * R * u_err;
}

// CoM position cost
double CoMPosCost(const Eigen::Vector3d& residual, double weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

// CoM velocity cost
double CoMVelCost(const Eigen::Vector3d& residual, double weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

// End-effector position cost
double EEPosCost(const Eigen::Vector3d& residual, double weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

// End-effector velocity cost
double EEVelCost(const Eigen::Vector3d& residual, double weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

// Upright cost: orientation tracking (0.5 factor for original behavior)
double uprightCost(const Eigen::Vector3d& residual, double weight, const NormParams& norm) {
    return 0.5 * weight * applyNorm(residual, norm);
}

// Balance cost: capture point stability (0.5 factor for original behavior)
double balanceCost(const Eigen::Vector2d& residual, double weight, const NormParams& norm) {
    return 0.5 * weight * applyNorm(residual, norm);
}

// ============================================================================
// CasADi Symbolic Versions (for derivatives)
// ============================================================================

::casadi::SX CoMPosCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

::casadi::SX CoMVelCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

::casadi::SX EEPosCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

::casadi::SX EEVelCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

::casadi::SX uprightCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return 0.5 * weight * applyNorm(residual, norm);
}

::casadi::SX balanceCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return 0.5 * weight * applyNorm(residual, norm);
}

} // namespace ilqr