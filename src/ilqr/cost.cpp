#include "ilqr/cost.hpp"

namespace ilqr {

// ============================================================================
// Core Cost Functions - Take residuals only
// ============================================================================

// Posture cost: penalises joint angles qpos[7:nq] from reference (Quadratic).
// Q has non-zero entries only at indices [7:nq]; base DOF and velocities are zero.
double PostureCost(const Eigen::VectorXd& x_err, const Eigen::MatrixXd& Q) {
    return 0.5 * x_err.transpose() * Q * x_err;
}

// Control cost: quadratic only (LQR design)
double ControlCost(const Eigen::VectorXd& u_err, const Eigen::MatrixXd& R) {
    return 0.5 * u_err.transpose() * R * u_err;
}

// Height cost: scalar residual (torso z - goal z), DeepMind "Height"
double HeightCost(double residual, double weight, const NormParams& norm) {
    Eigen::VectorXd r(1);
    r(0) = residual;
    return weight * applyNorm(r, norm);
}

// CoM velocity cost
double CoMVelCost(const Eigen::Vector3d& residual, double weight, const NormParams& norm) {
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

::casadi::SX HeightCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

::casadi::SX CoMVelCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return weight * applyNorm(residual, norm);
}

::casadi::SX uprightCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return 0.5 * weight * applyNorm(residual, norm);
}

::casadi::SX balanceCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm) {
    return 0.5 * weight * applyNorm(residual, norm);
}

} // namespace ilqr