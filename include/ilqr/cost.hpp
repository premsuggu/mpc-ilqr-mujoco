#pragma once

#include "norm.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace ilqr {

// ============================================================================
// Core Cost Functions - Clean residual-only interface
// ============================================================================

// Posture cost: 0.5 * x_err^T * Q * x_err
// Q has non-zero entries only at joint-angle indices [7:nq] (DeepMind "Posture")
double PostureCost(const Eigen::VectorXd& x_err, const Eigen::MatrixXd& Q);

// Control cost: 0.5 * u_err^T * R * u_err (always quadratic)
double ControlCost(const Eigen::VectorXd& u_err, const Eigen::MatrixXd& R);

// Height cost: weight * norm(torso_z - goal_z)  [scalar residual, DeepMind "Height"]
double HeightCost(double residual, double weight, const NormParams& norm);
::casadi::SX HeightCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm);

// CoM velocity cost: weight * norm(residual)
double CoMVelCost(const Eigen::Vector3d& residual, double weight, const NormParams& norm);
::casadi::SX CoMVelCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm);

// Upright cost: 0.5 * weight * norm(residual)
double uprightCost(const Eigen::Vector3d& residual, double weight, const NormParams& norm);
::casadi::SX uprightCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm);

// Balance cost: 0.5 * weight * norm(residual)
double balanceCost(const Eigen::Vector2d& residual, double weight, const NormParams& norm);
::casadi::SX balanceCost(const ::casadi::SX& residual, const ::casadi::SX& weight, const NormParams& norm);

} // namespace ilqr
