#pragma once

#include "norm.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace ilqr {

// ============================================================================
// Cost Helper Functions - Build cost terms with configurable norms
// ============================================================================

// Numerical (Eigen-based) cost helper for tracking tasks
// Computes: weight * norm(residual, norm_params)
double buildCostWithNorm(
    const Eigen::VectorXd& residual,
    double weight,
    const NormParams& norm_params
);

// Symbolic (CasADi-based) cost helper for tracking tasks
// Computes: weight * norm(residual, norm_params)
::casadi::SX buildCostWithNorm(
    const ::casadi::SX& residual,
    const ::casadi::SX& weight,
    const NormParams& norm_params
);

// ============================================================================
// Cost Term Builders - Semantic wrappers for specific cost terms
// ============================================================================

// CoM position tracking cost: weight * norm(com_pos - com_ref)
double buildCoMPositionCost(
    const Eigen::Vector3d& com_pos,
    const Eigen::Vector3d& com_ref,
    double weight,
    const NormParams& norm_params
);

::casadi::SX buildCoMPositionCost(
    const ::casadi::SX& com_pos,
    const ::casadi::SX& com_ref,
    const ::casadi::SX& weight,
    const NormParams& norm_params
);

// CoM velocity tracking cost: weight * norm(com_vel - com_vel_ref)
double buildCoMVelocityCost(
    const Eigen::Vector3d& com_vel,
    const Eigen::Vector3d& com_vel_ref,
    double weight,
    const NormParams& norm_params
);

::casadi::SX buildCoMVelocityCost(
    const ::casadi::SX& com_vel,
    const ::casadi::SX& com_vel_ref,
    const ::casadi::SX& weight,
    const NormParams& norm_params
);

// End-effector position tracking cost: weight * norm(ee_pos - ee_ref)
double buildEEPositionCost(
    const Eigen::Vector3d& ee_pos,
    const Eigen::Vector3d& ee_ref,
    double weight,
    const NormParams& norm_params
);

::casadi::SX buildEEPositionCost(
    const ::casadi::SX& ee_pos,
    const ::casadi::SX& ee_ref,
    const ::casadi::SX& weight,
    const NormParams& norm_params
);

// End-effector velocity tracking cost: weight * norm(ee_vel - ee_vel_ref)
double buildEEVelocityCost(
    const Eigen::Vector3d& ee_vel,
    const Eigen::Vector3d& ee_vel_ref,
    double weight,
    const NormParams& norm_params
);

::casadi::SX buildEEVelocityCost(
    const ::casadi::SX& ee_vel,
    const ::casadi::SX& ee_vel_ref,
    const ::casadi::SX& weight,
    const NormParams& norm_params
);

// Upright orientation cost: 0.5 * weight * norm(torso_z - [0,0,1])
// Note: Uses 0.5 factor to match original implementation
double buildUprightCost(
    const Eigen::Vector3d& torso_z,
    double weight,
    const NormParams& norm_params
);

::casadi::SX buildUprightCost(
    const ::casadi::SX& torso_z,
    const ::casadi::SX& weight,
    const NormParams& norm_params
);

// Balance cost: 0.5 * weight * norm(com_xy - support_center_xy)
// Note: Uses 0.5 factor to match original implementation
double buildBalanceCost(
    const Eigen::Vector2d& com_xy,
    const Eigen::Vector2d& support_center_xy,
    double weight,
    const NormParams& norm_params
);

::casadi::SX buildBalanceCost(
    const ::casadi::SX& com_xy,
    const ::casadi::SX& support_center_xy,
    const ::casadi::SX& weight,
    const NormParams& norm_params
);

} // namespace ilqr
