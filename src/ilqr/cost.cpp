#include "ilqr/cost.hpp"

namespace ilqr {

// ============================================================================
// Generic Cost Helpers
// ============================================================================

double buildCostWithNorm(
    const Eigen::VectorXd& residual,
    double weight,
    const NormParams& norm_params
) {
    double norm_value = applyNorm(residual, norm_params);
    return weight * norm_value;
}

::casadi::SX buildCostWithNorm(
    const ::casadi::SX& residual,
    const ::casadi::SX& weight,
    const NormParams& norm_params
) {
    ::casadi::SX norm_value = applyNorm(residual, norm_params);
    return weight * norm_value;
}

// ============================================================================
// CoM Position Cost
// ============================================================================

double buildCoMPositionCost(
    const Eigen::Vector3d& com_pos,
    const Eigen::Vector3d& com_ref,
    double weight,
    const NormParams& norm_params
) {
    Eigen::Vector3d residual = com_pos - com_ref;
    return buildCostWithNorm(residual, weight, norm_params);
}

::casadi::SX buildCoMPositionCost(
    const ::casadi::SX& com_pos,
    const ::casadi::SX& com_ref,
    const ::casadi::SX& weight,
    const NormParams& norm_params
) {
    ::casadi::SX residual = com_pos - com_ref;
    return buildCostWithNorm(residual, weight, norm_params);
}

// ============================================================================
// CoM Velocity Cost
// ============================================================================

double buildCoMVelocityCost(
    const Eigen::Vector3d& com_vel,
    const Eigen::Vector3d& com_vel_ref,
    double weight,
    const NormParams& norm_params
) {
    Eigen::Vector3d residual = com_vel - com_vel_ref;
    return buildCostWithNorm(residual, weight, norm_params);
}

::casadi::SX buildCoMVelocityCost(
    const ::casadi::SX& com_vel,
    const ::casadi::SX& com_vel_ref,
    const ::casadi::SX& weight,
    const NormParams& norm_params
) {
    ::casadi::SX residual = com_vel - com_vel_ref;
    return buildCostWithNorm(residual, weight, norm_params);
}

// ============================================================================
// End-Effector Position Cost
// ============================================================================

double buildEEPositionCost(
    const Eigen::Vector3d& ee_pos,
    const Eigen::Vector3d& ee_ref,
    double weight,
    const NormParams& norm_params
) {
    Eigen::Vector3d residual = ee_pos - ee_ref;
    return buildCostWithNorm(residual, weight, norm_params);
}

::casadi::SX buildEEPositionCost(
    const ::casadi::SX& ee_pos,
    const ::casadi::SX& ee_ref,
    const ::casadi::SX& weight,
    const NormParams& norm_params
) {
    ::casadi::SX residual = ee_pos - ee_ref;
    return buildCostWithNorm(residual, weight, norm_params);
}

// ============================================================================
// End-Effector Velocity Cost
// ============================================================================

double buildEEVelocityCost(
    const Eigen::Vector3d& ee_vel,
    const Eigen::Vector3d& ee_vel_ref,
    double weight,
    const NormParams& norm_params
) {
    Eigen::Vector3d residual = ee_vel - ee_vel_ref;
    return buildCostWithNorm(residual, weight, norm_params);
}

::casadi::SX buildEEVelocityCost(
    const ::casadi::SX& ee_vel,
    const ::casadi::SX& ee_vel_ref,
    const ::casadi::SX& weight,
    const NormParams& norm_params
) {
    ::casadi::SX residual = ee_vel - ee_vel_ref;
    return buildCostWithNorm(residual, weight, norm_params);
}

// ============================================================================
// Upright Cost
// Note: Uses 0.5 factor to match original implementation
// ============================================================================

double buildUprightCost(
    const Eigen::Vector3d& torso_z,
    double weight,
    const NormParams& norm_params
) {
    Eigen::Vector3d up(0.0, 0.0, 1.0);
    Eigen::Vector3d residual = torso_z - up;
    double norm_value = applyNorm(residual, norm_params);
    return 0.5 * weight * norm_value;
}

::casadi::SX buildUprightCost(
    const ::casadi::SX& torso_z,
    const ::casadi::SX& weight,
    const NormParams& norm_params
) {
    ::casadi::SX up = ::casadi::SX::vertcat({0.0, 0.0, 1.0});
    ::casadi::SX residual = torso_z - up;
    ::casadi::SX norm_value = applyNorm(residual, norm_params);
    return 0.5 * weight * norm_value;
}

// ============================================================================
// Balance Cost
// Note: Uses 0.5 factor to match original implementation
// ============================================================================

double buildBalanceCost(
    const Eigen::Vector2d& com_xy,
    const Eigen::Vector2d& support_center_xy,
    double weight,
    const NormParams& norm_params
) {
    Eigen::Vector2d residual = com_xy - support_center_xy;
    double norm_value = applyNorm(residual, norm_params);
    return 0.5 * weight * norm_value;
}

::casadi::SX buildBalanceCost(
    const ::casadi::SX& com_xy,
    const ::casadi::SX& support_center_xy,
    const ::casadi::SX& weight,
    const NormParams& norm_params
) {
    ::casadi::SX residual = com_xy - support_center_xy;
    ::casadi::SX norm_value = applyNorm(residual, norm_params);
    return 0.5 * weight * norm_value;
}

} // namespace ilqr
