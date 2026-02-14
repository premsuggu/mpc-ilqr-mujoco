#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <cmath>
#include <algorithm>

namespace ilqr {

/**
 * @brief Norm types for robust cost functions
 */
enum class NormType {
    Quadratic = 0,      // r^T * r (no 0.5 factor)
    L22 = 1,            // ((r·r)^(q/2) + p^(2q))^(1/(2q)) - p
    L2 = 2,             // sqrt(r^T * r + p^2) - p
    Cosh = 3,           // p^2 * (cosh(r/p) - 1)
    SmoothAbs2Loss = 7  // (|r|^q + p^q)^(1/q) - p
};

/**
 * @brief Parameters for norm computation
 */
struct NormParams {
    NormType type = NormType::Quadratic;
    double p = 0.0;
    double q = 0.0;
    
    NormParams() = default;
    NormParams(NormType t, double p_val = 0.0, double q_val = 0.0)
        : type(t), p(p_val), q(q_val) {}
};

// Numerical (Eigen) norm - value only
double applyNorm(const Eigen::VectorXd& residual, const NormParams& params);

// Symbolic (CasADi) norm functions
::casadi::SX applyNorm(const ::casadi::SX& residual, const NormParams& params);
::casadi::SX normQuadratic(const ::casadi::SX& r);
::casadi::SX normL2(const ::casadi::SX& r, double p);
::casadi::SX normL22(const ::casadi::SX& r, double p, double q);
::casadi::SX normCosh(const ::casadi::SX& r, double p);
::casadi::SX normSmoothAbs2Loss(const ::casadi::SX& r, double p, double q);

}  // namespace ilqr
