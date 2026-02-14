#include "ilqr/norm.hpp"
#include <stdexcept>

namespace ilqr {

// Numerical (Eigen) implementations
double applyNorm(const Eigen::VectorXd& residual, const NormParams& params) {
    switch (params.type) {
        case NormType::Quadratic:
            return residual.squaredNorm();  // r^T * r
        
        case NormType::L2:
            return std::sqrt(residual.squaredNorm() + params.p * params.p) - params.p;
        
        case NormType::L22: {
            double c = residual.squaredNorm();
            double a = std::pow(c, params.q / 2.0) + std::pow(params.p, params.q);
            return std::pow(a, 1.0 / params.q) - params.p;
        }
        
        case NormType::Cosh: {
            double value = 0.0;
            for (int i = 0; i < residual.size(); ++i) {
                value += params.p * params.p * (std::cosh(residual(i) / params.p) - 1.0);
            }
            return value;
        }
        
        case NormType::SmoothAbs2Loss: {
            double value = 0.0;
            double p_q = std::pow(params.p, params.q);
            for (int i = 0; i < residual.size(); ++i) {
                double abs_r = std::abs(residual(i));
                double d = std::pow(abs_r, params.q);
                value += std::pow(d + p_q, 1.0 / params.q) - params.p;
            }
            return value;
        }
        
        default:
            throw std::runtime_error("Unknown norm type");
    }
}

// Symbolic (CasADi) implementations
::casadi::SX applyNorm(const ::casadi::SX& residual, const NormParams& params) {
    switch (params.type) {
        case NormType::Quadratic:
            return normQuadratic(residual);
        case NormType::L2:
            return normL2(residual, params.p);
        case NormType::L22:
            return normL22(residual, params.p, params.q);
        case NormType::Cosh:
            return normCosh(residual, params.p);
        case NormType::SmoothAbs2Loss:
            return normSmoothAbs2Loss(residual, params.p, params.q);
        default:
            throw std::runtime_error("Unknown norm type");
    }
}

::casadi::SX normQuadratic(const ::casadi::SX& r) {
    return ::casadi::SX::dot(r, r);  // r^T * r (no 0.5)
}

::casadi::SX normL2(const ::casadi::SX& r, double p) {
    ::casadi::SX r_norm_sq = ::casadi::SX::dot(r, r);
    return ::casadi::SX::sqrt(r_norm_sq + p * p) - p;
}

::casadi::SX normL22(const ::casadi::SX& r, double p, double q) {
    ::casadi::SX c = ::casadi::SX::dot(r, r);
    ::casadi::SX a = ::casadi::SX::pow(c, q / 2.0) + std::pow(p, 2.0 * q);
    return ::casadi::SX::pow(a, 1.0 / (2.0 * q)) - p;
}

::casadi::SX normCosh(const ::casadi::SX& r, double p) {
    ::casadi::SX value = 0.0;
    for (int i = 0; i < r.size1(); ++i) {
        value += p * p * (::casadi::SX::cosh(r(i) / p) - 1.0);
    }
    return value;
}

::casadi::SX normSmoothAbs2Loss(const ::casadi::SX& r, double p, double q) {
    ::casadi::SX value = 0.0;
    double p_q = std::pow(p, q);
    for (int i = 0; i < r.size1(); ++i) {
        ::casadi::SX abs_r = ::casadi::SX::sqrt(r(i) * r(i));
        ::casadi::SX d = ::casadi::SX::pow(abs_r, q);
        value += ::casadi::SX::pow(d + p_q, 1.0 / q) - p;
    }
    return value;
}

}  // namespace ilqr
