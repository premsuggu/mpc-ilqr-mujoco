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
        
        case NormType::SmoothAbsLoss: {
            double value = 0.0;
            for (int i = 0; i < residual.size(); ++i) {
                double s = std::sqrt(residual(i) * residual(i) + params.p * params.p);
                value += s - params.p;
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
        
        case NormType::Rectify: {
            double value = 0.0;
            for (int i = 0; i < residual.size(); ++i) {
                if (params.p > 0) {
                    double exp_val = std::exp(residual(i) / params.p);
                    value += params.p * std::log(1.0 + exp_val);
                } else {
                    // p=0 case: max(0, x)
                    value += residual(i) > 0 ? residual(i) : 0.0;
                }
            }
            return value;
        }
        
        default:
            throw std::runtime_error("Unknown norm type");
    }
}

// Norm gradient: ∂norm/∂r
void applyNormGradient(const Eigen::VectorXd& residual, const NormParams& params, Eigen::VectorXd& gradient) {
    int n = residual.size();
    gradient.resize(n);
    
    switch (params.type) {
        case NormType::Quadratic:
            // ∂(r'*r)/∂r = 2r
            gradient = 2.0 * residual;
            break;
        
        case NormType::L2: {
            // ∂(sqrt(r'*r + p²) - p)/∂r = r / sqrt(r'*r + p²)
            double norm_val = std::sqrt(residual.squaredNorm() + params.p * params.p);
            if (norm_val > 1e-12) {
                gradient = residual / norm_val;
            } else {
                gradient.setZero();
            }
            break;
        }
        
        case NormType::Cosh:
            // ∂(p² * (cosh(r_i/p) - 1))/∂r_i = p * sinh(r_i/p)
            for (int i = 0; i < n; ++i) {
                gradient(i) = params.p * std::sinh(residual(i) / params.p);
            }
            break;
            
        case NormType::SmoothAbsLoss:
            // ∂(sqrt(r²+p²) - p)/∂r = r / sqrt(r²+p²)
            for (int i = 0; i < n; ++i) {
                double s = std::sqrt(residual(i)*residual(i) + params.p*params.p);
                gradient(i) = (s > 1e-12) ? residual(i) / s : 0.0;
            }
            break;
        
        case NormType::SmoothAbs2Loss: {
            // ∂((|r|^q + p^q)^(1/q) - p)/∂r
            double p_q = std::pow(params.p, params.q);
            for (int i = 0; i < n; ++i) {
                double abs_r = std::abs(residual(i));
                double d = std::pow(abs_r, params.q);
                double e = d + p_q;
                double s = std::pow(e, 1.0 / params.q);
                if (s > 1e-12 && abs_r > 1e-12) {
                    gradient(i) = (s / e) * std::pow(abs_r, params.q - 2) * residual(i);
                } else {
                    gradient(i) = 0.0;
                }
            }
            break;
        }
        
        case NormType::Rectify:
            // ∂(p*log(1+exp(r/p)))/∂r = exp(r/p)/(1+exp(r/p))
            for (int i = 0; i < n; ++i) {
                if (params.p > 0) {
                    double exp_val = std::exp(residual(i) / params.p);
                    gradient(i) = exp_val / (1.0 + exp_val);
                } else {
                    gradient(i) = (residual(i) > 0) ? 1.0 : 0.0;
                }
            }
            break;
        
        default:
            throw std::runtime_error("Unknown norm type for gradient");
    }
}

// Norm Hessian: ∂²norm/∂r²
void applyNormHessian(const Eigen::VectorXd& residual, const NormParams& params, Eigen::MatrixXd& hessian) {
    int n = residual.size();
    hessian.resize(n, n);
    hessian.setZero();
    
    switch (params.type) {
        case NormType::Quadratic:
            // ∂²(r'*r)/∂r² = 2I
            hessian = 2.0 * Eigen::MatrixXd::Identity(n, n);
            break;
        
        case NormType::L2: {
            // ∂²(sqrt(r'*r + p²))/∂r² = (I - g*g')/sqrt(r'*r + p²) where g = r/sqrt(r'*r+p²)
            double r_norm_sq = residual.squaredNorm();
            double denom = std::sqrt(r_norm_sq + params.p * params.p);
            if (denom > 1e-12) {
                Eigen::VectorXd g = residual / denom;
                hessian = (Eigen::MatrixXd::Identity(n, n) - g * g.transpose()) / denom;
            }
            break;
        }
        
        case NormType::Cosh:
            // ∂²(p² * (cosh(r_i/p) - 1))/∂r_i² = cosh(r_i/p)  (diagonal only)
            for (int i = 0; i < n; ++i) {
                hessian(i, i) = std::cosh(residual(i) / params.p);
            }
            break;
            
        case NormType::SmoothAbsLoss:
            // ∂²(sqrt(r²+p²) - p)/∂r² = (1 - g²)/sqrt(r²+p²) where g = r/sqrt(r²+p²)
            for (int i = 0; i < n; ++i) {
                double r_sq = residual(i) * residual(i);
                double s = std::sqrt(r_sq + params.p * params.p);
                if (s > 1e-12) {
                    double g_sq = r_sq / (r_sq + params.p * params.p);
                    hessian(i, i) = (1.0 - g_sq) / s;
                }
            }
            break;
        
        case NormType::SmoothAbs2Loss: {
            // Diagonal approximation for SmoothAbs2Loss
            double p_q = std::pow(params.p, params.q);
            for (int i = 0; i < n; ++i) {
                double abs_r = std::abs(residual(i));
                if (abs_r > 1e-12) {
                    double d = std::pow(abs_r, params.q);
                    double e = d + p_q;
                    double s = std::pow(e, 1.0 / params.q);
                    hessian(i, i) = s * std::pow(abs_r, params.q - 2) * (params.q - 1) * (1 - d / e) / e;
                }
            }
            break;
        }
        
        case NormType::Rectify:
            // ∂²(p*log(1+exp(r/p)))/∂r² = exp(r/p)/(p*(1+exp(r/p))²)
            for (int i = 0; i < n; ++i) {
                if (params.p > 0) {
                    double exp_val = std::exp(residual(i) / params.p);
                    hessian(i, i) = exp_val / (params.p * (1.0 + exp_val) * (1.0 + exp_val));
                }
            }
            break;
        
        default:
            throw std::runtime_error("Unknown norm type for Hessian");
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
        case NormType::SmoothAbsLoss:
            return normSmoothAbsLoss(residual, params.p);
        case NormType::SmoothAbs2Loss:
            return normSmoothAbs2Loss(residual, params.p, params.q);
        case NormType::Rectify:
            return normRectify(residual, params.p);
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

::casadi::SX normSmoothAbsLoss(const ::casadi::SX& r, double p) {
    ::casadi::SX value = 0.0;
    for (int i = 0; i < r.size1(); ++i) {
        ::casadi::SX s = ::casadi::SX::sqrt(r(i) * r(i) + p * p);
        value += s - p;
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

::casadi::SX normRectify(const ::casadi::SX& r, double p) {
    ::casadi::SX value = 0.0;
    if (p > 0) {
        for (int i = 0; i < r.size1(); ++i) {
            ::casadi::SX exp_val = ::casadi::SX::exp(r(i) / p);
            value += p * ::casadi::SX::log(1.0 + exp_val);
        }
    } else {
        // p=0 case: max(0, x)
        for (int i = 0; i < r.size1(); ++i) {
            value += ::casadi::SX::fmax(0.0, r(i));
        }
    }
    return value;
}

}  // namespace ilqr
