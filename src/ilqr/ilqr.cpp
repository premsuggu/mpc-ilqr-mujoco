#include "ilqr/ilqr.hpp"
#include "ilqr/cost.hpp"
#include <iostream>
#include <chrono>

#ifdef ENABLE_PROFILING
#include <map>
#include <vector>
struct ProfileData {
    std::vector<double> times;
};
extern std::map<std::string, ProfileData> prof_data;
#endif

iLQR::iLQR(RobotUtils& robot, int N, double dt, const std::string& urdf_path) 
        : robot_(robot), derivatives_(urdf_path, true),
          N_(N), dt_(dt), reg_lambda_(1e-6), max_iterations_(10), tolerance_(1e-4),
          reg_min_(1e-6), reg_max_(100.0), reg_increase_factor_(10.0), reg_decrease_factor_(10.0),
          trust_region_good_(0.75), trust_region_poor_(0.25),
          num_line_search_steps_(10), min_linesearch_step_(1e-3),
          line_search_tolerance_(1e-6), quu_regularization_(1e-4), convergence_threshold_(1e-8) {
    // Set up all the storage for trajectories, gains, and derivatives
    int nx = robot_.nx();
    int nu = robot_.nu();
    xbar_.resize(N_ + 1);
    ubar_.resize(N_);
    K_.resize(N_);
    kff_.resize(N_);
    A_.resize(N_);
    B_.resize(N_);
    lx_.resize(N_ + 1);
    lu_.resize(N_);
    lxx_.resize(N_ + 1);
    luu_.resize(N_);
    lxu_.resize(N_);
    
    // Initialize expected reduction vector
    dV_.setZero();
    
    for (int t = 0; t <= N_; ++t) {
        xbar_[t] = Eigen::VectorXd::Zero(nx);
        lx_[t] = Eigen::VectorXd::Zero(nx);
        lxx_[t] = Eigen::MatrixXd::Zero(nx, nx);
    }
    for (int t = 0; t < N_; ++t) {
        ubar_[t] = Eigen::VectorXd::Zero(nu);
        lu_[t] = Eigen::VectorXd::Zero(nu);
        K_[t] = Eigen::MatrixXd::Zero(nu, nx);
        kff_[t] = Eigen::VectorXd::Zero(nu);
        A_[t] = Eigen::MatrixXd::Zero(nx, nx);
        B_[t] = Eigen::MatrixXd::Zero(nx, nu);
        lxx_[t] = Eigen::MatrixXd::Zero(nx, nx);
        luu_[t] = Eigen::MatrixXd::Zero(nu, nu);
        lxu_[t] = Eigen::MatrixXd::Zero(nx, nu);
    }
    std::cout << "iLQR initialized with horizon N=" << N_ << ", dt=" << dt_ << std::endl;
    // Set body names for Pelvis/Feet cost derivatives (must be called after robot body names are set)
    derivatives_.setPelvisFeetBodyNames(
        robot_.getPelvisBodyName(),
        robot_.getRightFootBodyName(),
        robot_.getLeftFootBodyName()
    );
}

void iLQR::initializeWithReference(const Eigen::VectorXd& x0,
                                  const std::vector<Eigen::VectorXd>& x_ref,
                                  const std::vector<Eigen::VectorXd>& u_ref,
                                  const std::vector<Eigen::Vector3d>& height_ref,
                                  const std::vector<Eigen::VectorXd>* prev_xbar,
                                  const std::vector<Eigen::VectorXd>* prev_ubar) {
    
    // Store height reference
    height_ref_ = height_ref;
    
    xbar_[0] = x0;
    
    // INITIAL GUESS STRATEGY SELECTION
    // 0 = Zero Control (baseline)
    // 1 = Gravity Compensation (recommended)
    int strategy = 1;  // Use gravity compensation for best performance
    
    // Use warm start if available
    if (prev_xbar && prev_ubar && 
        prev_xbar->size() == xbar_.size() && prev_ubar->size() == ubar_.size()) {
        
        // Shift the entire previous solution forward by one timestep
        for (int t = 0; t < N_ - 1; ++t) {
            ubar_[t] = (*prev_ubar)[t + 1];
        }
        ubar_[N_ - 1] = (*prev_ubar)[N_ - 1];
        
        for (int t = 0; t < N_ - 1; ++t) {
            xbar_[t + 1] = (*prev_xbar)[t + 2];
        }
        robot_.rolloutOneStep(xbar_[N_ - 1], ubar_[N_ - 1], xbar_[N_]);
        
    } else {
        // COLD START - Apply selected strategy
        
        if (strategy == 0) {
            // Zero Control (baseline)
            std::cout << "Initial Guess Strategy: Zero Control" << std::endl;
            for (int t = 0; t < N_; ++t) {
                ubar_[t] = Eigen::VectorXd::Zero(robot_.nu());
            }
            
        } else if (strategy == 1) {
            // Gravity Compensation (recommended)
            std::cout << "Initial Guess Strategy: Gravity Compensation" << std::endl;
            
            // Compute gravity compensation using built-in function
            Eigen::VectorXd u_gravity;
            robot_.computeGravComp(u_gravity);
            
            for (int t = 0; t < N_; ++t) {
                ubar_[t] = u_gravity;  // Same gravity compensation for all steps
            }
            
        } else {
            // Default: Zero control
            std::cout << "Initial Guess Strategy: Default (Zero Control)" << std::endl;
            for (int t = 0; t < N_; ++t) {
                ubar_[t] = Eigen::VectorXd::Zero(robot_.nu());
            }
        }
        
        // Forward rollout to ensure consistent trajectory
        for (int t = 0; t < N_; ++t) {
            robot_.rolloutOneStep(xbar_[t], ubar_[t], xbar_[t + 1]);
        }
    }
}

void iLQR::forwardRolloutNominal() {
    // Roll out trajectory using current controls
    for (int t = 0; t < N_; ++t) {
        robot_.rolloutOneStep(xbar_[t], ubar_[t], xbar_[t + 1]);
    }
}

void iLQR::computeLinearization() {
    // Compute A_t, B_t matrices via finite differences
    for (int t = 0; t < N_; ++t) {
        robot_.linearizeDynamicsFD(xbar_[t], ubar_[t], A_[t], B_[t]);
    }
}

void iLQR::computeCostQuadratics(const std::vector<Eigen::VectorXd>& x_ref,
                                 const std::vector<Eigen::VectorXd>& u_ref) {
    // Store references for backward pass
    x_ref_ = x_ref;
    u_ref_ = u_ref;
    
    // Compute cost gradients and Hessians for all time steps
    for (int t = 0; t < N_; ++t) {
        Eigen::VectorXd x_err = xbar_[t] - x_ref[t];
        Eigen::VectorXd u_err = ubar_[t] - u_ref[t];
        
        // Tracking cost gradients
        lx_[t] = robot_.Q() * x_err;
        lu_[t] = robot_.R() * u_err;
        
        // Tracking cost hessians
        lxx_[t] = robot_.Q();
        luu_[t] = robot_.R();
        lxu_[t] = Eigen::MatrixXd::Zero(robot_.nx(), robot_.nu());
        
        // ADD HEIGHT TRACKING DERIVATIVES if weight > 0
        if (robot_.getHeightWeight() > 0.0) {
            addHeightCostDerivatives(t, height_ref_[t](2));
        }
        
        // ADD CoM VELOCITY TRACKING DERIVATIVES if weight > 0 (SEPARATE from position)
        if (robot_.getVelocityWeight() > 0.0) {
            addVelocityCostDerivatives(t);
        }
        
        // ADD UPRIGHT COST DERIVATIVES if weight > 0
        if (robot_.getUprightWeight() > 0.0) {
            addUprightCostDerivatives(t);
        }
        
        // ADD BALANCE COST DERIVATIVES if weight > 0
        if (robot_.getBalanceWeight() > 0.0) {
            addBalanceCostDerivatives(t);
        }
        
        // ADD PELVIS/FEET COST DERIVATIVES if weight > 0
        if (robot_.getPelvisFeetWeight() > 0.0) {
            addPelvisFeetCostDerivatives(t);
        }
        
        // ADD CONSTRAINT DERIVATIVES
        Eigen::VectorXd constraint_grad_x(robot_.nx());
        Eigen::VectorXd constraint_grad_u(robot_.nu());
        robot_.constraintGradients(xbar_[t], ubar_[t], constraint_grad_x, constraint_grad_u);
        
        // Add constraint gradients to cost gradients
        lx_[t] += constraint_grad_x;
        lu_[t] += constraint_grad_u;
        
        // Add constraint hessians to cost hessians
        Eigen::MatrixXd constraint_hess_xx(robot_.nx(), robot_.nx());
        Eigen::MatrixXd constraint_hess_uu(robot_.nu(), robot_.nu());
        robot_.constraintHessians(xbar_[t], ubar_[t], constraint_hess_xx, constraint_hess_uu);
        
        lxx_[t] += constraint_hess_xx;
        luu_[t] += constraint_hess_uu;
        // lxu remains zero for separable constraints
    }
    
    // Terminal cost (only joint limits, no control constraints)
    Eigen::VectorXd x_err_N = xbar_[N_] - x_ref[N_];
    lx_[N_] = robot_.Qf() * x_err_N;
    lxx_[N_] = robot_.Qf();
    
    // ADD TERMINAL HEIGHT TRACKING DERIVATIVES if weight > 0
    if (robot_.getHeightWeight() > 0.0) {
        addHeightCostDerivatives(N_, height_ref_[N_](2));
    }
    
    // ADD TERMINAL UPRIGHT COST DERIVATIVES if weight > 0
    if (robot_.getUprightWeight() > 0.0) {
        addUprightCostDerivatives(N_);
    }
    
    // ADD TERMINAL BALANCE COST DERIVATIVES if weight > 0
    if (robot_.getBalanceWeight() > 0.0) {
        addBalanceCostDerivatives(N_);
    }
    
    // ADD TERMINAL PELVIS/FEET COST DERIVATIVES if weight > 0
    if (robot_.getPelvisFeetWeight() > 0.0) {
        addPelvisFeetCostDerivatives(N_);
    }
    
    // Add terminal constraint gradients and hessians (joint limits only)
    Eigen::VectorXd terminal_constraint_grad_x(robot_.nx());
    Eigen::VectorXd dummy_u = Eigen::VectorXd::Zero(robot_.nu());  // No control at terminal
    Eigen::VectorXd dummy_grad_u(robot_.nu());
    robot_.constraintGradients(xbar_[N_], dummy_u, terminal_constraint_grad_x, dummy_grad_u);
    
    Eigen::MatrixXd terminal_constraint_hess_xx(robot_.nx(), robot_.nx());
    Eigen::MatrixXd dummy_hess_uu(robot_.nu(), robot_.nu());
    robot_.constraintHessians(xbar_[N_], dummy_u, terminal_constraint_hess_xx, dummy_hess_uu);
    
    lx_[N_] += terminal_constraint_grad_x;
    lxx_[N_] += terminal_constraint_hess_xx;
}

void iLQR::setRegularization(double lambda) {
    reg_lambda_ = lambda;
}

void iLQR::setNormParams(const std::map<std::string, ilqr::NormParams>& norm_params) {
    norm_params_ = norm_params;  // Store for use in computeTotalCost
    derivatives_.setNormParams(norm_params);
}

void iLQR::configureSolver(double reg_min, double reg_max, double reg_increase_factor,
                          double reg_decrease_factor, double trust_region_good,
                          double trust_region_poor, int num_line_search_steps,
                          double min_linesearch_step, double line_search_tolerance,
                          double quu_regularization, double convergence_threshold) {
    reg_min_ = reg_min;
    reg_max_ = reg_max;
    reg_increase_factor_ = reg_increase_factor;
    reg_decrease_factor_ = reg_decrease_factor;
    trust_region_good_ = trust_region_good;
    trust_region_poor_ = trust_region_poor;
    num_line_search_steps_ = num_line_search_steps;
    min_linesearch_step_ = min_linesearch_step;
    line_search_tolerance_ = line_search_tolerance;
    quu_regularization_ = quu_regularization;
    convergence_threshold_ = convergence_threshold;
}

void iLQR::backwardPass() { 
    // Reset expected reduction
    dV_.setZero();
    
    // V_N(x_N) = l_f(x_N), so ∇V_N = ∇l_f and ∇²V_N = ∇²l_f
    VxN_ = lx_[N_];   // Terminal cost gradient
    VxxN_ = lxx_[N_]; // Terminal cost Hessian
    
    // Backward recursion starts with terminal cost derivatives
    Eigen::VectorXd Vx = VxN_;
    Eigen::MatrixXd Vxx = VxxN_;
    
    for (int t = N_ - 1; t >= 0; --t) {
        // Q-function quadratics with safe Eigen evaluation
        Eigen::VectorXd Atv = (A_[t].transpose() * Vx).eval();
        Eigen::VectorXd Btv = (B_[t].transpose() * Vx).eval();
        
        Eigen::VectorXd Qx = lx_[t] + Atv;
        Eigen::VectorXd Qu = lu_[t] + Btv;
        Eigen::MatrixXd Qxx = lxx_[t] + A_[t].transpose() * Vxx * A_[t];
        Eigen::MatrixXd Quu = luu_[t] + B_[t].transpose() * Vxx * B_[t];
        
        // Cross-term with safe construction
        Eigen::MatrixXd Qxu(robot_.nx(), robot_.nu());
        Qxu = lxu_[t];
        Qxu.noalias() += A_[t].transpose() * Vxx * B_[t];
        
        // Regularization for numerical stability
        Quu += reg_lambda_ * Eigen::MatrixXd::Identity(Quu.rows(), Quu.cols());
        
        // Check positive definiteness of Quu
        Eigen::LLT<Eigen::MatrixXd> llt(Quu);
        if (llt.info() != Eigen::Success) {
            Quu += 1e-4 * Eigen::MatrixXd::Identity(Quu.rows(), Quu.cols());
        }
        
        // Compute gains
        K_[t] = -Quu.ldlt().solve(Qxu.transpose());  // K = -Quu^{-1} Qux
        kff_[t] = -Quu.ldlt().solve(Qu);             // k = -Quu^{-1} Qu
        
        // Check for non-finite gains and throw exception
        if (!K_[t].allFinite() || !kff_[t].allFinite()) {
            std::cerr << "ERROR: Non-finite gains at timestep " << t << std::endl;
            std::cerr << "  This indicates numerical instability (ill-conditioned Quu)" << std::endl;
            throw std::runtime_error("Backward pass failed: non-finite gains at t=" + std::to_string(t));
        }
        
        // Compute expected cost reduction (for trust region ratio)
        dV_(0) += kff_[t].dot(Qu);                     // Linear term
        dV_(1) += 0.5 * kff_[t].dot(Quu * kff_[t]);    // Quadratic term
        
        // Value function update with safe evaluation (corrected formulas from iLQR.tex)
        Eigen::VectorXd KTQu = (K_[t].transpose() * Qu).eval();
        Eigen::VectorXd KTQuuk = (K_[t].transpose() * Quu * kff_[t]).eval();
        // Q_ux^T * d_k - Note: In our notation Qxu is (nx x nu), so Q_ux = Qxu^T is (nu x nx)
        // Therefore Q_ux^T = (Qxu^T)^T = Qxu, and Q_ux^T * d_k = Qxu * kff_[t]
        Eigen::VectorXd Qux_T_dk = (Qxu * kff_[t]).eval(); 
        
        // Correct formula: s_k = Q_x + K_k^T Q_uu d_k + K_k^T Q_u + Q_ux^T d_k
        Vx = Qx + KTQuuk + KTQu + Qux_T_dk;
        
        // Correct formula: S_k = Q_xx + K_k^T Q_uu K_k + K_k^T Q_ux + Q_ux^T K_k
        Vxx = Qxx + K_[t].transpose() * Quu * K_[t] + K_[t].transpose() * Qxu.transpose() + Qxu * K_[t];
        
        // Ensure Vxx stays symmetric
        Vxx = 0.5 * (Vxx + Vxx.transpose());
    }
}

bool iLQR::forwardPassLineSearch(const Eigen::VectorXd& x0,
                                const std::vector<Eigen::VectorXd>& x_ref,
                                const std::vector<Eigen::VectorXd>& u_ref,
                                double& new_cost) {
    
    // Compute baseline cost
    double baseline_cost = computeTotalCost(xbar_, ubar_, x_ref, u_ref);
    
    // Generate log-scaled line search alphas (DeepMind MJPC style)
    // Formula: alpha[i] = exp(log(min) + i * step) where step = (log(max) - log(min)) / (num_steps - 1)
    // Note: Generated in increasing order, but we'll iterate in REVERSE to try largest steps first
    std::vector<double> alphas(num_line_search_steps_);
    if (num_line_search_steps_ > 1) {
        double log_max = std::log(1.0);
        double log_min = std::log(min_linesearch_step_);
        double step = (log_max - log_min) / std::max(num_line_search_steps_ - 1, 1);
        for (int i = 0; i < num_line_search_steps_; ++i) {
            alphas[i] = std::exp(log_min + i * step);
        }
    } else if (num_line_search_steps_ == 1) {
        alphas[0] = 1.0;
    }
    
    // Line search with log-scaled alphas
    // Standard line search: start aggressive (alpha=1.0), back off if needed
    for (int i = num_line_search_steps_ - 1; i >= 0; --i) {
        double alpha = alphas[i];
        // Forward pass with current alpha
        std::vector<Eigen::VectorXd> x_new(N_ + 1);
        std::vector<Eigen::VectorXd> u_new(N_);
        
        x_new[0] = x0;
        
        bool rollout_success = true;
        for (int t = 0; t < N_; ++t) {
            // Control law: u = ubar + alpha * k + K * (x - xbar)
            Eigen::VectorXd dx = x_new[t] - xbar_[t];
            u_new[t] = ubar_[t] + alpha * kff_[t] + K_[t] * dx;
            
            // Rollout one step
            try {
                robot_.rolloutOneStep(x_new[t], u_new[t], x_new[t + 1]);
            } catch (const std::exception& e) {
                rollout_success = false;
                break;
            }
        }
        
        if (!rollout_success) continue;
        
        // Compute cost of new trajectory
        double cost = computeTotalCost(x_new, u_new, x_ref, u_ref);
        
        // Accept if cost decreased (simple sufficient decrease condition)
        if (cost < baseline_cost - line_search_tolerance_) {
            xbar_ = x_new;
            ubar_ = u_new;
            new_cost = cost;
            return true;
        }
    }
    
    // If no alpha worked, line search failed
    new_cost = baseline_cost;
    return false;  // CRITICAL FIX: Signal that the line search failed
}

// ============================================================================
// Helper functions for residual computation
// ============================================================================

// Extract torso z-axis from quaternion for upright cost
inline Eigen::Vector3d computeTorsoZAxis(const Eigen::VectorXd& x) {
    // MuJoCo state: [x,y,z,qw,qx,qy,qz,...]
    double qw = x(3), qx = x(4), qy = x(5), qz = x(6);
    return Eigen::Vector3d(
        2.0 * (qx*qz + qw*qy),
        2.0 * (qy*qz - qw*qx),
        1.0 - 2.0 * (qx*qx + qy*qy)
    );
}

// Capture point: CP = com_xy + 0.3 * base_vel_xy (DeepMind fixed constant, world frame)
// x[nq+0], x[nq+1] = world-frame base linear velocity (MuJoCo convention)
inline Eigen::Vector2d computeCapturePoint(const Eigen::VectorXd& x, const Eigen::Vector3d& p_com, int nq) {
    return Eigen::Vector2d(
        p_com(0) + x(nq + 0) * 0.3,
        p_com(1) + x(nq + 1) * 0.3
    );
}

// Project capture point onto foot line segment [foot_l, foot_r] (clamp t in [0,1])
// Single-stance: pass same point for both endpoints
inline Eigen::Vector2d computePCP(const Eigen::Vector2d& cp,
                                   const Eigen::Vector2d& foot_l,
                                   const Eigen::Vector2d& foot_r) {
    Eigen::Vector2d seg = foot_r - foot_l;
    double len_sq = seg.squaredNorm();
    if (len_sq < 1e-6) return foot_l;
    double t = std::clamp((cp - foot_l).dot(seg) / len_sq, 0.0, 1.0);
    return foot_l + t * seg;
}

// Standing factor S = z / sqrt(z^2 + 0.45^2) - 0.4  (DeepMind walk.cc)
// Positive only when robot is sufficiently upright; gates balance and walk costs
inline double computeSfactor(double z_torso) {
    return z_torso / std::sqrt(z_torso * z_torso + 0.45 * 0.45) - 0.4;
}

// Compute support center from stance feet
inline Eigen::Vector2d computeSupportCenter(const RobotUtils& robot, int t) {
    bool left_stance = robot.isStance(0, t);
    bool right_stance = robot.isStance(1, t);
    
    if (left_stance && right_stance) {
        Eigen::Vector3d left_foot = robot.getEEReference(t, 0);
        Eigen::Vector3d right_foot = robot.getEEReference(t, 1);
        return Eigen::Vector2d(
            0.5 * (left_foot(0) + right_foot(0)),
            0.5 * (left_foot(1) + right_foot(1))
        );
    } else if (left_stance) {
        Eigen::Vector3d left_foot = robot.getEEReference(t, 0);
        return left_foot.head<2>();
    } else {                            // right_stance only
        Eigen::Vector3d right_foot = robot.getEEReference(t, 1);
        return right_foot.head<2>();
    }
}

// Get norm params with default fallback
inline ilqr::NormParams getNormParams(const std::map<std::string, ilqr::NormParams>& norm_params, 
                                     const std::string& key) {
    auto it = norm_params.find(key);
    return (it != norm_params.end()) ? it->second : ilqr::NormParams{ilqr::NormType::Quadratic, 1.0, 1.0};
}

double iLQR::computeTotalCost(const std::vector<Eigen::VectorXd>& x_traj,
                             const std::vector<Eigen::VectorXd>& u_traj,
                             const std::vector<Eigen::VectorXd>& x_ref,
                             const std::vector<Eigen::VectorXd>& u_ref) {
    double total_cost = 0.0;
    
    // Running cost
    for (int t = 0; t < N_; ++t) {
        // Posture cost (joint angles [7:nq], Quadratic)
        Eigen::VectorXd x_err = x_traj[t] - x_ref[t];
        total_cost += ilqr::PostureCost(x_err, robot_.Q());
        
        // Control cost
        Eigen::VectorXd u_err = u_traj[t] - u_ref[t];
        total_cost += ilqr::ControlCost(u_err, robot_.R());
        
        // Height (torso z) cost
        if (robot_.getHeightWeight() > 0.0) {
            double residual = x_traj[t](2) - height_ref_[t](2);
            total_cost += ilqr::HeightCost(residual, robot_.getHeightWeight(), getNormParams(norm_params_, "height"));
        }
        
        // Velocity cost: 2D world-frame base xy velocity, zero target
        if (robot_.getVelocityWeight() > 0.0) {
            Eigen::Vector2d residual = x_traj[t].segment(robot_.nq(), 2);
            total_cost += ilqr::VelocityCost(residual, robot_.getVelocityWeight(), getNormParams(norm_params_, "velocity"));
        }
        
        // Upright cost
        if (robot_.getUprightWeight() > 0.0) {
            Eigen::Vector3d residual = computeTorsoZAxis(x_traj[t]) - Eigen::Vector3d(0, 0, 1);
            total_cost += ilqr::uprightCost(residual, robot_.getUprightWeight(), getNormParams(norm_params_, "upright"));
        }
        
        // Balance cost: S*(CP - PCP), DeepMind formula
        if (robot_.getBalanceWeight() > 0.0) {
            bool left_stance = robot_.isStance(0, t);
            bool right_stance = robot_.isStance(1, t);
            
            if (left_stance || right_stance) {
                double S = computeSfactor(x_traj[t](2));
                if (S > 0.0) {
                    Eigen::Vector3d p_com = robot_.computeCoM(x_traj[t]);
                    Eigen::Vector2d p_cp = computeCapturePoint(x_traj[t], p_com, robot_.nq());
                    // Single-stance: both segment endpoints collapse to the stance foot
                    Eigen::Vector2d foot_l = left_stance  ? robot_.getEEReference(t, 0).head<2>()
                                                          : robot_.getEEReference(t, 1).head<2>();
                    Eigen::Vector2d foot_r = right_stance ? robot_.getEEReference(t, 1).head<2>()
                                                          : robot_.getEEReference(t, 0).head<2>();
                    Eigen::Vector2d pcp = computePCP(p_cp, foot_l, foot_r);
                    Eigen::Vector2d residual = S * (p_cp - pcp);
                    total_cost += ilqr::balanceCost(residual, robot_.getBalanceWeight(), getNormParams(norm_params_, "balance"));
                }
            }
        }
        // Pelvis/Feet cost: 0.5*(z_foot_r+z_foot_l) - z_pelvis - 0.2 (RectifyLoss, one-sided)
        if (robot_.getPelvisFeetWeight() > 0.0) {
            double z_pelvis = robot_.computeBodyZPos(x_traj[t], robot_.getPelvisBodyName());
            double z_foot_r = robot_.computeBodyZPos(x_traj[t], robot_.getRightFootBodyName());
            double z_foot_l = robot_.computeBodyZPos(x_traj[t], robot_.getLeftFootBodyName());
            double residual = 0.5*(z_foot_r + z_foot_l) - z_pelvis - 0.2;
            total_cost += ilqr::PelvisFeetCost(residual, robot_.getPelvisFeetWeight(), getNormParams(norm_params_, "pelvis_feet"));
        }
    }

    // Terminal posture cost (joint angles [7:nq], Quadratic)
    Eigen::VectorXd x_err_N = x_traj[N_] - x_ref[N_];
    total_cost += ilqr::PostureCost(x_err_N, robot_.Qf());
    
    // Terminal height (torso z) cost
    if (robot_.getHeightWeight() > 0.0) {
        double residual = x_traj[N_](2) - height_ref_[N_](2);
        total_cost += ilqr::HeightCost(residual, robot_.getHeightWeight(), getNormParams(norm_params_, "height"));
    }
    
    // Terminal velocity cost: 2D world-frame base xy velocity, zero target
    if (robot_.getVelocityWeight() > 0.0) {
        Eigen::Vector2d residual = x_traj[N_].segment(robot_.nq(), 2);
        total_cost += ilqr::VelocityCost(residual, robot_.getVelocityWeight(), getNormParams(norm_params_, "velocity"));
    }
    
    // Terminal upright cost
    if (robot_.getUprightWeight() > 0.0) {
        Eigen::Vector3d residual = computeTorsoZAxis(x_traj[N_]) - Eigen::Vector3d(0, 0, 1);
        total_cost += ilqr::uprightCost(residual, robot_.getUprightWeight(), getNormParams(norm_params_, "upright"));
    }
    
    // Terminal balance cost: S*(CP - PCP)
    if (robot_.getBalanceWeight() > 0.0) {
        bool left_stance = robot_.isStance(0, N_);
        bool right_stance = robot_.isStance(1, N_);
        
        if (left_stance || right_stance) {
            double S = computeSfactor(x_traj[N_](2));
            if (S > 0.0) {
                Eigen::Vector3d p_com = robot_.computeCoM(x_traj[N_]);
                Eigen::Vector2d p_cp = computeCapturePoint(x_traj[N_], p_com, robot_.nq());
                Eigen::Vector2d foot_l = left_stance  ? robot_.getEEReference(N_, 0).head<2>()
                                                      : robot_.getEEReference(N_, 1).head<2>();
                Eigen::Vector2d foot_r = right_stance ? robot_.getEEReference(N_, 1).head<2>()
                                                      : robot_.getEEReference(N_, 0).head<2>();
                Eigen::Vector2d pcp = computePCP(p_cp, foot_l, foot_r);
                Eigen::Vector2d residual = S * (p_cp - pcp);
                total_cost += ilqr::balanceCost(residual, robot_.getBalanceWeight(), getNormParams(norm_params_, "balance"));
            }
        }
    }

    // Terminal Pelvis/Feet cost
    if (robot_.getPelvisFeetWeight() > 0.0) {
        double z_pelvis = robot_.computeBodyZPos(x_traj[N_], robot_.getPelvisBodyName());
        double z_foot_r = robot_.computeBodyZPos(x_traj[N_], robot_.getRightFootBodyName());
        double z_foot_l = robot_.computeBodyZPos(x_traj[N_], robot_.getLeftFootBodyName());
        double residual = 0.5*(z_foot_r + z_foot_l) - z_pelvis - 0.2;
        total_cost += ilqr::PelvisFeetCost(residual, robot_.getPelvisFeetWeight(), getNormParams(norm_params_, "pelvis_feet"));
    }
    
    // Constraint costs
    for(int t = 0; t < N_; ++t){
        total_cost += robot_.constraintCost(x_traj[t], u_traj[t]);
    }
    total_cost += robot_.constraintCost(x_traj[N_], Eigen::VectorXd::Zero(robot_.nu()));

    return total_cost;
}

// Multi-iteration solve function (main interface)
bool iLQR::solve(const Eigen::VectorXd& x0,
                 const std::vector<Eigen::VectorXd>& x_ref,
                 const std::vector<Eigen::VectorXd>& u_ref,
                 const std::vector<Eigen::Vector3d>& height_ref,
                 double& cost_out) {
    if (x_ref.size() != (size_t)(N_ + 1) || u_ref.size() != (size_t)N_ || height_ref.size() != (size_t)(N_ + 1)) {
        std::cerr << "Reference size mismatch: x_ref=" << x_ref.size()
                  << " expected=" << N_ + 1 << ", u_ref=" << u_ref.size()
                  << " expected=" << N_ << ", height_ref=" << height_ref.size()
                  << " expected=" << N_ + 1 << std::endl;
        return false;
    }
    
    // Store height reference
    height_ref_ = height_ref;

#ifdef ENABLE_PROFILING
    auto t_cost_start = std::chrono::steady_clock::now();
#endif
    double current_cost = computeTotalCost(xbar_, ubar_, x_ref, u_ref);
#ifdef ENABLE_PROFILING
    auto t_cost_end = std::chrono::steady_clock::now();
    prof_data["iLQR_computeCost"].times.push_back(
        std::chrono::duration<double, std::milli>(t_cost_end - t_cost_start).count());
#endif

    for (int iter = 0; iter < max_iterations_; ++iter) {
        double previous_cost = current_cost;
        try {
            // Set current initial state
            xbar_[0] = x0;

            // Forward rollout using current nominal controls
#ifdef ENABLE_PROFILING
            {
                auto prof_start = std::chrono::steady_clock::now();
                forwardRolloutNominal();
                auto prof_end = std::chrono::steady_clock::now();
                prof_data["iLQR_forwardRollout"].times.push_back(
                    std::chrono::duration<double, std::milli>(prof_end - prof_start).count());
            }
#else
            forwardRolloutNominal();
#endif

            // Linearize dynamics & cost
#ifdef ENABLE_PROFILING
            {
                auto prof_start = std::chrono::steady_clock::now();
                computeLinearization();
                auto prof_end = std::chrono::steady_clock::now();
                prof_data["iLQR_linearization"].times.push_back(
                    std::chrono::duration<double, std::milli>(prof_end - prof_start).count());
            }
#else
            computeLinearization();
#endif

#ifdef ENABLE_PROFILING
            {
                auto prof_start = std::chrono::steady_clock::now();
                computeCostQuadratics(x_ref, u_ref);
                auto prof_end = std::chrono::steady_clock::now();
                prof_data["iLQR_costQuadratics"].times.push_back(
                    std::chrono::duration<double, std::milli>(prof_end - prof_start).count());
            }
#else
            computeCostQuadratics(x_ref, u_ref);
#endif

            // Backward pass for gains
#ifdef ENABLE_PROFILING
            {
                auto prof_start = std::chrono::steady_clock::now();
                backwardPass();
                auto prof_end = std::chrono::steady_clock::now();
                prof_data["iLQR_backwardPass"].times.push_back(
                    std::chrono::duration<double, std::milli>(prof_end - prof_start).count());
            }
#else
            backwardPass();
#endif

            // Forward line search to improve trajectory
            double new_cost;
            bool improved;
#ifdef ENABLE_PROFILING
            {
                auto prof_start = std::chrono::steady_clock::now();
                improved = forwardPassLineSearch(x0, x_ref, u_ref, new_cost);
                auto prof_end = std::chrono::steady_clock::now();
                prof_data["iLQR_lineSearch"].times.push_back(
                    std::chrono::duration<double, std::milli>(prof_end - prof_start).count());
            }
#else
            improved = forwardPassLineSearch(x0, x_ref, u_ref, new_cost);
#endif
            
            if (!improved) {
                // Experiment 1: Lambda scaling - increased from 1e-3 to 100.0
                reg_lambda_ = std::min(reg_lambda_ * 10.0, 100.0);
                if (reg_lambda_ >= 100.0) {
                    break;  // Maxed out regularization, give up
                }
#ifdef ENABLE_PROFILING
                {
                    auto prof_start = std::chrono::steady_clock::now();
                    backwardPass();
                    auto prof_end = std::chrono::steady_clock::now();
                    prof_data["iLQR_backwardPass"].times.push_back(
                        std::chrono::duration<double, std::milli>(prof_end - prof_start).count());
                }
                {
                    auto prof_start = std::chrono::steady_clock::now();
                    improved = forwardPassLineSearch(x0, x_ref, u_ref, new_cost);
                    auto prof_end = std::chrono::steady_clock::now();
                    prof_data["iLQR_lineSearch"].times.push_back(
                        std::chrono::duration<double, std::milli>(prof_end - prof_start).count());
                }
#else
                backwardPass();
                improved = forwardPassLineSearch(x0, x_ref, u_ref, new_cost);
#endif
                if (!improved) {
                    if (iter > 1) break; // give up after a couple failed attempts
                    continue;
                }
            } else {
                // SUCCESS: Ratio-based lambda adaptation (Trust Region)
                double actual_red = previous_cost - new_cost;
                double expected_red = -(dV_(0) + dV_(1));  // From backward pass
                
                // Only adapt if expected reduction is significant
                if (std::abs(expected_red) > 1e-10) {
                    double ratio = actual_red / expected_red;
                    
                    // Trust region: adapt lambda based on model quality
                    if (ratio > trust_region_good_) {
                        // Model is GOOD → decrease regularization (faster Newton)
                        reg_lambda_ = std::max(reg_lambda_ / reg_decrease_factor_, reg_min_);
                    } else if (ratio < trust_region_poor_) {
                        // Model is POOR → increase regularization (safer gradient)
                        reg_lambda_ = std::min(reg_lambda_ * reg_increase_factor_, reg_max_);
                    }
                    // else: trust_region_poor <= ratio <= trust_region_good → keep lambda unchanged
                } else {
                    // Fallback:  expected reduction too small, decrease lambda
                    reg_lambda_ = std::max(reg_lambda_ / 2.0, reg_min_);
                }
            }
            current_cost = new_cost;
        } catch (const std::exception& e) {
            std::cerr << "iLQR solve exception: " << e.what() << std::endl;
            break;
        }


        // Experiment 4: Relative convergence (scale-independent)
        double delta = std::abs(current_cost - previous_cost);
        double relative_delta = delta / std::max(1.0, std::abs(previous_cost));
        
        if (relative_delta < tolerance_ || delta < convergence_threshold_) {
            break;  // Converged
        }
        if (current_cost > 1e6) break;
    }

    cost_out = current_cost;
    return true;
}

void iLQR::addHeightCostDerivatives(int t, double goal_z) {
    const double w_height = robot_.getHeightWeight();
    
    // Use symbolic derivatives (fast and exact!)
    Eigen::VectorXd grad_height = derivatives_.HeightGrad(xbar_[t], goal_z, w_height);
    Eigen::MatrixXd hess_height = derivatives_.HeightHess(xbar_[t], goal_z, w_height);
    
    // Add to cost quadratics
    lx_[t] += grad_height;
    lxx_[t] += hess_height;
}

// Velocity Cost Derivatives — 2D world-frame base xy velocity, zero target (DeepMind "Velocity")
void iLQR::addVelocityCostDerivatives(int t) {
    const double w_vel = robot_.getVelocityWeight();
    
    // Skip if weight is zero (disabled)
    if (w_vel <= 0.0) return;
    
    try {
        // No reference needed — residual is raw base v_xy
        Eigen::VectorXd grad_vel = derivatives_.VelocityGrad(xbar_[t], w_vel);
        Eigen::MatrixXd hess_vel = derivatives_.VelocityHess(xbar_[t], w_vel);
        
        // Add to cost quadratics
        lx_[t] += grad_vel;
        lxx_[t] += hess_vel;
        
    } catch (const std::exception& e) {
        std::cerr << "Warning: Velocity cost error at t=" << t << ": " << e.what() << std::endl;
    }
}

void iLQR::addUprightCostDerivatives(int t) {
    const double w_upright = robot_.getUprightWeight();
    if (w_upright <= 0.0) return;
    
    // Compute upright cost derivatives
    Eigen::VectorXd grad_upright = derivatives_.UprightGrad(xbar_[t], w_upright);
    Eigen::MatrixXd hess_upright = derivatives_.UprightHess(xbar_[t], w_upright);
    
    // Add to cost quadratics
    lx_[t] += grad_upright;
    lxx_[t] += hess_upright;
}

void iLQR::addBalanceCostDerivatives(int t) {
    const double w_balance = robot_.getBalanceWeight();
    if (w_balance <= 0.0) return;
    
    bool left_stance  = robot_.isStance(0, t);  // ee_idx=0 is left foot
    bool right_stance = robot_.isStance(1, t);  // ee_idx=1 is right foot
    if (!left_stance && !right_stance) return;  // aerial phase
    
    // Standing factor S: gates balance cost (DeepMind walk.cc formula)
    double S = computeSfactor(xbar_[t](2));     // x[2] = torso z in MuJoCo state
    if (S <= 0.0) return;                       // crouching — zero contribution
    
    // Capture point: com_xy + 0.3 * base_vel_xy (world frame, MuJoCo x[nq+0/1])
    Eigen::Vector3d p_com = robot_.computeCoM(xbar_[t]);
    Eigen::Vector2d p_cp = computeCapturePoint(xbar_[t], p_com, robot_.nq());
    
    // Projected capture point onto foot segment (single-stance: both endpoints = stance foot)
    Eigen::Vector2d foot_l = left_stance  ? robot_.getEEReference(t, 0).head<2>()
                                          : robot_.getEEReference(t, 1).head<2>();
    Eigen::Vector2d foot_r = right_stance ? robot_.getEEReference(t, 1).head<2>()
                                          : robot_.getEEReference(t, 0).head<2>();
    Eigen::Vector2d pcp = computePCP(p_cp, foot_l, foot_r);
    
    // Effective weight absorbs S^2 since residual = S*(CP-PCP) and norm is applied to residual
    // d/dx[w * rho(S*r)]  ≈  d/dx[(S^2 * w) * rho(r)]  at the linearization point
    double effective_weight = w_balance * S * S;
    
    Eigen::VectorXd grad_balance = derivatives_.BalanceGrad(xbar_[t], pcp, effective_weight);
    Eigen::MatrixXd hess_balance = derivatives_.BalanceHess(xbar_[t], pcp, effective_weight);
    
    lx_[t] += grad_balance;
    lxx_[t] += hess_balance;
}

void iLQR::addPelvisFeetCostDerivatives(int t) {
    const double w = robot_.getPelvisFeetWeight();
    if (w <= 0.0) return;
    Eigen::VectorXd grad = derivatives_.PelvisFeetGrad(xbar_[t], w);
    Eigen::MatrixXd hess = derivatives_.PelvisFeetHess(xbar_[t], w);
    lx_[t] += grad;
    lxx_[t] += hess;
}