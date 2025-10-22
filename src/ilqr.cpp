#include "ilqr.hpp"
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
          N_(N), dt_(dt), reg_lambda_(1e-6), max_iterations_(10), tolerance_(1e-4) {
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
}

void iLQR::initializeWithReference(const Eigen::VectorXd& x0,
                                  const std::vector<Eigen::VectorXd>& x_ref,
                                  const std::vector<Eigen::VectorXd>& u_ref,
                                  const std::vector<Eigen::Vector3d>& com_ref,
                                  const std::vector<Eigen::VectorXd>* prev_xbar,
                                  const std::vector<Eigen::VectorXd>* prev_ubar) {
    
    // Store CoM reference
    com_ref_ = com_ref;
    
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
        
        // ADD CoM TRACKING DERIVATIVES if weight > 0
        if (robot_.getCoMWeight() > 0.0) {
            addCoMCostDerivatives(t, com_ref_[t]);
        }
        
        // ADD EE POSITION TRACKING DERIVATIVES if weight > 0
        if (robot_.getEEPosWeight() > 0.0) {
            addEEPosCostDerivatives(t);
        }
        
        // ADD EE VELOCITY TRACKING DERIVATIVES if weight > 0
        if (robot_.getEEVelWeight() > 0.0) {
            addEEVelCostDerivatives(t);
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
    
    // ADD TERMINAL CoM TRACKING DERIVATIVES if weight > 0
    if (robot_.getCoMWeight() > 0.0) {
        addCoMCostDerivatives(N_, com_ref_[N_]);
    }
    
    // ADD TERMINAL EE POSITION TRACKING DERIVATIVES if weight > 0
    if (robot_.getEEPosWeight() > 0.0) {
        addEEPosCostDerivatives(N_);
    }
    
    // ADD TERMINAL EE VELOCITY TRACKING DERIVATIVES if weight > 0
    if (robot_.getEEVelWeight() > 0.0) {
        addEEVelCostDerivatives(N_);
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

void iLQR::backwardPass() { 
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
        
        // Check for numerical issues
        if (!K_[t].allFinite() || !kff_[t].allFinite()) {
            std::cout << "Warning: Non-finite gains at timestep " << t << std::endl;
            // Continue with regularization instead of returning
        }
        
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
    
    // More aggressive line search parameters
    std::vector<double> alphas = {1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01};
    
    for (double alpha : alphas) {
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
        if (cost < baseline_cost - 1e-6) {
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

double iLQR::computeTotalCost(const std::vector<Eigen::VectorXd>& x_traj,
                             const std::vector<Eigen::VectorXd>& u_traj,
                             const std::vector<Eigen::VectorXd>& x_ref,
                             const std::vector<Eigen::VectorXd>& u_ref) {
    double total_cost = 0.0;
    
    // Running cost
    for (int t = 0; t < N_; ++t) {
        Eigen::VectorXd x_err = x_traj[t] - x_ref[t];
        Eigen::VectorXd u_err = u_traj[t] - u_ref[t];
        
        total_cost += 0.5 * x_err.transpose() * robot_.Q() * x_err;
        total_cost += 0.5 * u_err.transpose() * robot_.R() * u_err;
    }
    
    // Terminal cost
    Eigen::VectorXd x_err_N = x_traj[N_] - x_ref[N_];
    total_cost += 0.5 * x_err_N.transpose() * robot_.Qf() * x_err_N;

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
                 const std::vector<Eigen::Vector3d>& com_ref,
                 double& cost_out) {
    if (x_ref.size() != (size_t)(N_ + 1) || u_ref.size() != (size_t)N_ || com_ref.size() != (size_t)(N_ + 1)) {
        std::cerr << "Reference size mismatch: x_ref=" << x_ref.size()
                  << " expected=" << N_ + 1 << ", u_ref=" << u_ref.size()
                  << " expected=" << N_ << ", com_ref=" << com_ref.size()
                  << " expected=" << N_ + 1 << std::endl;
        return false;
    }
    
    // Store CoM reference
    com_ref_ = com_ref;

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
                reg_lambda_ = std::min(reg_lambda_ * 10.0, 1e-3);
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
            }
            current_cost = new_cost;
            reg_lambda_ = std::max(reg_lambda_ / 2.0, 1e-6);
        } catch (const std::exception& e) {
            std::cerr << "iLQR solve exception: " << e.what() << std::endl;
            break;
        }

        // Convergence / divergence checks
        double delta = std::abs(current_cost - previous_cost);
        if (delta < tolerance_) break;
        if (current_cost > 1e6) break;
    }

    cost_out = current_cost;
    return true;
}

void iLQR::addCoMCostDerivatives(int t, const Eigen::Vector3d& com_ref) {
    const double w_com = robot_.getCoMWeight();
    
    // Use symbolic derivatives (fast and exact!)
    Eigen::VectorXd grad_com = derivatives_.CoMGrad(xbar_[t], com_ref, w_com);
    Eigen::MatrixXd hess_com = derivatives_.CoMHess(xbar_[t], com_ref, w_com);
    
    // Add to cost quadratics
    lx_[t] += grad_com;
    lxx_[t] += hess_com;
}

void iLQR::addEEPosCostDerivatives(int t) {
    const double w_ee = robot_.getEEPosWeight();
    
    // Add derivatives for each end-effector
    for (int ee_idx = 0; ee_idx < 2; ++ee_idx) {  // left_ankle_link, right_ankle_link for H1
        // Skip position cost during stance phase (foot should stay planted)
        if (robot_.isStance(ee_idx, t)) continue;
        
        try {
            std::string frame_name = robot_.getEEFrameName(ee_idx);
            Eigen::Vector3d ee_ref = robot_.getEEReference(t, ee_idx);
            
            // Use symbolic derivatives
            Eigen::VectorXd grad_ee = derivatives_.EEposGrad(xbar_[t], ee_ref, frame_name, w_ee);
            Eigen::MatrixXd hess_ee = derivatives_.EEposHess(xbar_[t], ee_ref, frame_name, w_ee);
            
            // Add to cost quadratics
            lx_[t] += grad_ee;
            lxx_[t] += hess_ee;
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: EE cost error for idx " << ee_idx << ": " << e.what() << std::endl;
        }
    }
}

void iLQR::addEEVelCostDerivatives(int t) {
    const double w_ee_vel = robot_.getEEVelWeight();
    
    // Add derivatives for each end-effector
    for (int ee_idx = 0; ee_idx < 2; ++ee_idx) {  // left_ankle_link, right_ankle_link
        // Skip velocity cost during swing phase (foot needs to move)
        if (!robot_.isStance(ee_idx, t)) continue;
        
        try {
            std::string frame_name = robot_.getEEFrameName(ee_idx);
            // During stance, penalize velocity (target zero velocity to keep foot planted)
            Eigen::Vector3d ee_vel_ref = Eigen::Vector3d::Zero();
            
            // Use symbolic derivatives (fast and exact!)
            Eigen::VectorXd grad_ee_vel = derivatives_.EEvelGrad(xbar_[t], ee_vel_ref, frame_name, w_ee_vel);
            Eigen::MatrixXd hess_ee_vel = derivatives_.EEvelHess(xbar_[t], ee_vel_ref, frame_name, w_ee_vel);
            
            // Add to cost quadratics
            lx_[t] += grad_ee_vel;
            lxx_[t] += hess_ee_vel;
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: EE velocity cost error for idx " << ee_idx << ": " << e.what() << std::endl;
        }
    }
}