#include "ilqr.hpp"
#include <iostream>
#include <chrono>

iLQR::iLQR(RobotUtils& robot, int N, double dt) 
        : robot_(robot), N_(N), dt_(dt), reg_lambda_(1e-6),
            max_iterations_(10), tolerance_(1e-4) {
    
    // Pre-allocate all trajectories and matrices
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
                                  const std::vector<Eigen::VectorXd>* prev_xbar,
                                  const std::vector<Eigen::VectorXd>* prev_ubar) {
    
    xbar_[0] = x0;
    
    // Use warm start if available (same as before)
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
        // REFERENCE-AWARE COLD START: Much better initial guess using actual reference
        
        for (int t = 0; t < N_; ++t) {
            if (robot_.nu() == 1 && robot_.nx() >= 1) {
                // Compute tracking error relative to actual reference
                double angle_error = x0(0) - x_ref[t](0);
                double velocity_error = (robot_.nx() > 1) ? x0(1) - x_ref[t](1) : 0.0;
                
                // Enhanced PD control with reference feedforward
                double kp = 25.0;  // Higher proportional gain
                double kd = 8.0;   // Higher derivative gain
                
                // Control = reference_feedforward + feedback_correction
                double feedforward = (t < u_ref.size()) ? u_ref[t](0) : 0.0;
                double feedback = -kp * angle_error - kd * velocity_error;
                
                // Combine feedforward and feedback with decay over horizon
                double decay = 1.0 - 0.3 * (double)t / N_;
                ubar_[t] = Eigen::VectorXd::Ones(robot_.nu()) * ((feedforward + feedback) * decay);
                
            } else {
                // For general systems, use reference control as starting point
                if (t < u_ref.size()) {
                    ubar_[t] = u_ref[t];
                } else {
                    ubar_[t] = Eigen::VectorXd::Zero(robot_.nu());
                }
            }
        }
        
        // Forward rollout to ensure consistent trajectory
        for (int t = 0; t < N_; ++t) {
            robot_.rolloutOneStep(xbar_[t], ubar_[t], xbar_[t + 1]);
        }
    }
}

void iLQR::forwardRolloutNominal() {
    // Roll out trajectory using current controls with numerical stability checks
    for (int t = 0; t < N_; ++t) {
        // Check for NaN/Inf in current state and control before rollout
        if (!xbar_[t].allFinite()) {
            std::cout << "WARNING: Non-finite state detected at timestep " << t 
                      << " before rollout. Clamping to finite values." << std::endl;
            // Clamp non-finite values to reasonable bounds
            for (int i = 0; i < xbar_[t].size(); ++i) {
                if (!std::isfinite(xbar_[t](i))) {
                    xbar_[t](i) = 0.0;  // Reset to zero for non-finite values
                }
            }
        }
        
        if (!ubar_[t].allFinite()) {
            std::cout << "WARNING: Non-finite control detected at timestep " << t 
                      << " before rollout. Clamping to zero." << std::endl;
            ubar_[t].setZero();
        }
        
        // Store state before rollout for diagnostics
        Eigen::VectorXd x_before = xbar_[t];
        
        robot_.rolloutOneStep(xbar_[t], ubar_[t], xbar_[t + 1]);
        
        // Check for NaN/Inf in resulting state after rollout
        if (!xbar_[t + 1].allFinite()) {
            std::cout << "WARNING: Non-finite state generated at timestep " << t + 1 
                      << " after rollout. This indicates physics instability." << std::endl;
            std::cout << "  State before: " << x_before.transpose().head(10) << "..." << std::endl;
            std::cout << "  Control: " << ubar_[t].transpose().head(5) << "..." << std::endl;
            std::cout << "  Problematic state: " << xbar_[t + 1].transpose().head(10) << "..." << std::endl;
            
            // Emergency fallback: use previous state with zero velocity
            xbar_[t + 1] = x_before;
            // Zero out velocities (second half of state vector)
            int nq = robot_.nq();
            for (int i = nq; i < xbar_[t + 1].size(); ++i) {
                xbar_[t + 1](i) = 0.0;
            }
        }
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
    
    // Check terminal conditions for numerical issues
    if (!VxN_.allFinite() || !VxxN_.allFinite()) {
        std::cout << "WARNING: Non-finite terminal cost derivatives detected. Using regularized values." << std::endl;
        VxN_.setZero();
        VxxN_ = Eigen::MatrixXd::Identity(robot_.nx(), robot_.nx()) * 1e-3;
    }
    
    // Backward recursion starts with terminal cost derivatives
    Eigen::VectorXd Vx = VxN_;
    Eigen::MatrixXd Vxx = VxxN_;
    
    for (int t = N_ - 1; t >= 0; --t) {
        // Check linearization matrices for numerical issues
        if (!A_[t].allFinite() || !B_[t].allFinite()) {
            std::cout << "WARNING: Non-finite linearization matrices at timestep " << t 
                      << ". Using identity/zero approximation." << std::endl;
            A_[t] = Eigen::MatrixXd::Identity(robot_.nx(), robot_.nx());
            B_[t] = Eigen::MatrixXd::Zero(robot_.nx(), robot_.nu());
        }
        
        // Check cost derivatives for numerical issues
        if (!lx_[t].allFinite() || !lu_[t].allFinite() || !lxx_[t].allFinite() || !luu_[t].allFinite()) {
            std::cout << "WARNING: Non-finite cost derivatives at timestep " << t 
                      << ". Using zero/identity approximation." << std::endl;
            lx_[t].setZero();
            lu_[t].setZero();
            lxx_[t] = Eigen::MatrixXd::Identity(robot_.nx(), robot_.nx()) * 1e-3;
            luu_[t] = Eigen::MatrixXd::Identity(robot_.nu(), robot_.nu()) * 1e-3;
        }
        
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
        
        // Check Q-function derivatives for numerical issues
        if (!Qx.allFinite() || !Qu.allFinite() || !Qxx.allFinite() || !Quu.allFinite() || !Qxu.allFinite()) {
            std::cout << "WARNING: Non-finite Q-function derivatives at timestep " << t 
                      << ". Applying emergency regularization." << std::endl;
            
            // Replace non-finite values with safe defaults
            if (!Qx.allFinite()) Qx.setZero();
            if (!Qu.allFinite()) Qu.setZero();
            if (!Qxx.allFinite()) Qxx = Eigen::MatrixXd::Identity(robot_.nx(), robot_.nx()) * 1e-3;
            if (!Quu.allFinite()) Quu = Eigen::MatrixXd::Identity(robot_.nu(), robot_.nu()) * 1e-3;
            if (!Qxu.allFinite()) Qxu.setZero();
        }
        
        // Regularization for numerical stability
        Quu += reg_lambda_ * Eigen::MatrixXd::Identity(Quu.rows(), Quu.cols());
        
        // Check positive definiteness of Quu with enhanced regularization
        Eigen::LLT<Eigen::MatrixXd> llt(Quu);
        if (llt.info() != Eigen::Success) {
            double extra_reg = 1e-4;
            Quu += extra_reg * Eigen::MatrixXd::Identity(Quu.rows(), Quu.cols());
            std::cout << "WARNING: Quu not positive definite at timestep " << t 
                      << ". Applied additional regularization: " << extra_reg << std::endl;
        }
        
        // Compute gains with enhanced numerical checks
        Eigen::LDLT<Eigen::MatrixXd> solver(Quu);
        if (solver.info() != Eigen::Success) {
            std::cout << "ERROR: Failed to decompose Quu at timestep " << t 
                      << ". Using zero gains." << std::endl;
            K_[t].setZero();
            kff_[t].setZero();
        } else {
            K_[t] = -solver.solve(Qxu.transpose());  // K = -Quu^{-1} Qux
            kff_[t] = -solver.solve(Qu);             // k = -Quu^{-1} Qu
        }
        
        // Check for numerical issues in computed gains
        if (!K_[t].allFinite() || !kff_[t].allFinite()) {
            std::cout << "WARNING: Non-finite gains computed at timestep " << t 
                      << ". Using zero gains as fallback." << std::endl;
            K_[t].setZero();
            kff_[t].setZero();
        }
        
        // Clamp gains to reasonable bounds to prevent extreme control actions
        double max_gain = 1000.0;  // Reasonable upper bound for gains
        K_[t] = K_[t].cwiseMax(-max_gain).cwiseMin(max_gain);
        kff_[t] = kff_[t].cwiseMax(-max_gain).cwiseMin(max_gain);
        
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
        
        // Ensure Vxx stays symmetric and check for numerical issues
        Vxx = 0.5 * (Vxx + Vxx.transpose());
        
        if (!Vx.allFinite() || !Vxx.allFinite()) {
            std::cout << "WARNING: Non-finite value function derivatives at timestep " << t 
                      << ". Using regularized values." << std::endl;
            if (!Vx.allFinite()) Vx.setZero();
            if (!Vxx.allFinite()) Vxx = Eigen::MatrixXd::Identity(robot_.nx(), robot_.nx()) * 1e-3;
        }
    }
}

bool iLQR::forwardPassLineSearch(const Eigen::VectorXd& x0,
                                const std::vector<Eigen::VectorXd>& x_ref,
                                const std::vector<Eigen::VectorXd>& u_ref,
                                double& new_cost) {
    
    // Compute baseline cost
    double baseline_cost = computeTotalCost(xbar_, ubar_, x_ref, u_ref);
    
    // Check if baseline cost is reasonable
    if (!std::isfinite(baseline_cost) || baseline_cost > 1e10) {
        std::cout << "WARNING: Baseline cost is not finite or too large: " << baseline_cost 
                  << ". Skipping line search." << std::endl;
        new_cost = baseline_cost;
        return false;
    }
    
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
            
            // Check for numerical issues in feedback computation
            if (!dx.allFinite()) {
                std::cout << "WARNING: Non-finite state error at timestep " << t 
                          << " with alpha " << alpha << ". Skipping this alpha." << std::endl;
                rollout_success = false;
                break;
            }
            
            u_new[t] = ubar_[t] + alpha * kff_[t] + K_[t] * dx;
            
            // Check for numerical issues and clamp control to reasonable bounds
            if (!u_new[t].allFinite()) {
                std::cout << "WARNING: Non-finite control computed at timestep " << t 
                          << " with alpha " << alpha << ". Clamping to baseline." << std::endl;
                u_new[t] = ubar_[t];
            }
            
            // Clamp control to actuator limits for stability
            for (int i = 0; i < robot_.nu(); ++i) {
                if (robot_.model() && i < robot_.model()->nu) {
                    double u_min = robot_.model()->actuator_ctrlrange[i * 2];
                    double u_max = robot_.model()->actuator_ctrlrange[i * 2 + 1];
                    if (std::isfinite(u_min) && std::isfinite(u_max)) {
                        u_new[t](i) = std::max(u_min, std::min(u_max, u_new[t](i)));
                    }
                }
                
                // Emergency clamp to prevent extreme values
                u_new[t](i) = std::max(-1000.0, std::min(1000.0, u_new[t](i)));
            }
            
            // Rollout one step with error handling
            try {
                robot_.rolloutOneStep(x_new[t], u_new[t], x_new[t + 1]);
                
                // Check resulting state for numerical issues
                if (!x_new[t + 1].allFinite()) {
                    std::cout << "WARNING: Non-finite state from rollout at timestep " << t 
                              << " with alpha " << alpha << ". Physics unstable." << std::endl;
                    rollout_success = false;
                    break;
                }
                
                // Check for extreme state values that indicate instability
                for (int i = 0; i < x_new[t + 1].size(); ++i) {
                    if (std::abs(x_new[t + 1](i)) > 1e6) {
                        std::cout << "WARNING: Extreme state value detected at timestep " << t 
                                  << " with alpha " << alpha << ". State[" << i << "] = " 
                                  << x_new[t + 1](i) << std::endl;
                        rollout_success = false;
                        break;
                    }
                }
                
            } catch (const std::exception& e) {
                std::cout << "WARNING: Exception during rollout at timestep " << t 
                          << " with alpha " << alpha << ": " << e.what() << std::endl;
                rollout_success = false;
                break;
            }
        }
        
        if (!rollout_success) continue;
        
        // Compute cost of new trajectory
        double cost = computeTotalCost(x_new, u_new, x_ref, u_ref);
        
        // Check if cost is reasonable
        if (!std::isfinite(cost) || cost > 1e10) {
            std::cout << "WARNING: Line search produced non-finite or extreme cost with alpha " 
                      << alpha << ": " << cost << std::endl;
            continue;
        }
        
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
                 double& cost_out) {
    if (x_ref.size() != (size_t)(N_ + 1) || u_ref.size() != (size_t)N_) {
        std::cerr << "Reference size mismatch: x_ref=" << x_ref.size()
                  << " expected=" << N_ + 1 << ", u_ref=" << u_ref.size()
                  << " expected=" << N_ << std::endl;
        return false;
    }

    // Validate initial state for numerical issues
    if (!x0.allFinite()) {
        std::cerr << "ERROR: Non-finite initial state passed to iLQR solve." << std::endl;
        cost_out = 1e10;
        return false;
    }

    double current_cost = computeTotalCost(xbar_, ubar_, x_ref, u_ref);
    
    // Check if initial cost is reasonable
    if (!std::isfinite(current_cost) || current_cost > 1e8) {
        std::cout << "WARNING: Initial cost is problematic: " << current_cost 
                  << ". Attempting trajectory reinitialization." << std::endl;
        
        // Reinitialize with reference-aware cold start
        initializeWithReference(x0, x_ref, u_ref);
        current_cost = computeTotalCost(xbar_, ubar_, x_ref, u_ref);
        
        if (!std::isfinite(current_cost) || current_cost > 1e8) {
            std::cerr << "ERROR: Cannot recover from problematic initial cost." << std::endl;
            cost_out = current_cost;
            return false;
        }
    }

    int consecutive_failures = 0;
    const int max_consecutive_failures = 3;

    for (int iter = 0; iter < max_iterations_; ++iter) {
        double previous_cost = current_cost;
        try {
            // Set current initial state
            xbar_[0] = x0;

            // Forward rollout using current nominal controls
            forwardRolloutNominal();

            // Check if rollout produced reasonable trajectory
            bool trajectory_valid = true;
            for (int t = 0; t <= N_; ++t) {
                if (!xbar_[t].allFinite()) {
                    std::cout << "WARNING: Non-finite state in trajectory at timestep " << t 
                              << " in iteration " << iter << std::endl;
                    trajectory_valid = false;
                    break;
                }
            }
            
            if (!trajectory_valid) {
                consecutive_failures++;
                if (consecutive_failures >= max_consecutive_failures) {
                    std::cout << "ERROR: Too many consecutive trajectory failures. Terminating solve." << std::endl;
                    break;
                }
                
                // Try with increased regularization
                reg_lambda_ = std::min(reg_lambda_ * 10.0, 1e-2);
                std::cout << "Increasing regularization to " << reg_lambda_ << " and retrying." << std::endl;
                continue;
            }

            // Linearize dynamics & cost
            computeLinearization();
            computeCostQuadratics(x_ref, u_ref);

            // Backward pass for gains
            backwardPass();

            // Forward line search to improve trajectory
            double new_cost;
            bool improved = forwardPassLineSearch(x0, x_ref, u_ref, new_cost);
            
            if (!improved) {
                consecutive_failures++;
                reg_lambda_ = std::min(reg_lambda_ * 10.0, 1e-2);
                std::cout << "Line search failed in iteration " << iter 
                          << ". Increasing regularization to " << reg_lambda_ << std::endl;
                
                backwardPass();
                improved = forwardPassLineSearch(x0, x_ref, u_ref, new_cost);
                
                if (!improved) {
                    if (consecutive_failures >= max_consecutive_failures) {
                        std::cout << "ERROR: Maximum consecutive failures reached. Terminating solve." << std::endl;
                        break;
                    }
                    continue;
                }
            }
            
            // Success - reset failure counter and update regularization
            consecutive_failures = 0;
            current_cost = new_cost;
            reg_lambda_ = std::max(reg_lambda_ / 2.0, 1e-6);
            
        } catch (const std::exception& e) {
            std::cerr << "iLQR solve exception in iteration " << iter << ": " << e.what() << std::endl;
            consecutive_failures++;
            if (consecutive_failures >= max_consecutive_failures) {
                break;
            }
            continue;
        }

        // Convergence / divergence checks
        double delta = std::abs(current_cost - previous_cost);
        if (delta < tolerance_) {
            std::cout << "iLQR converged in " << iter + 1 << " iterations. Final cost: " 
                      << current_cost << std::endl;
            break;
        }
        
        if (current_cost > 1e8) {
            std::cout << "ERROR: Cost diverged to " << current_cost << ". Terminating solve." << std::endl;
            break;
        }
        
        // Log progress for long solves
        if (iter > 5) {
            std::cout << "iLQR iteration " << iter << ": cost = " << current_cost 
                      << ", delta = " << delta << ", reg = " << reg_lambda_ << std::endl;
        }
    }

    cost_out = current_cost;
    
    // Final validation
    if (!std::isfinite(cost_out) || cost_out > 1e8) {
        std::cerr << "ERROR: iLQR solve completed with problematic final cost: " << cost_out << std::endl;
        return false;
    }
    
    return true;
}
