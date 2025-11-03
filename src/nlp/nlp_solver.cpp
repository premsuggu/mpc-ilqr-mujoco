#include "nlp/nlp_solver.hpp"
#include "nlp/sym_utils.hpp"
#include <iostream>
#include <chrono>
#include <dlfcn.h>

namespace nlp {

NLPSolver::NLPSolver(const NLPConfig& config, const pinocchio::Model& model)
    : config_(config), model_(model)
{
    nq_ = model_.nq;
    nv_ = model_.nv;
    nu_ = nv_ - 6;
    n_ee_ = config_.ee_names.size();
    
    // Create symbolic utilities (handles ALL symbolic math)
    sym_ = new SymUtils(model_, config_.ee_names);
    
    std::cout << "[NLPSolver] Initialized (N=" << config_.N << ", dt=" << config_.dt << " s)" << std::endl;
}

NLPSolver::~NLPSolver() {
    delete sym_;
}

void NLPSolver::createSymbolicVariables() {
    a_sym_.resize(config_.N);
    f_sym_.resize(config_.N);
    q_sym_.resize(config_.N + 1);
    v_sym_.resize(config_.N + 1);
    
    for (int k = 0; k < config_.N; ++k) {
        a_sym_[k] = casadi::SX::sym("a_" + std::to_string(k), nv_);
        f_sym_[k] = casadi::SX::sym("f_" + std::to_string(k), 6 * n_ee_);
    }
    
    q0_param_ = casadi::SX::sym("q0", nq_);
    v0_param_ = casadi::SX::sym("v0", nv_);
    
    q_ref_param_.resize(config_.N + 1);
    v_ref_param_.resize(config_.N + 1);
    com_ref_param_.resize(config_.N + 1);
    ee_pos_ref_param_.resize(config_.N + 1);
    
    for (int k = 0; k <= config_.N; ++k) {
        q_ref_param_[k] = casadi::SX::sym("q_ref_" + std::to_string(k), nq_);
        v_ref_param_[k] = casadi::SX::sym("v_ref_" + std::to_string(k), nv_);
        com_ref_param_[k] = casadi::SX::sym("com_ref_" + std::to_string(k), 3);
        
        ee_pos_ref_param_[k].resize(n_ee_);
        for (int ee = 0; ee < n_ee_; ++ee) {
            ee_pos_ref_param_[k][ee] = casadi::SX::sym("ee_pos_ref_" + std::to_string(k) + 
                                                       "_" + std::to_string(ee), 3);
        }
    }
}

void NLPSolver::buildIntegrationChain() {
    q_sym_[0] = q0_param_;
    v_sym_[0] = v0_param_;
    
    const casadi::Function& integrate_fn = sym_->getIntegrateFunction();
    
    for (int k = 0; k < config_.N; ++k) {
        v_sym_[k+1] = v_sym_[k] + config_.dt * a_sym_[k];
        q_sym_[k+1] = integrate_fn(casadi::SXVector{q_sym_[k], v_sym_[k+1], casadi::SX(config_.dt)})[0];
    }
}

casadi::SX NLPSolver::buildCost(const ContactSchedule& contacts) {
    casadi::SX cost = 0;
    
    for (int k = 0; k < config_.N; ++k) {
        cost += sym_->computeStageCost(
            q_sym_[k], v_sym_[k], a_sym_[k], f_sym_[k],
            q_ref_param_[k], v_ref_param_[k], com_ref_param_[k],
            ee_pos_ref_param_[k], contacts[k], config_.weights);
    }
    
    cost += sym_->computeTerminalCost(
        q_sym_[config_.N], v_sym_[config_.N],
        q_ref_param_[config_.N], v_ref_param_[config_.N], config_.weights);
    
    return cost;
}

casadi::SX NLPSolver::buildConstraints(const ContactSchedule& contacts) {
    std::vector<casadi::SX> all_constraints;
    
    for (int k = 0; k < config_.N; ++k) {
        // Compute torques: τ = M*a + C + g - J^T*f
        casadi::SX tau = sym_->computeTorques(q_sym_[k], v_sym_[k], a_sym_[k], 
                                              f_sym_[k], config_.ee_names);
        
        // Floating base constraints: τ_fb = 0
        casadi::SX tau_fb = tau(casadi::Slice(0, 6));
        for (int i = 0; i < 6; ++i) {
            all_constraints.push_back(tau_fb(i));
        }
        
        // Joint torque limits: τ_min ≤ τ_joints ≤ τ_max
        casadi::SX tau_joints = tau(casadi::Slice(6, nv_));
        for (int i = 0; i < (nv_ - 6); ++i) {
            all_constraints.push_back(tau_joints(i));
        }
    }
    
    if (all_constraints.empty()) {
        return casadi::SX::zeros(0, 1);
    }
    
    return casadi::SX::vertcat(all_constraints);
}

void NLPSolver::setupBounds(const ContactSchedule& contacts) {
    int n_accel_vars = config_.N * nv_;
    int n_force_vars = config_.N * (6 * n_ee_);
    int total_vars = n_accel_vars + n_force_vars;
    
    lbx_ = casadi::DM::zeros(total_vars, 1);
    ubx_ = casadi::DM::zeros(total_vars, 1);
    
    // Acceleration bounds
    for (int i = 0; i < n_accel_vars; ++i) {
        lbx_(i) = -50.0;
        ubx_(i) = 50.0;
    }
    
    // Force bounds (contact-aware)
    for (int k = 0; k < config_.N; ++k) {
        for (int ee = 0; ee < n_ee_; ++ee) {
            int f_start = n_accel_vars + k * (6 * n_ee_) + ee * 6;
            
            bool is_stance = contacts[k][ee];
            
            if (is_stance) {
                lbx_(f_start + 0) = -500.0; ubx_(f_start + 0) = 500.0;
                lbx_(f_start + 1) = -500.0; ubx_(f_start + 1) = 500.0;
                lbx_(f_start + 2) = 0.0;    ubx_(f_start + 2) = 1000.0;
                lbx_(f_start + 3) = -250.0; ubx_(f_start + 3) = 250.0;
                lbx_(f_start + 4) = -250.0; ubx_(f_start + 4) = 250.0;
                lbx_(f_start + 5) = -250.0; ubx_(f_start + 5) = 250.0;
            } else {
                for (int j = 0; j < 6; ++j) {
                    lbx_(f_start + j) = 0.0;
                    ubx_(f_start + j) = 0.0;
                }
            }
        }
    }
}

void NLPSolver::setupSolver(const casadi::SX& cost, const casadi::SX& constraints) {
    // Decision variables: W = [a_0...a_{N-1}; f_0...f_{N-1}]
    std::vector<casadi::SX> opt_vars;
    for (int k = 0; k < config_.N; ++k) {
        opt_vars.push_back(a_sym_[k]);
    }
    for (int k = 0; k < config_.N; ++k) {
        opt_vars.push_back(f_sym_[k]);
    }
    casadi::SX W = casadi::SX::vertcat(opt_vars);
    
    // Parameters: p = [q0, v0, q_ref, v_ref, com_ref, ee_pos_ref]
    std::vector<casadi::SX> params;
    params.push_back(q0_param_);
    params.push_back(v0_param_);
    
    for (int k = 0; k <= config_.N; ++k) {
        params.push_back(q_ref_param_[k]);
        params.push_back(v_ref_param_[k]);
        params.push_back(com_ref_param_[k]);
        for (int ee = 0; ee < n_ee_; ++ee) {
            params.push_back(ee_pos_ref_param_[k][ee]);
        }
    }
    
    casadi::SX p = casadi::SX::vertcat(params);
    
    // Constraint bounds
    int n_constraints = constraints.size1();
    lbg_ = casadi::DM::zeros(n_constraints, 1);
    ubg_ = casadi::DM::zeros(n_constraints, 1);
    
    int idx = 0;
    
    // Floating base: τ_fb = 0
    for (int k = 0; k < config_.N; ++k) {
        for (int i = 0; i < 6; ++i) {
            lbg_(idx) = 0.0;
            ubg_(idx) = 0.0;
            idx++;
        }
    }
    
    // Joint torque limits
    for (int k = 0; k < config_.N; ++k) {
        for (int i = 6; i < nv_; ++i) {
            double tau_max = 1000.0;
            int joint_idx = i - 6;
            
            if (joint_idx < model_.effortLimit.size()) {
                tau_max = model_.effortLimit(joint_idx);
            }
            
            lbg_(idx) = -tau_max;
            ubg_(idx) = tau_max;
            idx++;
        }
    }
    
    // Build NLP
    casadi::SXDict nlp;
    nlp["x"] = W;
    nlp["p"] = p;
    nlp["f"] = cost;
    nlp["g"] = constraints;
    
    // IPOPT options
    casadi::Dict opts;
    opts["print_time"] = false;
    opts["verbose"] = false;
    opts["ipopt.print_level"] = config_.solver_options.print_level;
    opts["ipopt.max_iter"] = config_.solver_options.max_iter;
    opts["ipopt.tol"] = config_.solver_options.tol;
    opts["ipopt.acceptable_tol"] = config_.solver_options.acceptable_tol;
    opts["ipopt.warm_start_init_point"] = "yes";
    opts["ipopt.mu_init"] = config_.solver_options.mu_init;
    
    // Try to use HSL if available
    if (config_.solver_options.use_hsl) {
        void* hsl_handle = dlopen("libcoinhsl.so", RTLD_LAZY);
        if (!hsl_handle) {
            hsl_handle = dlopen("libhsl.so", RTLD_LAZY);
        }
        
        if (hsl_handle) {
            opts["ipopt.linear_solver"] = config_.solver_options.linear_solver;
            opts["ipopt.ma57_print_level"] = 0;
            dlclose(hsl_handle);
        }
    }
    
    solver_ = casadi::nlpsol("nlp_solver", "ipopt", nlp, opts);
}

Eigen::VectorXd NLPSolver::solve(const Eigen::VectorXd& x0,
                                 const References& refs,
                                 const ContactSchedule& contacts,
                                 const Eigen::VectorXd& W_guess) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Build NLP on first solve
    static bool first_solve = true;
    if (first_solve) {
        std::cout << "[NLP] Building symbolic problem..." << std::endl;
        
        createSymbolicVariables();
        buildIntegrationChain();
        
        casadi::SX cost = buildCost(contacts);
        casadi::SX constraints = buildConstraints(contacts);
        
        setupBounds(contacts);
        setupSolver(cost, constraints);
        
        first_solve = false;
    }
    
    // Pack parameters
    Eigen::VectorXd q0 = x0.head(nq_);
    Eigen::VectorXd v0 = x0.tail(nv_);
    
    std::vector<double> p_vec;
    for (int i = 0; i < nq_; ++i) p_vec.push_back(q0(i));
    for (int i = 0; i < nv_; ++i) p_vec.push_back(v0(i));
    
    for (int k = 0; k <= config_.N; ++k) {
        for (int i = 0; i < nq_; ++i) p_vec.push_back(refs.q_ref[k](i));
        for (int i = 0; i < nv_; ++i) p_vec.push_back(refs.v_ref[k](i));
        for (int i = 0; i < 3; ++i) p_vec.push_back(refs.com_ref[k](i));
        for (int ee = 0; ee < n_ee_; ++ee) {
            for (int i = 0; i < 3; ++i) p_vec.push_back(refs.ee_pos_ref[k][ee](i));
        }
    }
    
    casadi::DM p_val = casadi::DM(p_vec);
    
    // Initial guess
    int total_vars = config_.N * nv_ + config_.N * (6 * n_ee_);
    casadi::DM x0_dm;
    
    if (W_guess.size() == total_vars) {
        std::vector<double> x0_vec(W_guess.data(), W_guess.data() + W_guess.size());
        x0_dm = casadi::DM(x0_vec);
    } else {
        x0_dm = casadi::DM::zeros(total_vars, 1);
    }
    
    // Solve
    casadi::DMDict arg;
    arg["x0"] = x0_dm;
    arg["p"] = p_val;
    arg["lbx"] = lbx_;
    arg["ubx"] = ubx_;
    arg["lbg"] = lbg_;
    arg["ubg"] = ubg_;
    
    casadi::DMDict res;
    try {
        res = solver_(arg);
    } catch (const std::exception& e) {
        std::cerr << "[NLP] Solver failed: " << e.what() << std::endl;
        return Eigen::VectorXd::Zero(total_vars);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double solve_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Extract solution
    casadi::DM W_sol = res["x"];
    Eigen::VectorXd W_eigen(total_vars);
    for (int i = 0; i < total_vars; ++i) {
        W_eigen(i) = double(W_sol(i));
    }
    
    std::cout << "[NLP] Solved in " << solve_time_ms << " ms" << std::endl;
    
    return W_eigen;
}

Eigen::VectorXd NLPSolver::extractFirstControl(const Eigen::VectorXd& W_sol) {
    // For contact-implicit: control is contact forces
    // Currently not used - return zeros
    Eigen::VectorXd u0 = Eigen::VectorXd::Zero(nu_);
    return u0;
}

Eigen::VectorXd NLPSolver::warmStart(const Eigen::VectorXd& W_sol) {
    int n_accel_vars = config_.N * nv_;
    int n_force_vars = config_.N * (6 * n_ee_);
    int total_vars = n_accel_vars + n_force_vars;
    
    Eigen::VectorXd W_guess = Eigen::VectorXd::Zero(total_vars);
    
    // Shift accelerations
    for (int k = 0; k < config_.N - 1; ++k) {
        W_guess.segment(k * nv_, nv_) = W_sol.segment((k + 1) * nv_, nv_);
    }
    W_guess.segment((config_.N - 1) * nv_, nv_) = W_sol.segment((config_.N - 1) * nv_, nv_);
    
    // Shift forces
    for (int k = 0; k < config_.N - 1; ++k) {
        int src = n_accel_vars + (k + 1) * (6 * n_ee_);
        int dst = n_accel_vars + k * (6 * n_ee_);
        W_guess.segment(dst, 6 * n_ee_) = W_sol.segment(src, 6 * n_ee_);
    }
    int last_src = n_accel_vars + (config_.N - 1) * (6 * n_ee_);
    int last_dst = n_accel_vars + (config_.N - 1) * (6 * n_ee_);
    W_guess.segment(last_dst, 6 * n_ee_) = W_sol.segment(last_src, 6 * n_ee_);
    
    return W_guess;
}

} // namespace nlp
