#include "nlp/mpc_utils.hpp"
#include "nlp/nlp_solver.hpp"
#include "nlp/nlp_config.hpp"
#include "nlp/nlp_utils.hpp"
#include <iostream>
#include <chrono>

namespace nlp {

MPCUtils::MPCUtils(const NLPConfig& config,
                   const pinocchio::Model& model,
                   const References& refs,
                   NLPSolver& solver)
    : config_(config), model_(model), refs_(refs), solver_(solver),
      t_idx_(0), has_warm_start_(false)
{
    std::cout << "[MPCUtils] Initialized with N=" << config_.N 
              << ", dt=" << config_.dt << " s" << std::endl;
}

MPCResults MPCUtils::run() {
    std::cout << "========== STARTING MPC LOOP ==========\n" << std::endl;
    
    MPCResults results;
    results.success = false;
    results.num_steps_completed = 0;
    
    // Initial state from reference
    Eigen::VectorXd x_current(model_.nq + model_.nv);
    x_current.head(model_.nq) = refs_.q_ref[0];
    x_current.tail(model_.nv) = refs_.v_ref[0];
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < config_.num_mpc_steps; ++step) {
        std::cout << "[STEP " << step << "] t = " << step * config_.dt << " s" << std::endl;
        
        setTimeIndex(step);
        
        // Extract references for this horizon
        References refs_window = extractReferenceWindow();
        ContactSchedule contacts = createContactSchedule();
        
        // Solve NLP
        auto solve_start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd W_sol = solver_.solve(x_current, refs_window, contacts, W_warm_);
        auto solve_end = std::chrono::high_resolution_clock::now();
        
        double solve_time_ms = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
        
        if (W_sol.norm() < 1e-10) {
            std::cerr << "[STEP " << step << "] Solver failed! Stopping." << std::endl;
            break;
        }
        
        std::cout << "  Solve time: " << solve_time_ms << " ms" << std::endl;
        
        // Extract first control
        Eigen::VectorXd u_opt = solver_.extractFirstControl(W_sol);
        Eigen::VectorXd a_opt = W_sol.head(model_.nv);
        
        // Integrate state forward
        x_current = integrateState(x_current, a_opt);
        
        // Store results
        results.q_trajectory.push_back(x_current.head(model_.nq));
        results.v_trajectory.push_back(x_current.tail(model_.nv));
        results.u_trajectory.push_back(u_opt);
        
        // Warm start for next iteration
        W_warm_ = solver_.warmStart(W_sol);
        has_warm_start_ = true;
        
        results.num_steps_completed++;
        
        std::cout << "  State norm: " << x_current.norm() << "\n" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    results.total_time_s = std::chrono::duration<double>(total_end - total_start).count();
    results.success = (results.num_steps_completed == config_.num_mpc_steps);
    
    std::cout << "========== MPC LOOP COMPLETE ==========\n" << std::endl;
    std::cout << "Total time: " << results.total_time_s << " s" << std::endl;
    std::cout << "Average time per step: " << (results.total_time_s / results.num_steps_completed) * 1000.0 << " ms\n" << std::endl;
    
    return results;
}

bool MPCUtils::stepOnce(const Eigen::VectorXd& x_current, Eigen::VectorXd& u_apply) {
    // Extract references
    References refs_window = extractReferenceWindow();
    ContactSchedule contacts = createContactSchedule();
    
    // Solve
    Eigen::VectorXd W_sol = solver_.solve(x_current, refs_window, contacts, W_warm_);
    
    if (W_sol.norm() < 1e-10) {
        std::cerr << "[MPCUtils] Solver failed at t=" << t_idx_ << std::endl;
        return false;
    }
    
    // Extract control
    u_apply = solver_.extractFirstControl(W_sol);
    
    // Update warm start
    W_warm_ = solver_.warmStart(W_sol);
    has_warm_start_ = true;
    
    // Advance time
    t_idx_++;
    
    return true;
}

void MPCUtils::reset() {
    t_idx_ = 0;
    has_warm_start_ = false;
    W_warm_ = Eigen::VectorXd();
}

References MPCUtils::extractReferenceWindow() {
    References window;
    window.q_ref.clear();
    window.v_ref.clear();
    window.com_ref.clear();
    window.ee_pos_ref.clear();
    
    for (int k = 0; k <= config_.N; ++k) {
        int idx = std::min(t_idx_ + k, (int)refs_.q_ref.size() - 1);
        
        window.q_ref.push_back(refs_.q_ref[idx]);
        window.v_ref.push_back(refs_.v_ref[idx]);
        window.com_ref.push_back(refs_.com_ref[idx]);
        window.ee_pos_ref.push_back(refs_.ee_pos_ref[idx]);
    }
    
    return window;
}

ContactSchedule MPCUtils::createContactSchedule() {
    ContactSchedule contacts;
    int n_ee = config_.ee_names.size();
    
    for (int k = 0; k < config_.N; ++k) {
        std::vector<bool> contact_k(n_ee, true);  // All stance for now
        contacts.push_back(contact_k);
    }
    
    return contacts;
}

Eigen::VectorXd MPCUtils::integrateState(const Eigen::VectorXd& x_current,
                                          const Eigen::VectorXd& a) {
    return nlp::integrateState(model_, x_current, a, config_.dt);
}

} // namespace nlp
