// src/mpc.cpp
#include "mpc.hpp"
#include <iostream>
#include <chrono>

MPC::MPC(RobotUtils& robot, int N, double dt) 
        : robot_(robot), ilqr_(robot, N, dt), N_(N), dt_(dt), 
            t_idx_(0), has_prev_solution_(false),
            last_solve_cost_(0.0), last_solve_time_ms_(0.0) {
    
    // Pre-allocate reference windows
    x_ref_window_.resize(N_ + 1);
    u_ref_window_.resize(N_);
    
    // Initialize with zero vectors
    int nx = robot_.nx();
    int nu = robot_.nu();
    
    for (int i = 0; i <= N_; ++i) {
        x_ref_window_[i] = Eigen::VectorXd::Zero(nx);
    }
    for (int i = 0; i < N_; ++i) {
        u_ref_window_[i] = Eigen::VectorXd::Zero(nu);
    }
    
    // Initialize previous solution storage
    prev_xbar_.resize(N_ + 1);
    prev_ubar_.resize(N_);
    prev_K_.resize(N_);
    
    for (int i = 0; i <= N_; ++i) {
        prev_xbar_[i] = Eigen::VectorXd::Zero(nx);
    }
    for (int i = 0; i < N_; ++i) {
        prev_ubar_[i] = Eigen::VectorXd::Zero(nu);
        prev_K_[i] = Eigen::MatrixXd::Zero(nu, nx);
    }
    
    std::cout << "MPC initialized with N=" << N_ << ", dt=" << dt_ << std::endl;
}

bool MPC::stepOnce(const Eigen::VectorXd& x_measured, Eigen::VectorXd& u_apply) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Extract reference window for current time index
        extractReferenceWindow();
        
        // Initialize iLQR with improved warm start logic
        if (has_prev_solution_) {
            // Use reference-aware initialization with warm start from previous solution
            ilqr_.initializeWithReference(x_measured, x_ref_window_, u_ref_window_, &prev_xbar_, &prev_ubar_);
        } else {
            // Use reference-aware cold start initialization  
            ilqr_.initializeWithReference(x_measured, x_ref_window_, u_ref_window_);
        }
        
        // Solve multi-iteration iLQR
        double solve_cost;
        bool success = ilqr_.solve(x_measured, x_ref_window_, u_ref_window_, solve_cost);
        
        if (!success) {
            std::cerr << "iLQR solve failed at time index " << t_idx_ << std::endl;
            
            // Fallback: use previous solution or zero control
            if (has_prev_solution_) {
                u_apply = prev_ubar_[0];
            } else {
                u_apply = Eigen::VectorXd::Zero(robot_.nu());
            }
            return false;
        }
        
        // Get current solution
        const auto& xbar = ilqr_.xbar();
        const auto& ubar = ilqr_.ubar();
        const auto& K = ilqr_.gainsK();
        
        // Compute TV-LQR control: u = ubar[0] + K * (x_measured - xbar)
        Eigen::VectorXd x_err = x_measured - xbar[0];
        u_apply = ubar[0] + K[0] * x_err;

        // Store solution for next iteration's warm start
        prev_xbar_ = xbar;
        prev_ubar_ = ubar;
        prev_K_ = K;
        has_prev_solution_ = true;
        
        // Update statistics
        last_solve_cost_ = solve_cost;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        last_solve_time_ms_ = duration.count() / 1000.0;
        
        // Advance time index
        t_idx_++;
        
    // Log current step (if file opened)
    logCurrentStep(x_measured, u_apply);

    // Log optimal trajectory (if enabled)
    logAppliedOptimal(x_measured, u_apply);
                
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in MPC step: " << e.what() << std::endl;
        u_apply = Eigen::VectorXd::Zero(robot_.nu());
        return false;
    }
}

/* bool MPC::stepOnce(const Eigen::VectorXd& x_measured, Eigen::VectorXd& u_apply) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // TEST 1: COMPLETELY SKIP ALL MPC SOLVING
        // Just apply zero control without any rollouts or solving
        u_apply = Eigen::VectorXd::Zero(robot_.nu());
        
        // Update timing for consistency
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        last_solve_time_ms_ = duration.count() / 1000.0;
        last_solve_cost_ = 0.0;  // No solve = zero cost
        
        // Advance time index
        t_idx_++;
        
        // Log if enabled
        logCurrentStep(x_measured, u_apply);
        logAppliedOptimal(x_measured, u_apply);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in MPC step: " << e.what() << std::endl;
        u_apply = Eigen::VectorXd::Zero(robot_.nu());
        return false;
    }
} */


void MPC::reset() {
    t_idx_ = 0;
    has_prev_solution_ = false;
    last_solve_cost_ = 0.0;
    last_solve_time_ms_ = 0.0;
    
    // Clear previous solution
    int nx = robot_.nx();
    int nu = robot_.nu();
    
    for (int i = 0; i <= N_; ++i) {
        prev_xbar_[i] = Eigen::VectorXd::Zero(nx);
    }
    for (int i = 0; i < N_; ++i) {
        prev_ubar_[i] = Eigen::VectorXd::Zero(nu);
        prev_K_[i] = Eigen::MatrixXd::Zero(nu, nx);
    }
    
    std::cout << "MPC reset" << std::endl;
}

void MPC::getNominalTrajectory(std::vector<Eigen::VectorXd>& x_traj,
                               std::vector<Eigen::VectorXd>& u_traj) const {
    if (has_prev_solution_) {
        x_traj = prev_xbar_;
        u_traj = prev_ubar_;
    } else {
        // Return empty trajectories if no solution available
        x_traj.clear();
        u_traj.clear();
    }
}

void MPC::extractReferenceWindow() {
    // Get reference window starting from current time index
    robot_.getReferenceWindow(t_idx_, N_, x_ref_window_, u_ref_window_);
}

Eigen::VectorXd MPC::computeTVLQRControl(const Eigen::VectorXd& x_measured) {
    // This method provides TV-LQR control between MPC solves  (Can be called at higher frequency than stepOnce)
    
    if (!has_prev_solution_) {
        return Eigen::VectorXd::Zero(robot_.nu());
    }
    
    // Use first gains and nominal from previous solution
    Eigen::VectorXd x_err = x_measured - prev_xbar_[0];
    // return prev_ubar_ + prev_K_ * x_err;
    return prev_ubar_[0] + prev_K_[0] * x_err;
}

void MPC::enableCSVLogging(const std::string& filename) {
    csv_filename_ = filename;
    csv_file_.open(csv_filename_, std::ios::out | std::ios::trunc);
    
    if (!csv_file_.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_filename_ << std::endl;
        return;
    }
    
    // Write CSV header
    csv_file_ << "time_index,time_sec,solve_cost,solve_time_ms";
    
    // Add state columns
    for (int i = 0; i < robot_.nx(); ++i) {
        csv_file_ << ",x_" << i;
    }
    
    // Add control columns
    for (int i = 0; i < robot_.nu(); ++i) {
        csv_file_ << ",u_" << i;
    }
    
    // Add reference state columns
    for (int i = 0; i < robot_.nx(); ++i) {
        csv_file_ << ",x_ref_" << i;
    }
    
    // Add reference control columns
    for (int i = 0; i < robot_.nu(); ++i) {
        csv_file_ << ",u_ref_" << i;
    }
    
    csv_file_ << std::endl;
    
    std::cout << "CSV logging started: " << csv_filename_ << std::endl;
}

void MPC::logCurrentStep(const Eigen::VectorXd& x_measured, const Eigen::VectorXd& u_applied) {
    if (!csv_file_.is_open()) {
        return;
    }
    
    // Basic info
    csv_file_ << t_idx_ << "," << (t_idx_ * dt_) << "," 
              << last_solve_cost_ << "," << last_solve_time_ms_;
    
    // Measured state
    for (int i = 0; i < x_measured.size(); ++i) {
        csv_file_ << "," << x_measured(i);
    }
    
    // Applied control
    for (int i = 0; i < u_applied.size(); ++i) {
        csv_file_ << "," << u_applied(i);
    }
    
    // Reference state (first element of current window)
    if (!x_ref_window_.empty()) {
        for (int i = 0; i < x_ref_window_[0].size(); ++i) {
            csv_file_ << "," << x_ref_window_[0](i);
        }
    } else {
        for (int i = 0; i < robot_.nx(); ++i) {
            csv_file_ << ",0.0";
        }
    }
    
    // Reference control (first element of current window)
    if (!u_ref_window_.empty()) {
        for (int i = 0; i < u_ref_window_[0].size(); ++i) {
            csv_file_ << "," << u_ref_window_[0](i);
        }
    } else {
        for (int i = 0; i < robot_.nu(); ++i) {
            csv_file_ << ",0.0";
        }
    }
    
    csv_file_ << std::endl;
}

void MPC::finalizeCSVLog() {
    if (csv_file_.is_open()) {
        csv_file_.flush();
        csv_file_.close();
        std::cout << "CSV log finalized: " << csv_filename_ << std::endl;
    }
}

void MPC::enableOptimalTrajectoryLogging(const std::string& base_path) {
    trajectory_base_path_ = base_path;
    
    // Open q_optimal.csv file
    std::string q_filename = base_path + "/q_optimal.csv";
    q_optimal_file_.open(q_filename, std::ios::out | std::ios::trunc);
    
    // Open u_optimal.csv file  
    std::string u_filename = base_path + "/u_optimal.csv";
    u_optimal_file_.open(u_filename, std::ios::out | std::ios::trunc);
    
    if (!q_optimal_file_.is_open() || !u_optimal_file_.is_open()) {
        std::cerr << "Failed to open optimal trajectory files in: " << base_path << std::endl;
        return;
    }
    
    // Write headers for q_optimal.csv (only position coordinates)
    q_optimal_file_ << "step,time_sec";
    for (int i = 0; i < robot_.nq(); ++i) {  // Only nq, not nx
        q_optimal_file_ << ",q_" << i;
    }
    q_optimal_file_ << std::endl;
    
    // Write headers for u_optimal.csv
    u_optimal_file_ << "step,time_sec";
    for (int i = 0; i < robot_.nu(); ++i) {
        u_optimal_file_ << ",u_" << i;
    }
    u_optimal_file_ << std::endl;
    
    // std::cout << "Optimal trajectory logging enabled. Files: " << q_filename << " and " << u_filename << std::endl;
}

void MPC::logAppliedOptimal(const Eigen::VectorXd& x_applied, const Eigen::VectorXd& u_applied) {
    if (!q_optimal_file_.is_open() || !u_optimal_file_.is_open()) {
        return;
    }
    
    // Log the applied state (x_applied is the current state, not the optimal)
    // We want the first element of the optimal trajectory from iLQR
    const auto& x_optimal = ilqr_.xbar();  // Get optimal state trajectory
    const auto& u_optimal = ilqr_.ubar();  // Get optimal control trajectory
    
    // Write applied optimal state (only q part - position coordinates)
    q_optimal_file_ << t_idx_ << "," << (t_idx_ * dt_);
    if (!x_optimal.empty()) {
        // Extract only q part (first nq elements) from state vector x = [q, v]
        int nq = robot_.nq();
        for (int i = 0; i < nq; ++i) {
            q_optimal_file_ << "," << x_optimal[0](i);
        }
    } else {
        // Fallback to current state q part if no optimal trajectory available
        int nq = robot_.nq();
        for (int i = 0; i < nq; ++i) {
            q_optimal_file_ << "," << x_applied(i);
        }
    }
    q_optimal_file_ << std::endl;
    
    // Write applied optimal control (first element of optimized trajectory)
    u_optimal_file_ << t_idx_ << "," << (t_idx_ * dt_);
    if (!u_optimal.empty()) {
        for (int i = 0; i < u_optimal[0].size(); ++i) {
            u_optimal_file_ << "," << u_optimal[0](i);
        }
    } else {
        // Fallback to applied control if no optimal trajectory available
        for (int i = 0; i < u_applied.size(); ++i) {
            u_optimal_file_ << "," << u_applied(i);
        }
    }
    u_optimal_file_ << std::endl;
}

void MPC::finalizeOptimalTrajectoryLog() {
    if (q_optimal_file_.is_open()) {
        q_optimal_file_.flush();
        q_optimal_file_.close();
    }
    if (u_optimal_file_.is_open()) {
        u_optimal_file_.flush();
        u_optimal_file_.close();
    }
    std::cout << "Optimal trajectory logs finalized: " << trajectory_base_path_ << "/q_optimal.csv and u_optimal.csv" << std::endl;
}
