#pragma once

#include "robot_utils.hpp"
#include "ilqr.hpp"
#include <vector>
#include <fstream>
#include <string>

/**
 * @brief Model Predictive Control orchestrator
 * 
 * Manages the MPC loop:
 * - Maintains time index and reference windows
 * - Calls iLQR solver for single iterations
 * - Applies TV-LQR feedback between solves
 * - Handles warm starting and logging
 */
class MPC {
public:
    MPC(RobotUtils& robot, int N, double dt);

    // Main MPC step (call at 50Hz)
    bool stepOnce(const Eigen::VectorXd& x_measured, Eigen::VectorXd& u_apply);

    // Configuration and state
    void reset();
    void setTimeIndex(int t_idx) { t_idx_ = t_idx; }
    int getTimeIndex() const { return t_idx_; }

    // CSV logging (now always enabled once initialized)
    void enableCSVLogging(const std::string& filename); // kept for filename selection
    void logCurrentStep(const Eigen::VectorXd& x_measured, const Eigen::VectorXd& u_applied);
    void finalizeCSVLog(); // optional explicit flush/close
    
    // Optimal trajectory logging
    void enableOptimalTrajectoryLogging(const std::string& base_path); // choose path
    void logAppliedOptimal(const Eigen::VectorXd& x_applied, const Eigen::VectorXd& u_applied);
    void finalizeOptimalTrajectoryLog();

    // Access to internal components
    const iLQR& solver() const { return ilqr_; }
    const std::vector<Eigen::MatrixXd>& gainsK() const { return prev_K_; }
    double getLastSolveCost() const { return last_solve_cost_; }

    // Trajectory access for logging
    void getNominalTrajectory(std::vector<Eigen::VectorXd>& x_traj,
                              std::vector<Eigen::VectorXd>& u_traj) const;

private:
    RobotUtils& robot_;
    iLQR ilqr_;
    
    int N_;           // Horizon length  
    double dt_;       // Time step
    int t_idx_;       // Current time index into reference

    // Reference window (extracted each iteration)
    std::vector<Eigen::VectorXd> x_ref_window_, u_ref_window_;
    std::vector<Eigen::Vector3d> com_ref_window_;  // CoM reference window

    // Previous solution for warm starting
    bool has_prev_solution_;
    std::vector<Eigen::VectorXd> prev_xbar_, prev_ubar_;
    std::vector<Eigen::MatrixXd> prev_K_;

    // Statistics
    double last_solve_cost_;
    double last_solve_time_ms_;

    // CSV logging
    std::ofstream csv_file_;
    std::string csv_filename_;
    
    // Optimal trajectory logging (always active once enabled)
    std::string trajectory_base_path_;
    std::ofstream q_optimal_file_;
    std::ofstream u_optimal_file_;

    // Helper functions
    void extractReferenceWindow();
    Eigen::VectorXd computeTVLQRControl(const Eigen::VectorXd& x_measured);
};