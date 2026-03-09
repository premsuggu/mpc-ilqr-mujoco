#pragma once

#include "ilqr/robot_utils.hpp"
#include "ilqr/ilqr.hpp"
#include "ilqr/norm.hpp"
#include <vector>
#include <fstream>
#include <string>
#include <map>

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
    MPC(RobotUtils& robot, int N, double dt, const std::string& urdf_path);

    // Main MPC step (call at 50Hz)
    bool stepOnce(const Eigen::VectorXd& x_measured, Eigen::VectorXd& u_apply);

    // Configuration and state
    void reset();
    void setTimeIndex(int t_idx) { t_idx_ = t_idx; }
    int getTimeIndex() const { return t_idx_; }
    
    /**
     * @brief Set gravity magnitude for balance cost computation
     * @param g Gravity magnitude (m/s^2)
     */
    void setGravity(double g) { ilqr_.setGravity(g); }
    
    /**
     * @brief Set balance time constant (omega_0) for capture point computation
     * @param omega Balance time constant (s). DeepMind: 0.2s for stand, 0.3s for walk
     */
    void setBalanceTimeConstant(double omega) { ilqr_.setBalanceTimeConstant(omega); }
    
    /**
     * @brief Configure norm parameters for all cost terms
     * @param norm_params Map of cost term names to their norm configurations
     */
    void configureNorms(const std::map<std::string, ilqr::NormParams>& norm_params);
    
    /**
     * @brief Configure iLQR solver settings
     */
    void configureSolver(double reg_min, double reg_max, double reg_increase_factor,
                        double reg_decrease_factor, double trust_region_good,
                        double trust_region_poor, int num_line_search_steps,
                        double min_linesearch_step, double line_search_tolerance,
                        double quu_regularization, double convergence_threshold);

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
    std::vector<Eigen::Vector3d> height_ref_window_;  // Height reference window

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