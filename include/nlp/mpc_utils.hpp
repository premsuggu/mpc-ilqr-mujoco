#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <vector>
#include <string>

namespace nlp {

// Forward declarations
class NLPSolver;
struct NLPConfig;

// Type aliases for clarity
using ContactSchedule = std::vector<std::vector<bool>>;

/**
 * @brief Reference trajectories for MPC
 */
struct References {
    std::vector<Eigen::VectorXd> q_ref;
    std::vector<Eigen::VectorXd> v_ref;
    std::vector<Eigen::Vector3d> com_ref;
    std::vector<std::vector<Eigen::Vector3d>> ee_pos_ref;
};

/**
 * @brief MPC execution results
 */
struct MPCResults {
    std::vector<Eigen::VectorXd> q_trajectory;
    std::vector<Eigen::VectorXd> v_trajectory;
    std::vector<Eigen::VectorXd> u_trajectory;
    double total_time_s;
    int num_steps_completed;
    bool success;
};

/**
 * @brief Model Predictive Control orchestrator for NLP solver
 * 
 * Manages the MPC loop following the same pattern as iLQR MPC:
 * - Maintains time index and reference windows
 * - Calls NLP solver for optimization
 * - Handles warm starting
 * - Integrates state forward
 * - Logs results
 */
class MPCUtils {
public:
    /**
     * @brief Constructor
     * @param config MPC configuration
     * @param model Pinocchio model
     * @param refs Full reference trajectories
     * @param solver NLP solver instance
     */
    MPCUtils(const NLPConfig& config,
             const pinocchio::Model& model,
             const References& refs,
             NLPSolver& solver);

    /**
     * @brief Execute full MPC loop
     * @return Results with trajectories and statistics
     */
    MPCResults run();

    /**
     * @brief Single MPC step (call at control rate)
     * @param x_current Current state [q; v]
     * @param u_apply Output: control to apply
     * @return true if successful
     */
    bool stepOnce(const Eigen::VectorXd& x_current, Eigen::VectorXd& u_apply);

    /**
     * @brief Reset MPC to initial state
     */
    void reset();

    /**
     * @brief Set current time index
     */
    void setTimeIndex(int t_idx) { t_idx_ = t_idx; }

    /**
     * @brief Get current time index
     */
    int getTimeIndex() const { return t_idx_; }

private:
    const NLPConfig& config_;
    const pinocchio::Model& model_;
    const References& refs_;
    NLPSolver& solver_;
    
    int t_idx_;  // Current time index into reference
    bool has_warm_start_;
    Eigen::VectorXd W_warm_;  // Warm start decision variables

    /**
     * @brief Extract reference window for current timestep
     */
    References extractReferenceWindow();

    /**
     * @brief Create contact schedule for horizon
     */
    ContactSchedule createContactSchedule();

    /**
     * @brief Integrate state forward
     */
    Eigen::VectorXd integrateState(const Eigen::VectorXd& x_current,
                                    const Eigen::VectorXd& a);
};

} // namespace nlp
