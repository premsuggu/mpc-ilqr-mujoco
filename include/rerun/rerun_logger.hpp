#pragma once
#include <rerun.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

/**
 * @brief Minimal Rerun logger - PLOTS ONLY
 * 
 * No 3D visualization, no arrows, no transforms.
 * Just log scalar time-series data for plotting.
 */
class RerunLogger {
public:
    RerunLogger(const std::string& app_name = "H1_Humanoid_MPC");
    ~RerunLogger();
    
    // Initialize (always spawns viewer if enabled)
    bool initialize();
    
    // Set simulation time
    void setTime(int step, double sim_time_sec);
    
    // === Simple scalar logging ===
    
    // Log base state (position and velocity components)
    void logBaseState(
        const Eigen::VectorXd& x_current,  // Full state [q, v]
        const Eigen::VectorXd* x_ref,      // Optional reference state
        int nq                              // Number of position states
    );
    
    // Log joint positions with limits
    void logJoints(
        const Eigen::VectorXd& q,          // Joint positions (full q including base)
        const Eigen::VectorXd& q_lower,
        const Eigen::VectorXd& q_upper,
        const std::vector<std::string>& joint_names,
        const Eigen::VectorXd* q_ref = nullptr  // Optional reference positions
    );
    
    // Log CoM state
    void logCoM(
        const Eigen::Vector3d& com_pos,
        const Eigen::Vector3d& com_vel,
        const Eigen::Vector3d* com_pos_ref = nullptr,
        const Eigen::Vector3d* com_vel_ref = nullptr
    );
    
    // Log end effector positions
    void logEndEffectors(
        const std::vector<Eigen::Vector3d>& ee_positions,  // Current positions
        const std::vector<bool>& contact_states,           // Contact flags
        const std::vector<Eigen::Vector3d>* ee_pos_ref = nullptr  // Optional references
    );
    
    // Log control torques
    void logTorques(
        const Eigen::VectorXd& u,
        const Eigen::VectorXd& u_limits,
        const std::vector<std::string>& joint_names
    );
    
private:
    rerun::RecordingStream rec_;
    bool initialized_;
};
