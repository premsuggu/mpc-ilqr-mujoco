#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <string>

namespace derivatives {

/**
 * @brief Convert MuJoCo state to Pinocchio-compatible state (fixes quaternion ordering)
 * @param mujoco_state State vector from MuJoCo [q, v] with quaternion [qw,qx,qy,qz]
 * @param nq Number of position DOF (to locate quaternion)
 * @return Pinocchio-compatible state with quaternion [qx,qy,qz,qw]
 */
Eigen::VectorXd convertMuJoCoToPinocchio(const Eigen::VectorXd& mujoco_state, int nq);

/**
 * @brief Efficient end-effector derivatives using Pinocchio-CasADi symbolic differentiation
 */
class symDerivatives {
public:
    /**
     * @brief Initialize with robot model
     * @param urdf_path Path to URDF model
     * @param floating_base Use floating base model
     */
    symDerivatives(const std::string& urdf_path, bool floating_base = true);

    /**
     * @brief Compute end-effector position gradient (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param target_pos Target position [x, y, z]  
     * @param frame_name End-effector frame name
     * @param weight Cost weight
     * @return Gradient vector w.r.t. full state [q, v]
     */
    Eigen::VectorXd EEposGrad(const Eigen::VectorXd& x, 
                              const Eigen::Vector3d& target_pos,
                              const std::string& frame_name,
                              double weight = 1.0);

    /**
     * @brief Compute end-effector position hessian (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param target_pos Target position [x, y, z]
     * @param frame_name End-effector frame name  
     * @param weight Cost weight
     * @return Hessian matrix w.r.t. full state [q, v]
     */
    Eigen::MatrixXd EEposHess(const Eigen::VectorXd& x,
                              const Eigen::Vector3d& target_pos,
                              const std::string& frame_name,
                              double weight = 1.0);
                              
    /**
     * @brief Compute center-of-mass position gradient (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param target_com Target CoM position [x, y, z]
     * @param weight Cost weight
     * @return Gradient vector w.r.t. full state [q, v]
     */
    Eigen::VectorXd CoMGrad(const Eigen::VectorXd& x,
                            const Eigen::Vector3d& target_com,
                            double weight = 1.0);

    /**
     * @brief Compute center-of-mass position hessian (cached, fast evaluation)  
     * @param x Full state vector [q, v]
     * @param target_com Target CoM position [x, y, z]
     * @param weight Cost weight
     * @return Hessian matrix w.r.t. full state [q, v]
     */
    Eigen::MatrixXd CoMHess(const Eigen::VectorXd& x,
                            const Eigen::Vector3d& target_com,
                            double weight = 1.0);

    /**
     * @brief Compute end-effector velocity gradient (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param target_vel Target velocity [vx, vy, vz]
     * @param frame_name End-effector frame name
     * @param weight Cost weight
     * @return Gradient vector w.r.t. full state [q, v]
     */
    Eigen::VectorXd EEvelGrad(const Eigen::VectorXd& x,
                              const Eigen::Vector3d& target_vel,
                              const std::string& frame_name,
                              double weight = 1.0);

    /**
     * @brief Compute end-effector velocity hessian (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param target_vel Target velocity [vx, vy, vz]
     * @param frame_name End-effector frame name
     * @param weight Cost weight
     * @return Hessian matrix w.r.t. full state [q, v]
     */
    Eigen::MatrixXd EEvelHess(const Eigen::VectorXd& x,
                              const Eigen::Vector3d& target_vel,
                              const std::string& frame_name,
                              double weight = 1.0);
    
    /**
     * @brief Compute center-of-mass velocity gradient (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param target_com_vel Target CoM velocity [vx, vy, vz]
     * @param weight Cost weight
     * @return Gradient vector w.r.t. full state [q, v]
     */
    Eigen::VectorXd CoMVelGrad(const Eigen::VectorXd& x,
                               const Eigen::Vector3d& target_com_vel,
                               double weight = 1.0);

    /**
     * @brief Compute center-of-mass velocity hessian (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param target_com_vel Target CoM velocity [vx, vy, vz]
     * @param weight Cost weight
     * @return Hessian matrix w.r.t. full state [q, v]
     */
    Eigen::MatrixXd CoMVelHess(const Eigen::VectorXd& x,
                               const Eigen::Vector3d& target_com_vel,
                               double weight = 1.0);

    Eigen::VectorXd UprightGrad(const Eigen::VectorXd& x, 
                                double w_upright);

    Eigen::MatrixXd UprightHess(const Eigen::VectorXd& x, 
                                double w_upright);

    /**
     * @brief Compute balance cost gradient using capture point (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param p_support Support center position [x, y]
     * @param w_balance Balance cost weight
     * @return Gradient vector w.r.t. full state [q, v]
     */
    Eigen::VectorXd BalanceGrad(const Eigen::VectorXd& x,
                                const Eigen::Vector2d& p_support,
                                double w_balance);

    /**
     * @brief Compute balance cost hessian using capture point (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param p_support Support center position [x, y]
     * @param w_balance Balance cost weight
     * @return Hessian matrix w.r.t. full state [q, v]
     */
    Eigen::MatrixXd BalanceHess(const Eigen::VectorXd& x,
                                const Eigen::Vector2d& p_support,
                                double w_balance);

    /**
     * @brief Pre-build functions for specific end-effector frame
     * @param frame_name Frame to prepare functions for
     */
    void prepareFrame(const std::string& frame_name);

    /**
     * @brief Get configuration DOF
     */
    int nq() const { return model_.nq; }

    // Make data accessible to validation functions
    pinocchio::Model model_;
    pinocchio::Data data_;

private:
    // CasADi symbolic computation setup
    typedef ::casadi::SX ADScalar;
    pinocchio::ModelTpl<ADScalar> ad_model_;
    pinocchio::DataTpl<ADScalar> ad_data_;
    ::casadi::SX x_sym_;  // Full state [q, v]
    
    // Pre-compiled function caches for different cost types
    std::map<std::string, ::casadi::Function> ee_pos_fns_;     // End-effector position functions
    std::map<std::string, ::casadi::Function> ee_grad_fns_;    // EE position gradient functions  
    std::map<std::string, ::casadi::Function> ee_hess_fns_;    // EE position Hessian functions
    std::map<std::string, ::casadi::Function> ee_vel_grad_fns_; // EE velocity gradient functions
    std::map<std::string, ::casadi::Function> ee_vel_hess_fns_; // EE velocity Hessian functions
    
    // CoM cost functions (single instance)
    ::casadi::Function com_grad_fn_;     // CoM gradient function
    ::casadi::Function com_hess_fn_;     // CoM Hessian function
    bool com_functions_built_;
    
    // CoM velocity cost functions (single instance, separate from position)
    ::casadi::Function com_vel_grad_fn_;     // CoM velocity gradient function
    ::casadi::Function com_vel_hess_fn_;     // CoM velocity Hessian function
    bool com_vel_functions_built_;

    // Upright cost funtion
    ::casadi::Function upright_grad_fn_;
    ::casadi::Function upright_hess_fn_;
    bool upright_functions_built_;
    
    // Balance cost functions (capture point)
    ::casadi::Function balance_grad_fn_;
    ::casadi::Function balance_hess_fn_;
    bool balance_functions_built_;
    
    // State dimensions (cached for efficiency)
    int nx_;  // Full state size (nq + nv)
    
    // Build all symbolic functions once in constructor
    void buildSymbolicFunctions();
    
    // Helper to build end-effector position functions for a specific frame
    void buildEEFunctions(const std::string& frame_name);
    
    // Helper to build end-effector velocity functions for a specific frame
    void buildEEVelFunctions(const std::string& frame_name);
    
    // Helper to build CoM functions (once)
    void buildCoMFunctions();
    
    // Helper to build CoM velocity functions (once, separate from position)
    void buildCoMVelFunctions();

    // Helper to build upright cost functions
    void buildUprightFunctions();
    
    // Helper to build balance cost functions (capture point)
    void buildBalanceFunctions();
    
    // Symbolic cost expression helpers 
    ::casadi::SX symCoMPos(const ::casadi::SX& target_com,
                           const ::casadi::SX& weight);
    
    ::casadi::SX symCoMVel(const ::casadi::SX& target_com_vel,
                           const ::casadi::SX& weight);
    
    ::casadi::SX symEEPos(const ::casadi::SX& target_pos,
                          const ::casadi::SX& weight,
                          pinocchio::FrameIndex frame_id);
    
    ::casadi::SX symEEVel(const ::casadi::SX& target_vel,
                          const ::casadi::SX& weight,
                          pinocchio::FrameIndex frame_id);
    
    ::casadi::SX symUpright(const ::casadi::SX& weight);
    
    ::casadi::SX symBalance(const ::casadi::SX& p_support,
                            const ::casadi::SX& weight);
public:
    pinocchio::FrameIndex getFrameId(const std::string& frame_name);
};
} // namespace derivatives