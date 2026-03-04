#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <string>
#include <map>

// Forward declare NormParams to avoid circular dependency
namespace ilqr {
    enum class NormType;
    struct NormParams;
}

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
     * @brief Compute height (torso z) cost gradient (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param goal_z Target torso z height
     * @param weight Cost weight
     * @return Gradient vector w.r.t. full state [q, v]
     */
    Eigen::VectorXd HeightGrad(const Eigen::VectorXd& x,
                               double goal_z,
                               double weight = 1.0);

    /**
     * @brief Compute height (torso z) cost hessian (cached, fast evaluation)
     * @param x Full state vector [q, v]
     * @param goal_z Target torso z height
     * @param weight Cost weight
     * @return Hessian matrix w.r.t. full state [q, v]
     */
    Eigen::MatrixXd HeightHess(const Eigen::VectorXd& x,
                               double goal_z,
                               double weight = 1.0);

    /**
     * @brief Compute CoM xy velocity cost gradient (2D, zero target, DeepMind "Velocity")
     * @param x Full state vector [q, v]
     * @param weight Cost weight
     * @return Gradient vector w.r.t. full state [q, v]
     */
    Eigen::VectorXd VelocityGrad(const Eigen::VectorXd& x,
                               double weight = 1.0);

    /**
     * @brief Compute CoM xy velocity cost hessian (2D, zero target, DeepMind "Velocity")
     * @param x Full state vector [q, v]
     * @param weight Cost weight
     * @return Hessian matrix w.r.t. full state [q, v]
     */
    Eigen::MatrixXd VelocityHess(const Eigen::VectorXd& x,
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
     * @brief Get configuration DOF
     */
    int nq() const { return model_.nq; }

    /**
     * @brief Set gravity magnitude for balance cost computation
     * @param g Gravity magnitude (m/s^2)
     */
    void setGravity(double g) { gravity_ = g; }

    /**
     * @brief Get current gravity magnitude
     */
    double getGravity() const { return gravity_; }
    
    /**
     * @brief Set norm parameters for all cost terms
     * @param norm_params Map of cost term names to their norm configurations
     */
    void setNormParams(const std::map<std::string, ilqr::NormParams>& norm_params);

    /**
     * @brief Set body names for the Pelvis/Feet cost (robot-agnostic, read from config).
     * @param pelvis  Pelvis body name (z-position reference), e.g. "pelvis"
     * @param foot_r  Right foot body name, e.g. "foot_right"
     * @param foot_l  Left foot body name,  e.g. "foot_left"
     */
    void setPelvisFeetBodyNames(const std::string& pelvis,
                                const std::string& foot_r,
                                const std::string& foot_l);

    Eigen::VectorXd PelvisFeetGrad(const Eigen::VectorXd& x, double w);
    Eigen::MatrixXd PelvisFeetHess(const Eigen::VectorXd& x, double w);

    // Make data accessible to validation functions
    pinocchio::Model model_;
    pinocchio::Data data_;

private:
    // CasADi symbolic computation setup
    typedef ::casadi::SX ADScalar;
    pinocchio::ModelTpl<ADScalar> ad_model_;
    pinocchio::DataTpl<ADScalar> ad_data_;
    ::casadi::SX x_sym_;  // Full state [q, v]
    
    // Height cost functions (single instance)
    ::casadi::Function height_grad_fn_;     // Height gradient function
    ::casadi::Function height_hess_fn_;     // Height Hessian function
    bool height_functions_built_;
    
    // CoM velocity cost functions (single instance, separate from position)
    ::casadi::Function vel_grad_fn_;     // Velocity gradient function
    ::casadi::Function vel_hess_fn_;     // Velocity Hessian function
    bool vel_functions_built_;

    // Upright cost funtion
    ::casadi::Function upright_grad_fn_;
    ::casadi::Function upright_hess_fn_;
    bool upright_functions_built_;
    
    // Balance cost functions (capture point)
    ::casadi::Function balance_grad_fn_;
    ::casadi::Function balance_hess_fn_;
    bool balance_functions_built_;
    
    // Pelvis/Feet cost functions (body z-position alignment)
    ::casadi::Function pelvis_feet_grad_fn_;
    ::casadi::Function pelvis_feet_hess_fn_;
    bool pelvis_feet_functions_built_;
    std::string pf_pelvis_body_;  // e.g. "pelvis"
    std::string pf_foot_r_body_;  // e.g. "foot_right"
    std::string pf_foot_l_body_;  // e.g. "foot_left"
    
    // State dimensions (cached for efficiency)
    int nx_;  // Full state size (nq + nv)
    
    // Gravity magnitude for balance cost (set from config)
    double gravity_;
    
    // Build all symbolic functions once in constructor
    void buildSymbolicFunctions();
    
    // Helper to build height functions (once)
    void buildHeightFunctions();
    
    // Helper to build CoM velocity functions (once, separate from position)
    void buildVelocityFunctions();

    // Helper to build upright cost functions
    void buildUprightFunctions();
    
    // Helper to build balance cost functions (capture point)
    void buildBalanceFunctions();
    
    // Helper to build pelvis/feet cost functions
    void buildPelvisFeetFunctions();
    
    // Symbolic cost expression helpers 
    ::casadi::SX symHeight(const ::casadi::SX& target_z,
                           const ::casadi::SX& weight);
    
    ::casadi::SX symVelocity(const ::casadi::SX& weight);
    
    ::casadi::SX symUpright(const ::casadi::SX& weight);
    
    ::casadi::SX symBalance(const ::casadi::SX& p_support,
                            const ::casadi::SX& weight);
    
    ::casadi::SX symPelvisFeet(const ::casadi::SX& weight);
                            
    // Norm parameters for each cost term
    std::map<std::string, ilqr::NormParams> norm_params_;
public:
    pinocchio::FrameIndex getFrameId(const std::string& frame_name);
};
} // namespace derivatives