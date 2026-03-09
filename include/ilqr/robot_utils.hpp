#pragma once

#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>

/**
 * @brief MuJoCo-backed robot dynamics and utilities for MPC
 * 
 * This class wraps MuJoCo model/data and provides:
 * - State/control packing/unpacking 
 * - Forward simulation and finite-difference linearization
 * - Cost function evaluation with reference tracking
 * - Reference trajectory loading and windowing
 */
class RobotUtils {
public:
    RobotUtils();
    ~RobotUtils();

    // Model loading and configuration
    bool loadModel(const std::string& xml_path,
                   const std::string& left_foot_name = "left_ankle_link",
                   const std::string& right_foot_name = "right_ankle_link");
    void setContactImpratio(double impratio);
    void setTimeStep(double dt);

    // Dimensions
    int nx() const { return nx_; }
    int nu() const { return nu_; }
    int nq() const { return model_ ? model_->nq : 0; }
    int nv() const { return model_ ? model_->nv : 0; }
    double dt() const { return dt_; }

    // getters
    const Eigen::MatrixXd& Q() const { return Q_; }
    const Eigen::MatrixXd& R() const { return R_; }
    const Eigen::MatrixXd& Qf() const { return Qf_; }

    // State and control interface
    void setState(const Eigen::VectorXd& x);
    void getState(Eigen::VectorXd& x) const;
    void setControl(const Eigen::VectorXd& u);
    void step();

    // Forward dynamics (single step)
    void rolloutOneStep(const Eigen::VectorXd& x, const Eigen::VectorXd& u, 
                        Eigen::VectorXd& x_next);

    // Finite difference linearization
    void linearizeDynamicsFD(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                             Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                             double eps = 1e-5);

    // Cost functions
    double stageCost(int t, const Eigen::VectorXd& x, const Eigen::VectorXd& u) const;
    double terminalCost(const Eigen::VectorXd& x) const;
    void setCostWeights(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, 
                        const Eigen::MatrixXd& Qf);
    void setHeightWeight(double w) { w_height_ = w; }
    double getHeightWeight() const { return w_height_; }
    void setVelocityWeight(double w) { w_vel_ = w; }
    double getVelocityWeight() const { return w_vel_; }
    void setJointVelWeight(double w) { w_joint_vel_ = w; }
    double getJointVelWeight() const { return w_joint_vel_; }
    double getUprightWeight() const { return w_upright_; }
    void setUprightWeight(double w) { w_upright_ = w; }
    void setBalanceWeight(double w) { w_balance_ = w; }
    double getBalanceWeight() const { return w_balance_; }
    void setPelvisFeetWeight(double w) { w_pelvis_feet_ = w; }
    double getPelvisFeetWeight() const { return w_pelvis_feet_; }
    void setWalkWeight(double w)       { w_walk_ = w; }
    double getWalkWeight()       const { return w_walk_; }
    void setSpeedGoal(double sg)       { speed_goal_ = sg; }
    double getSpeedGoal()        const { return speed_goal_; }
    // Body name setters/getters (Pelvis/Feet cost, config-driven, robot-agnostic)
    void setLeftFootBodyName(const std::string& n)       { left_foot_body_name_ = n; }
    void setRightFootBodyName(const std::string& n)      { right_foot_body_name_ = n; }
    void setPelvisBodyName(const std::string& n)         { pelvis_body_name_ = n; }
    void setTorsoBodyName(const std::string& n)          { torso_body_name_ = n; }
    void setWaistLowerBodyName(const std::string& n)     { waist_lower_body_name_ = n; }
    const std::string& getLeftFootBodyName()       const { return left_foot_body_name_; }
    const std::string& getRightFootBodyName()      const { return right_foot_body_name_; }
    const std::string& getPelvisBodyName()         const { return pelvis_body_name_; }
    const std::string& getTorsoBodyName()          const { return torso_body_name_; }
    const std::string& getWaistLowerBodyName()     const { return waist_lower_body_name_; }
    
    // Constraint cost functions
    double constraintCost(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const;
    void setConstraintWeights(double w_joint_limits, double w_control_limits);
    
    // Constraint gradients and hessians for iLQR
    void constraintGradients(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                           Eigen::VectorXd& grad_x, Eigen::VectorXd& grad_u) const;
    void constraintHessians(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                          Eigen::MatrixXd& hess_xx, Eigen::MatrixXd& hess_uu) const;

    // Reference trajectories
    bool loadReferences(const std::string& q_ref_path, const std::string& v_ref_path);
    void getReferenceWindow(int t0, int N, 
                            std::vector<Eigen::VectorXd>& x_ref_window,
                            std::vector<Eigen::VectorXd>& u_ref_window,
                            std::vector<Eigen::Vector3d>& height_ref_window) const;
    
    // Contact schedule
    bool loadContactSchedule(const std::string& contact_path);
    bool isStance(int ee_idx, int t) const;

    // Utility functions
    int jointId(const std::string& name) const;
    std::string getEEFrameName(int ee_idx) const;
    Eigen::Vector3d getEEReference(int t, int ee_idx) const;
    Eigen::Vector3d getEEVelReference(int t, int ee_idx) const;
    Eigen::Vector3d getCoMVelReference(int t) const;
    void resetToReference(int t);
    void initializeStandingPose();
    void computeGravComp(Eigen::VectorXd& ugrav) const;

    mjModel* model() const { return model_; }
    mjData* data() const { return data_; }
    void setGravity(double gx = 0.0, double gy = 0.0, double gz = 0.0);
    void scaleRobotMass(double scale_factor);
    
    // Public reference trajectories for Rerun visualization
    std::vector<Eigen::VectorXd> x_ref_full_;
    std::vector<Eigen::VectorXd> u_ref_full_;
    std::vector<Eigen::Vector3d> height_ref_full_;
    std::vector<Eigen::Vector3d> com_vel_ref_full_;
    std::vector<std::vector<Eigen::Vector3d>> ee_pos_ref_full_;
    std::vector<std::vector<Eigen::Vector3d>> ee_vel_ref_full_;
    std::vector<std::vector<int>> contact_schedule_;
    
private:
    // MuJoCo model and data
    mjModel* model_;
    mjData* data_;
    mjData* data_temp_;  // For finite difference computations

    // Dimensions
    int nx_, nu_;  // State and control dimensions
    double dt_;

    // Cost matrices
    Eigen::MatrixXd Q_, R_, Qf_;
    double w_height_;  // Height (torso z) tracking weight
    double w_vel_;  // CoM velocity cost weight (world-frame base xy, DeepMind "ComVel.")
    double w_joint_vel_;  // Joint velocity cost weight (21 joints, DeepMind "Joint Vel.")
    double w_upright_; // Upright Posture Penalty
    double w_balance_; // Balance cost weight (capture point)
    double w_pelvis_feet_; // Pelvis/Feet cost weight
    double w_walk_;        // Walk cost weight
    double speed_goal_;    // Target forward speed (m/s)
    std::string left_foot_body_name_;     // MuJoCo body name for left foot
    std::string right_foot_body_name_;    // MuJoCo body name for right foot
    std::string pelvis_body_name_;        // MuJoCo body name for pelvis
    std::string torso_body_name_;         // MuJoCo body name for torso (Walk forward direction)
    std::string waist_lower_body_name_;   // MuJoCo body for waist_lower subtree CoM vel (Walk)
    
    // Constraint weights
    double w_joint_limits_;
    double w_control_limits_;
    
    // Experiment 2: Linearization parameters
    double linearization_epsilon_;  // Finite difference step size
    
    // End-effector site IDs
    std::vector<int> ee_site_ids_;

    // Joint name to ID mapping
    std::unordered_map<std::string, int> joint_name_to_id_;

    // Helper functions for packing/unpacking (optimized with Eigen::Map)
    void buildJointNameMap();
    void unpackStateToData(const Eigen::VectorXd& x, mjData* target_data);
    void unpackControlToData(const Eigen::VectorXd& u, mjData* target_data);  
    void packStateFromData(Eigen::VectorXd& x, mjData* source_data) const;

public:
    
    // CoM and EE computation
    Eigen::Vector3d computeCoM(const Eigen::VectorXd& x) const;
    Eigen::Vector3d computeCoMVelocity(const Eigen::VectorXd& x) const;
    Eigen::Vector3d computeEEPos(const Eigen::VectorXd& x, int ee_idx) const;
    Eigen::Vector3d computeEEVel(const Eigen::VectorXd& x, int ee_idx) const;
    // World-frame z-position of a named MuJoCo body (for Pelvis/Feet cost)
    double computeBodyZPos(const Eigen::VectorXd& x, const std::string& body_name) const;
    // World-frame x-axis (column 0 of xmat) of a named MuJoCo body (for Walk forward direction)
    Eigen::Vector3d computeBodyXAxis(const Eigen::VectorXd& x, const std::string& body_name) const;
    // World-frame xy subtree CoM linear velocity of a named body (for Walk com_vel: waist_lower_subcomvel)
    // Reads data_temp_->subtreelinvel which is populated by mj_kinematics.
    Eigen::Vector2d computeSubtreeLinVel2d(const Eigen::VectorXd& x, const std::string& body_name) const;
    
    // Rerun visualization helpers
    Eigen::VectorXd getJointLowerLimits() const;
    Eigen::VectorXd getJointUpperLimits() const;
    Eigen::VectorXd getTorqueLimits() const;
    std::vector<std::string> getJointNames() const;
    std::vector<Eigen::Vector3d> getEndEffectorPositions() const;
    std::vector<bool> getContactStates(int time_step) const;

};