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
    bool loadModel(const std::string& xml_path);
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
    void setCoMWeight(double w_com) { w_com_ = w_com; }
    double getCoMWeight() const { return w_com_; }
    void setCoMVelWeight(double w_com_vel) { w_com_vel_ = w_com_vel; }
    double getCoMVelWeight() const { return w_com_vel_; }
    void setEEPosWeight(double w_ee) { w_ee_pos_ = w_ee; }
    double getEEPosWeight() const { return w_ee_pos_; }
    void setEEVelWeight(double w_ee_vel) { w_ee_vel_ = w_ee_vel; }
    double getEEVelWeight() const { return w_ee_vel_; }
    double getUprightWeight() const { return w_upright_; }
    void setUprightWeight(double w_upright) { w_upright_ = w_upright;}
    void setBalanceWeight(double w_balance) { w_balance_ = w_balance; }
    double getBalanceWeight() const { return w_balance_; }
    
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
                            std::vector<Eigen::Vector3d>& com_ref_window) const;
    
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
    double w_com_;  // CoM tracking weight
    double w_com_vel_;  // CoM velocity tracking weight (separate from position)
    double w_ee_pos_, w_ee_vel_;
    double w_upright_; // Upright Posture Penalty
    double w_balance_; // Balance cost weight (capture point)
    
    // Constraint weights
    double w_joint_limits_;
    double w_control_limits_;

    // Reference trajectories (full length)
    std::vector<Eigen::VectorXd> x_ref_full_;
    std::vector<Eigen::VectorXd> u_ref_full_;
    std::vector<Eigen::Vector3d> com_ref_full_;
    std::vector<Eigen::Vector3d> com_vel_ref_full_;  // CoM velocity references (separate)
    std::vector<std::vector<Eigen::Vector3d>> ee_pos_ref_full_;  // [time][ee_idx] = position
    std::vector<std::vector<Eigen::Vector3d>> ee_vel_ref_full_;  // [time][ee_idx] = velocity
    
    // Contact schedule: contact_schedule_[t][ee_idx] = 1 (stance) or 0 (swing)
    std::vector<std::vector<int>> contact_schedule_;
    
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
    
    // CoM computation
    Eigen::Vector3d computeCoM(const Eigen::VectorXd& x) const;

};