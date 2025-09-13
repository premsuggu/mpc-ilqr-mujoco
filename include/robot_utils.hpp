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
                            std::vector<Eigen::VectorXd>& u_ref_window) const;

    // Utility functions
    int jointId(const std::string& name) const;
    void resetToReference(int t);
    void initializeStandingPose();
    

    // EXTRAS
    void diagnoseContactForces() const;
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
    
    // Constraint weights
    double w_joint_limits_;
    double w_control_limits_;

    // Reference trajectories (full length)
    std::vector<Eigen::VectorXd> x_ref_full_;
    std::vector<Eigen::VectorXd> u_ref_full_;

    // Joint name to ID mapping
    std::unordered_map<std::string, int> joint_name_to_id_;

    // Helper functions
    void buildJointNameMap();
    void packState(Eigen::VectorXd& x) const;
    void unpackState(const Eigen::VectorXd& x);
    void packControl(Eigen::VectorXd& u) const;
    void unpackControl(const Eigen::VectorXd& u);

    // EXTRAS
    void unpackStateToData(const Eigen::VectorXd& x, mjData* target_data);
    void unpackControlToData(const Eigen::VectorXd& u, mjData* target_data);  
    void packStateFromData(Eigen::VectorXd& x, mjData* source_data) const;

};
