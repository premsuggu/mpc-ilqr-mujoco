#include "robot_utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

RobotUtils::RobotUtils() 
    : model_(nullptr), data_(nullptr), data_temp_(nullptr),
      nx_(0), nu_(0), dt_(0.01), w_com_(0.0), w_joint_limits_(500.0), w_control_limits_(1000.0) {
}

RobotUtils::~RobotUtils() {
    if (data_temp_) mj_deleteData(data_temp_);
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
}

bool RobotUtils::loadModel(const std::string& xml_path) {
    char error[1024] = {0};
    // Load the MuJoCo model from XML
    model_ = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    if (!model_) {
        std::cerr << "Failed to load model: " << error << std::endl;
        return false;
    }
    // Set up simulation data
    data_ = mj_makeData(model_);
    data_temp_ = mj_makeData(model_);
    if (!data_ || !data_temp_) {
        std::cerr << "Failed to create MuJoCo data structures" << std::endl;
        return false;
    }
    // Figure out state and control dimensions
    nx_ = model_->nq + model_->nv;
    nu_ = model_->nu;
    dt_ = model_->opt.timestep;
    std::cout << "Model loaded successfully:" << std::endl;
    // Build a map from joint names to IDs
    buildJointNameMap();
    
    // Initialize end-effector body IDs (using ankle links as foot end-effectors)
    ee_site_ids_.clear();
    int left_ankle_id = mj_name2id(model_, mjOBJ_BODY, "left_ankle_link");
    int right_ankle_id = mj_name2id(model_, mjOBJ_BODY, "right_ankle_link");
    if (left_ankle_id >= 0) ee_site_ids_.push_back(left_ankle_id);
    if (right_ankle_id >= 0) ee_site_ids_.push_back(right_ankle_id);
    std::cout << "Found " << ee_site_ids_.size() << " end-effector bodies" << std::endl;
    
    // Set up default cost matrices
    Q_ = Eigen::MatrixXd::Identity(nx_, nx_);
    R_ = Eigen::MatrixXd::Identity(nu_, nu_);
    Qf_ = Eigen::MatrixXd::Identity(nx_, nx_);
    return true;
}

// Tweak MuJoCo's contact solver for better stability
void RobotUtils::setContactImpratio(double impratio) {
    if (model_) {
        model_->opt.impratio = impratio;
        std::cout << "Set IMPRATIO to: " << impratio << std::endl;
    }
}

// Change the simulation timestep
void RobotUtils::setTimeStep(double dt) {
    dt_ = dt;
    if (model_) {
        model_->opt.timestep = dt;
        std::cout << "Set timestep to: " << dt << std::endl;
    }
}

// Set the robot's state (positions and velocities)
void RobotUtils::setState(const Eigen::VectorXd& x) {
    if (!data_ || x.size() != nx_) {
        std::cerr << "Invalid state size: " << x.size() << " (expected " << nx_ << ")" << std::endl;
        return;
    }
    unpackState(x);
}

// Get the robot's current state
void RobotUtils::getState(Eigen::VectorXd& x) const {
    if (!data_) return;
    x.resize(nx_);
    packState(x);
}

// Set the robot's control input (actuator commands)
void RobotUtils::setControl(const Eigen::VectorXd& u) {
    if (!data_ || u.size() != nu_) {
        std::cerr << "Invalid control size: " << u.size() << " (expected " << nu_ << ")" << std::endl;
        return;
    }
    unpackControl(u);
}

// Advance the simulation by one step
void RobotUtils::step() {
    if (!model_ || !data_) return;
    mj_step(model_, data_);
}

// Predict the next state given x and u, using a separate data buffer
void RobotUtils::rolloutOneStep(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                               Eigen::VectorXd& x_next) {
    if (!model_ || !data_temp_) return;
    // Save current state, do prediction in temp buffer
    mj_copyData(data_temp_, model_, data_);
    unpackStateToData(x, data_temp_);
    unpackControlToData(u, data_temp_);
    mj_forward(model_, data_temp_);
    mj_step(model_, data_temp_);
    packStateFromData(x_next, data_temp_);
    // No need to restore original state
}


void RobotUtils::linearizeDynamicsFD(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                                     Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                                     double eps) {
    if (!model_ || !data_ || !data_temp_) return;
    
    A.resize(nx_, nx_);
    B.resize(nx_, nu_);
    
    // Save original state
    setState(x);
    mj_copyData(data_temp_, model_, data_);
    
    // Get baseline next state: x_next = f(x, u)
    Eigen::VectorXd x_next_baseline(nx_);
    rolloutOneStep(x, u, x_next_baseline);
    
    // Compute A matrix: ∂f/∂x using forward differences
    for (int i = 0; i < nx_; ++i) {
        Eigen::VectorXd x_pert = x;
        x_pert(i) += eps;
        
        Eigen::VectorXd x_next_pert(nx_);
        rolloutOneStep(x_pert, u, x_next_pert);
        
        A.col(i) = (x_next_pert - x_next_baseline) / eps;
    }
    
    // Compute B matrix: ∂f/∂u using forward differences
    for (int j = 0; j < nu_; ++j) {
        Eigen::VectorXd u_pert = u;
        u_pert(j) += eps;
        
        Eigen::VectorXd x_next_pert(nx_);
        rolloutOneStep(x, u_pert, x_next_pert);
        
        B.col(j) = (x_next_pert - x_next_baseline) / eps;
    }
    
    // Restore original state
    mj_copyData(data_, model_, data_temp_);
}

double RobotUtils::stageCost(int t, const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
    if (t >= (int)x_ref_full_.size() || t >= (int)u_ref_full_.size()) {
        // Use last available reference if beyond loaded data
        int ref_idx = std::min(t, (int)x_ref_full_.size() - 1);
        int u_ref_idx = std::min(t, (int)u_ref_full_.size() - 1);
        
        Eigen::VectorXd x_err = x - x_ref_full_[ref_idx];
        Eigen::VectorXd u_err = u - u_ref_full_[u_ref_idx];
        
        double tracking_cost = 0.5 * (x_err.transpose() * Q_ * x_err + u_err.transpose() * R_ * u_err)(0, 0);
        
        // Add CoM tracking cost
        double com_cost = 0.0;
        if (w_com_ > 0.0 && !com_ref_full_.empty()) {
            Eigen::Vector3d com_current = computeCoM(x);
            int com_ref_idx = std::min(t, (int)com_ref_full_.size() - 1);
            Eigen::Vector3d com_err = com_current - com_ref_full_[com_ref_idx];
            com_cost = 0.5 * w_com_ * com_err.squaredNorm();
        }
        
        double constraint_cost_val = constraintCost(x, u);
        
        return tracking_cost + com_cost + constraint_cost_val;
    }
    
    Eigen::VectorXd x_err = x - x_ref_full_[t];
    Eigen::VectorXd u_err = u - u_ref_full_[t];
    
    double tracking_cost = 0.5 * (x_err.transpose() * Q_ * x_err + u_err.transpose() * R_ * u_err)(0, 0);
    
    // Add CoM tracking cost
    double com_cost = 0.0;
    if (w_com_ > 0.0 && t < (int)com_ref_full_.size()) {
        Eigen::Vector3d com_current = computeCoM(x);
        Eigen::Vector3d com_err = com_current - com_ref_full_[t];
        com_cost = 0.5 * w_com_ * com_err.squaredNorm();
    }
    
    double constraint_cost_val = constraintCost(x, u);
    
    return tracking_cost + com_cost + constraint_cost_val;
}

double RobotUtils::terminalCost(const Eigen::VectorXd& x) const {
    if (x_ref_full_.empty()) {
        return 0.0;  // No reference available
    }
    
    // Use last available reference
    Eigen::VectorXd x_err = x - x_ref_full_.back();
    double tracking_cost = 0.5 * (x_err.transpose() * Qf_ * x_err)(0, 0);
    
    // Add terminal CoM tracking cost
    double com_cost = 0.0;
    if (w_com_ > 0.0 && !com_ref_full_.empty()) {
        Eigen::Vector3d com_current = computeCoM(x);
        Eigen::Vector3d com_err = com_current - com_ref_full_.back();
        com_cost = 0.5 * w_com_ * com_err.squaredNorm();
    }
    
    // Terminal constraint cost (only joint limits, no control at terminal state)
    double constraint_cost_val = 0.0;
    if (model_) {
        for (int i = 1; i < model_->njnt; ++i) {  // Skip joint 0 (free joint)
            int qpos_idx = model_->jnt_qposadr[i];
            if (qpos_idx >= model_->nq) continue;
            
            double q_val = x(qpos_idx);
            double q_min = model_->jnt_range[i * 2];
            double q_max = model_->jnt_range[i * 2 + 1];
            
            if (std::isfinite(q_min) && std::isfinite(q_max) && q_min < q_max) {
                double margin = 0.1 * (q_max - q_min);
                double q_min_safe = q_min + margin;
                double q_max_safe = q_max - margin;
                
                if (q_val > q_max_safe) {
                    double violation = q_val - q_max_safe;
                    constraint_cost_val += w_joint_limits_ * violation * violation;
                }
                if (q_val < q_min_safe) {
                    double violation = q_min_safe - q_val;
                    constraint_cost_val += w_joint_limits_ * violation * violation;
                }
            }
        }
    }
    
    return tracking_cost + com_cost + constraint_cost_val;
}

void RobotUtils::setCostWeights(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, 
                                const Eigen::MatrixXd& Qf) {
    // Check dimensions before assignment
    if (Q.rows() != nx_ || Q.cols() != nx_) {
        std::cerr << "ERROR: Q matrix dimension mismatch! Expected " << nx_ << "x" << nx_ 
                  << ", got " << Q.rows() << "x" << Q.cols() << std::endl;
        return;
    }
    
    if (R.rows() != nu_ || R.cols() != nu_) {
        std::cerr << "ERROR: R matrix dimension mismatch! Expected " << nu_ << "x" << nu_ 
                  << ", got " << R.rows() << "x" << R.cols() << std::endl;
        return;
    }
    
    if (Qf.rows() != nx_ || Qf.cols() != nx_) {
        std::cerr << "ERROR: Qf matrix dimension mismatch! Expected " << nx_ << "x" << nx_ 
                  << ", got " << Qf.rows() << "x" << Qf.cols() << std::endl;
        return;
    }
    
    Q_ = Q;
    R_ = R;
    Qf_ = Qf;
    
    std::cout << "Cost weights set successfully" << std::endl;
}

bool RobotUtils::loadReferences(const std::string& q_ref_path, const std::string& v_ref_path) {
    // Load position references
    std::ifstream q_file(q_ref_path);
    if (!q_file.is_open()) {
        std::cerr << "Failed to open position reference file: " << q_ref_path << std::endl;
        return false;
    }
    
    // Load velocity references
    std::ifstream v_file(v_ref_path);
    if (!v_file.is_open()) {
        std::cerr << "Failed to open velocity reference file: " << v_ref_path << std::endl;
        return false;
    }
    
    x_ref_full_.clear();
    u_ref_full_.clear();
    com_ref_full_.clear();
    ee_pos_ref_full_.clear();
    ee_vel_ref_full_.clear();
    
    std::string q_line, v_line;
    int line_count = 0;
    
    while (std::getline(q_file, q_line) && std::getline(v_file, v_line)) {
        std::stringstream q_ss(q_line), v_ss(v_line);
        std::vector<double> q_vals, v_vals;
        
        // Parse position values
        std::string val;
        while (std::getline(q_ss, val, ',')) {
            try {
                q_vals.push_back(std::stod(val));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing position value at line " << line_count << ": " << val << std::endl;
                continue;
            }
        }
        
        // Parse velocity values
        while (std::getline(v_ss, val, ',')) {
            try {
                v_vals.push_back(std::stod(val));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing velocity value at line " << line_count << ": " << val << std::endl;
                continue;
            }
        }
        
        // Check dimensions
        if ((int)q_vals.size() != model_->nq || (int)v_vals.size() != model_->nv) {
            std::cerr << "Dimension mismatch at line " << line_count 
                      << ": got q=" << q_vals.size() << " (expected " << model_->nq 
                      << "), v=" << v_vals.size() << " (expected " << model_->nv << ")" << std::endl;
            continue;
        }
        
        // Create state vector [q; v]
        Eigen::VectorXd x_ref(nx_);
        for (int i = 0; i < model_->nq; ++i) x_ref(i) = q_vals[i];
        for (int i = 0; i < model_->nv; ++i) x_ref(model_->nq + i) = v_vals[i];
        
        x_ref_full_.push_back(x_ref);
        
        // zero control reference (will be updated if needed)
        u_ref_full_.push_back(Eigen::VectorXd::Zero(nu_));
        
        // Compute CoM and end-effector references for this state
        mjData* temp_data = mj_makeData(model_);
        for (int i = 0; i < model_->nq; ++i) temp_data->qpos[i] = q_vals[i];
        for (int i = 0; i < model_->nv; ++i) temp_data->qvel[i] = v_vals[i];
        mj_forward(model_, temp_data);
        
        // CoM reference
        Eigen::Vector3d com_ref;
        for (int i = 0; i < 3; ++i) {
            com_ref(i) = temp_data->subtree_com[3 + i];  // Root body CoM is at index 1, skip first 3 elements
        }
        com_ref_full_.push_back(com_ref);
        
        // End-effector position and velocity references (using body positions)
        std::vector<Eigen::Vector3d> ee_pos_refs, ee_vel_refs;
        for (int ee_idx = 0; ee_idx < (int)ee_site_ids_.size(); ++ee_idx) {
            int body_id = ee_site_ids_[ee_idx];
            
            // Position reference (body position)
            Eigen::Vector3d ee_pos;
            for (int i = 0; i < 3; ++i) {
                ee_pos(i) = temp_data->xpos[3 * body_id + i];
            }
            ee_pos_refs.push_back(ee_pos);
            
            // Velocity reference: compute using body Jacobian
            Eigen::MatrixXd jac_pos(3, model_->nv), jac_rot(3, model_->nv);
            mj_jacBody(model_, temp_data, jac_pos.data(), jac_rot.data(), body_id);
            
            Eigen::Vector3d ee_vel = jac_pos * Eigen::Map<const Eigen::VectorXd>(temp_data->qvel, model_->nv);
            ee_vel_refs.push_back(ee_vel);
        }
        ee_pos_ref_full_.push_back(ee_pos_refs);
        ee_vel_ref_full_.push_back(ee_vel_refs);
        
        mj_deleteData(temp_data);
        
        ++line_count;
    }
    
    std::cout << "Loaded " << x_ref_full_.size() << " reference states" << std::endl;
    return !x_ref_full_.empty();
}

void RobotUtils::getReferenceWindow(int t0, int N, 
                                    std::vector<Eigen::VectorXd>& x_ref_window,
                                    std::vector<Eigen::VectorXd>& u_ref_window,
                                    std::vector<Eigen::Vector3d>& com_ref_window) const {
    x_ref_window.clear();
    u_ref_window.clear();
    com_ref_window.clear();
    
    for (int i = 0; i <= N; ++i) {  // N+1 states, N controls, N+1 CoM references
        int ref_idx = std::min(t0 + i, (int)x_ref_full_.size() - 1);
        x_ref_window.push_back(x_ref_full_[ref_idx]);
        
        // Add CoM reference for this timestep
        int com_ref_idx = std::min(t0 + i, (int)com_ref_full_.size() - 1);
        com_ref_window.push_back(com_ref_full_[com_ref_idx]);
        
        if (i < N) {  // Only N controls
            int u_ref_idx = std::min(t0 + i, (int)u_ref_full_.size() - 1);
            u_ref_window.push_back(u_ref_full_[u_ref_idx]);
        }
    }
}

int RobotUtils::jointId(const std::string& name) const {
    auto it = joint_name_to_id_.find(name);
    return (it != joint_name_to_id_.end()) ? it->second : -1;
}

std::string RobotUtils::getEEFrameName(int ee_idx) const {
    if (ee_idx >= (int)ee_site_ids_.size()) {
        throw std::runtime_error("Invalid EE index: " + std::to_string(ee_idx));
    }
    
    int body_id = ee_site_ids_[ee_idx];
    const char* body_name = mj_id2name(model_, mjOBJ_BODY, body_id);
    if (!body_name) {
        throw std::runtime_error("Failed to get body name for EE index: " + std::to_string(ee_idx));
    }
    
    return std::string(body_name);
}

Eigen::Vector3d RobotUtils::getEEReference(int t, int ee_idx) const {
    if (t >= (int)ee_pos_ref_full_.size() || ee_idx >= (int)ee_pos_ref_full_[t].size()) {
        throw std::runtime_error("Invalid reference index: t=" + std::to_string(t) + 
                                ", ee_idx=" + std::to_string(ee_idx));
    }
    
    return ee_pos_ref_full_[t][ee_idx];
}

void RobotUtils::resetToReference(int t) {
    if (t < (int)x_ref_full_.size()) {
        setState(x_ref_full_[t]);
    }
}

void RobotUtils::initializeStandingPose() {
    if (!model_ || !data_) {
        std::cerr << "Model not loaded, cannot initialize standing pose" << std::endl;
        return;
    }
    
    // Reset to default key frame (should be standing pose)
    mj_resetData(model_, data_);
    
    // Set floating base position (assuming first 7 DOFs are free joint: x,y,z,qw,qx,qy,qz)
    for(int i = 0; i < model_-> nq; ++i) {
        if (i == 2) {
            // Position: slightly above ground
            data_->qpos[2] = 1.0432;  // z (height above ground)
        }
        else if(i == 3){
            // Orientation: identity quaternion (no rotation)
            data_->qpos[3] = 1.0;  // qw
        }
        else{
            data_->qpos[i] = 0.0;  // rest of the qs
        }
    }
    
    // Set all joint velocities to zero
    for (int i = 0; i < model_->nv; ++i) {
        data_->qvel[i] = 0.0;
    }
    
    // Improve numerical stability
    // model_->opt.solver = mjSOL_PGS;      // Projected Gauss-Seidel (fast, less accurate)
    model_->opt.cone = mjCONE_ELLIPTIC;     // Elliptic cone (more accurate)
    model_->opt.jacobian = mjJAC_SPARSE;    // Sparse Jacobian
    model_->opt.solver = mjSOL_NEWTON;      // Newton solver for hard contacts
    model_->opt.iterations = 500;           // More solver iterations
    model_->opt.tolerance = 1e-8;           // Tighter tolerance
    
    // Forward kinematics to compute dependent quantities
    mj_forward(model_, data_);
    
    // std::cout << "Robot initialized to standing pose with improved stability settings" << std::endl;
}

// Private helper functions
void RobotUtils::buildJointNameMap() {
    if (!model_) return;
    
    joint_name_to_id_.clear();
    for (int i = 0; i < model_->njnt; ++i) {
        const char* name = mj_id2name(model_, mjOBJ_JOINT, i);
        if (name) {
            joint_name_to_id_[std::string(name)] = i;
        }
    }
    
    // std::cout << "Built joint name mapping for " << joint_name_to_id_.size() << " joints" << std::endl;
}

void RobotUtils::packState(Eigen::VectorXd& x) const {
    // Pack MuJoCo state [qpos; qvel] into Eigen vector
    for (int i = 0; i < model_->nq; ++i) {
        x(i) = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
        x(model_->nq + i) = data_->qvel[i];
    }
}

void RobotUtils::unpackState(const Eigen::VectorXd& x) {
    // Unpack Eigen vector into MuJoCo state [qpos; qvel]
    for (int i = 0; i < model_->nq; ++i) {
        data_->qpos[i] = x(i);
    }
    for (int i = 0; i < model_->nv; ++i) {
        data_->qvel[i] = x(model_->nq + i);
    }
}

void RobotUtils::packControl(Eigen::VectorXd& u) const {
    // Pack MuJoCo controls into Eigen vector
    for (int i = 0; i < model_->nu; ++i) {
        u(i) = data_->ctrl[i];
    }
}

void RobotUtils::unpackControl(const Eigen::VectorXd& u) {
    // Unpack Eigen vector into MuJoCo controls
    for (int i = 0; i < model_->nu; ++i) {
        data_->ctrl[i] = u(i);
    }
}

// ============================================================================
// CONSTRAINT COST FUNCTIONS
// ============================================================================

double RobotUtils::constraintCost(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
    if (!model_) return 0.0;
    
    double constraint_cost = 0.0;
    
    // CONTROL CONSTRAINTS (Torque limits)
    for (int i = 0; i < nu_; ++i) {
        double u_val = u(i);
        // Access 2D array as [i][0] and [i][1]
        double u_min = model_->actuator_ctrlrange[i * 2];     
        double u_max = model_->actuator_ctrlrange[i * 2 + 1]; 
        
        // 10% safety margin to avoid exact boundaries
        double margin = 0.1 * (u_max - u_min);
        double u_min_safe = u_min + margin;
        double u_max_safe = u_max - margin;
        
        // Quadratic penalty for violations
        if (u_val > u_max_safe) {
            double violation = u_val - u_max_safe;
            constraint_cost += w_control_limits_ * violation * violation;
        }
        if (u_val < u_min_safe) {
            double violation = u_min_safe - u_val;
            constraint_cost += w_control_limits_ * violation * violation;
        }
    }
    
    // JOINT POSITION CONSTRAINTS  
    for (int i = 1; i < model_->njnt; ++i) {  // Skip joint 0 (free joint)
        // Map joint index to qpos index
        int qpos_idx = model_->jnt_qposadr[i];
        if (qpos_idx >= model_->nq) continue;  // Safety check
        
        double q_val = x(qpos_idx);
        // Access 2D array as [i][0] and [i][1]
        double q_min = model_->jnt_range[i * 2];     
        double q_max = model_->jnt_range[i * 2 + 1]; 
        
        // Only apply constraints if we have valid finite limits
        if (std::isfinite(q_min) && std::isfinite(q_max) && q_min < q_max) {
            double margin = 0.1 * (q_max - q_min);
            double q_min_safe = q_min + margin;
            double q_max_safe = q_max - margin;
            
            if (q_val > q_max_safe) {
                double violation = q_val - q_max_safe;
                constraint_cost += w_joint_limits_ * violation * violation;
            }
            if (q_val < q_min_safe) {
                double violation = q_min_safe - q_val;
                constraint_cost += w_joint_limits_ * violation * violation;
            }
        }
    }
    
    return constraint_cost;
}

void RobotUtils::setConstraintWeights(double w_joint_limits, double w_control_limits) {
    w_joint_limits_ = w_joint_limits;
    w_control_limits_ = w_control_limits;
    
    std::cout << "Constraint weights set: joint_limits=" << w_joint_limits 
              << ", control_limits=" << w_control_limits << std::endl;
}

void RobotUtils::constraintGradients(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                                    Eigen::VectorXd& grad_x, Eigen::VectorXd& grad_u) const {
    if (!model_) return;
    
    grad_x.setZero(nx_);
    grad_u.setZero(nu_);
    
    // CONTROL CONSTRAINT GRADIENTS
    for (int i = 0; i < nu_; ++i) {
        double u_val = u(i);
        double u_min = model_->actuator_ctrlrange[i * 2];
        double u_max = model_->actuator_ctrlrange[i * 2 + 1];
        
        double margin = 0.1 * (u_max - u_min);
        double u_min_safe = u_min + margin;
        double u_max_safe = u_max - margin;
        
        // ∂J/∂u = 2 * w * violation for quadratic penalty
        if (u_val > u_max_safe) {
            double violation = u_val - u_max_safe;
            grad_u(i) += 2.0 * w_control_limits_ * violation;
        }
        if (u_val < u_min_safe) {
            double violation = u_min_safe - u_val;
            grad_u(i) += -2.0 * w_control_limits_ * violation;
        }
    }
    
    // JOINT POSITION CONSTRAINT GRADIENTS
    for (int i = 1; i < model_->njnt; ++i) {  // Skip joint 0 (free joint)
        int qpos_idx = model_->jnt_qposadr[i];
        if (qpos_idx >= model_->nq) continue;
        
        double q_val = x(qpos_idx);
        double q_min = model_->jnt_range[i * 2];
        double q_max = model_->jnt_range[i * 2 + 1];
        
        if (std::isfinite(q_min) && std::isfinite(q_max) && q_min < q_max) {
            double margin = 0.1 * (q_max - q_min);
            double q_min_safe = q_min + margin;
            double q_max_safe = q_max - margin;
            
            if (q_val > q_max_safe) {
                double violation = q_val - q_max_safe;
                grad_x(qpos_idx) += 2.0 * w_joint_limits_ * violation;
            }
            if (q_val < q_min_safe) {
                double violation = q_min_safe - q_val;
                grad_x(qpos_idx) += -2.0 * w_joint_limits_ * violation;
            }
        }
    }
}

void RobotUtils::constraintHessians(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                                   Eigen::MatrixXd& hess_xx, Eigen::MatrixXd& hess_uu) const {
    if (!model_) return;
    
    hess_xx.setZero(nx_, nx_);
    hess_uu.setZero(nu_, nu_);
    
    // CONTROL CONSTRAINT HESSIANS
    for (int i = 0; i < nu_; ++i) {
        double u_val = u(i);
        double u_min = model_->actuator_ctrlrange[i * 2];
        double u_max = model_->actuator_ctrlrange[i * 2 + 1];
        
        double margin = 0.1 * (u_max - u_min);
        double u_min_safe = u_min + margin;
        double u_max_safe = u_max - margin;
        
        // ∂²J/∂u² = 2 * w for quadratic penalty (only when violating)
        if (u_val > u_max_safe || u_val < u_min_safe) {
            hess_uu(i, i) += 2.0 * w_control_limits_;
        }
    }
    
    // JOINT POSITION CONSTRAINT HESSIANS
    for (int i = 1; i < model_->njnt; ++i) {
        int qpos_idx = model_->jnt_qposadr[i];
        if (qpos_idx >= model_->nq) continue;
        
        double q_val = x(qpos_idx);
        double q_min = model_->jnt_range[i * 2];
        double q_max = model_->jnt_range[i * 2 + 1];
        
        if (std::isfinite(q_min) && std::isfinite(q_max) && q_min < q_max) {
            double margin = 0.1 * (q_max - q_min);
            double q_min_safe = q_min + margin;
            double q_max_safe = q_max - margin;
            
            if (q_val > q_max_safe || q_val < q_min_safe) {
                hess_xx(qpos_idx, qpos_idx) += 2.0 * w_joint_limits_;
            }
        }
    }
}


// DEBUG
void RobotUtils::diagnoseContactForces() const {
    if (!model_ || !data_) return;
    
    std::cout << "\n=== CONTACT DIAGNOSTICS ===" << std::endl;
    std::cout << "Active contacts: " << data_->ncon << std::endl;
    
    // Force MuJoCo to compute contact forces
    mj_rnePostConstraint(model_, data_);
    
    double total_vertical_force = 0;
    for (int i = 0; i < data_->ncon; ++i) {
        // CORRECT: Access actual constraint forces
        double normal_force = data_->efc_force ? data_->efc_force[i] : 0.0;
        
        // Extract contact normal (z-component for vertical force)  
        double contact_normal_z = data_->contact[i].frame[2]; // Z component
        double vertical_force_component = normal_force * contact_normal_z;
        total_vertical_force += vertical_force_component;
        
        std::cout << "Contact " << i << ":" << std::endl;
        std::cout << "  Penetration: " << data_->contact[i].dist << "m" << std::endl;
        std::cout << "  Constraint force: " << normal_force << "N" << std::endl;
        std::cout << "  Vertical component: " << vertical_force_component << "N" << std::endl;
    }
    
    // Rest of diagnostics...
    double robot_mass = 0;
    for (int i = 0; i < model_->nbody; ++i) {
        robot_mass += model_->body_mass[i];
    }

    double required_force = robot_mass * 9.81;
    std::cout << "Total vertical force: " << total_vertical_force << "N" << std::endl;
    std::cout << "Required force: " << required_force << "N" << std::endl;
    std::cout << "Force ratio: " << (total_vertical_force / required_force) << std::endl;
}
/*
void RobotUtils::debugContactSolver() {
    std::cout << "Solver iterations used: " << data_->solver_iter << std::endl;
    if (data_->solver_iter > 0) {
        const mjSolverStat& stat = data_->solver[data_->solver_iter - 1];
        std::cout << "Solver improvement: " << stat.improvement
                  << " gradient: " << stat.gradient << std::endl;
    }
    std::cout << "Contact solver time (total): " << data_->timer[mjTIMER_CONSTRAINT].duration << std::endl;
}
*/

// Utility Functions
void RobotUtils::setGravity(double gx, double gy, double gz) {
    if (model_) {
        model_->opt.gravity[0] = gx;  // X gravity
        model_->opt.gravity[1] = gy;  // Y gravity  
        model_->opt.gravity[2] = gz;  // Z gravity
        std::cout << "Set gravity to: (" << gx << "," << gy << "," << gz << ")m/s²" << std::endl;
    }
}

void RobotUtils::unpackStateToData(const Eigen::VectorXd& x, mjData* target_data) {
    // Unpack state directly to specified data
    for (int i = 0; i < model_->nq; ++i) {
        target_data->qpos[i] = x(i);
    }
    for (int i = 0; i < model_->nv; ++i) {
        target_data->qvel[i] = x(model_->nq + i);
    }
}

void RobotUtils::unpackControlToData(const Eigen::VectorXd& u, mjData* target_data) {
    // Unpack control directly to specified data
    for (int i = 0; i < model_->nu; ++i) {
        target_data->ctrl[i] = u(i);
    }
}

void RobotUtils::packStateFromData(Eigen::VectorXd& x, mjData* source_data) const {
    // Pack state from specified data
    x.resize(nx_);
    for (int i = 0; i < model_->nq; ++i) {
        x(i) = source_data->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
        x(model_->nq + i) = source_data->qvel[i];
    }
}

Eigen::Vector3d RobotUtils::computeCoM(const Eigen::VectorXd& x) const {
    if (!model_ || !data_temp_) return Eigen::Vector3d::Zero();
    
    // Set state in temporary data and compute forward kinematics
    // We need to cast away const to use helper functions
    const_cast<RobotUtils*>(this)->unpackStateToData(x, data_temp_);
    mj_forward(model_, data_temp_);
    
    // Compute CoM using mass-weighted average
    double total_mass = 0.0;
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    
    for (int i = 1; i < model_->nbody; ++i) {  // Skip world body
        double body_mass = model_->body_mass[i];
        if (body_mass > 0) {
            total_mass += body_mass;
            for (int j = 0; j < 3; ++j) {
                com(j) += body_mass * data_temp_->xipos[i * 3 + j];
            }
        }
    }
    
    return (total_mass > 0) ? com / total_mass : com;
}

void RobotUtils::scaleRobotMass(double scale_factor) {
    if (model_) {
        for (int i = 0; i < model_->nbody; ++i) {
            model_->body_mass[i] *= scale_factor;
        }
        std::cout << "Scaled robot mass by factor: " << scale_factor << std::endl;
    }
}

void RobotUtils::computeGravComp(Eigen::VectorXd& ugrav) const {
    if (!model_ || !data_) {
        ugrav.setZero();
        return;
    }
    
    ugrav.resize(nu_);
    
    // Compute forward dynamics to get qfrc_bias (includes gravity + passive forces)
    // Use const_cast to temporarily modify data for computation
    mjData* temp_data = const_cast<mjData*>(data_);
    mj_forward(model_, temp_data);
    
    // Extract gravity compensation torques from qfrc_bias
    for (int i = 0; i < nu_; ++i) {
        // Map actuator index to joint index
        int joint_id = model_->actuator_trnid[i * 2];
        int qpos_addr = model_->jnt_qposadr[joint_id];
        
        // qfrc_bias contains gravity + Coriolis + centrifugal forces
        ugrav(i) = temp_data->qfrc_bias[qpos_addr];
    }
}

// REMOVED: CoM tracking functions

// Get CoM Jacobian w.r.t. joint positions using MuJoCo's built-in function
// REMOVED: CoM and EE tracking functions