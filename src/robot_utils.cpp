// src/robot_utils.cpp
#include "robot_utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

RobotUtils::RobotUtils() 
    : model_(nullptr), data_(nullptr), data_temp_(nullptr),
      nx_(0), nu_(0), dt_(0.01), w_joint_limits_(500.0), w_control_limits_(1000.0) {
}

RobotUtils::~RobotUtils() {
    if (data_temp_) mj_deleteData(data_temp_);
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
}

bool RobotUtils::loadModel(const std::string& xml_path) {
    char error[1024] = {0};
    
    // Load model
    model_ = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    if (!model_) {
        std::cerr << "Failed to load model: " << error << std::endl;
        return false;
    }
    
    // Create data structures
    data_ = mj_makeData(model_);
    data_temp_ = mj_makeData(model_);
    if (!data_ || !data_temp_) {
        std::cerr << "Failed to create MuJoCo data structures" << std::endl;
        return false;
    }
    
    // Set dimensions
    // State: [qpos, qvel] (positions and velocities)
    nx_ = model_->nq + model_->nv;  // Position coords + velocity coords
    nu_ = model_->nu;               // Number of actuators
    dt_ = model_->opt.timestep;
    
    std::cout << "Model loaded successfully:" << std::endl;
    /* std::cout << "  nq (positions): " << model_->nq << std::endl;
    std::cout << "  nv (velocities): " << model_->nv << std::endl;
    std::cout << "  nu (controls): " << model_->nu << std::endl;
    std::cout << "  nx (state dim): " << nx_ << std::endl;
    std::cout << "  timestep: " << dt_ << std::endl; */
    
    // Build joint name mapping
    buildJointNameMap();
    
    // Initialize cost matrices (will be set later)
    Q_ = Eigen::MatrixXd::Identity(nx_, nx_);
    R_ = Eigen::MatrixXd::Identity(nu_, nu_);
    Qf_ = Eigen::MatrixXd::Identity(nx_, nx_);

    
    return true;
}

void RobotUtils::setContactImpratio(double impratio) {
    if (model_) {
        model_->opt.impratio = impratio;
        std::cout << "Set IMPRATIO to: " << impratio << std::endl;
    }
}

void RobotUtils::setTimeStep(double dt) {
    dt_ = dt;
    if (model_) {
        model_->opt.timestep = dt;
        std::cout << "Set timestep to: " << dt << std::endl;
    }
}

void RobotUtils::setState(const Eigen::VectorXd& x) {
    if (!data_ || x.size() != nx_) {
        std::cerr << "Invalid state size: " << x.size() << " (expected " << nx_ << ")" << std::endl;
        return;
    }
    unpackState(x);
}

void RobotUtils::getState(Eigen::VectorXd& x) const {
    if (!data_) return;
    x.resize(nx_);
    packState(x);
}

void RobotUtils::setControl(const Eigen::VectorXd& u) {
    if (!data_ || u.size() != nu_) {
        std::cerr << "Invalid control size: " << u.size() << " (expected " << nu_ << ")" << std::endl;
        return;
    }
    unpackControl(u);
}

void RobotUtils::step() {
    if (!model_ || !data_) return;
    mj_step(model_, data_);
}

/* void RobotUtils::rolloutOneStep(const Eigen::VectorXd& x, const Eigen::VectorXd& u, 
                                Eigen::VectorXd& x_next) {
    if (!model_ || !data_temp_) return;
    
    // Set state and control
    setState(x);
    mj_copyData(data_temp_, model_, data_);  // Save to temp
    
    setControl(u);
    step();
    
    // Get resulting state
    getState(x_next);
    
    // Restore original state
    mj_copyData(data_, model_, data_temp_);
} */

void RobotUtils::rolloutOneStep(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                               Eigen::VectorXd& x_next) {
    if (!model_ || !data_temp_) return;
    
    // Validate input state and control for numerical issues
    if (!x.allFinite()) {
        std::cerr << "ERROR: Non-finite state passed to rolloutOneStep. Using zero state." << std::endl;
        x_next = Eigen::VectorXd::Zero(nx_);
        return;
    }
    
    if (!u.allFinite()) {
        std::cerr << "WARNING: Non-finite control passed to rolloutOneStep. Using zero control." << std::endl;
    }
    
    // Use separate data for rollout predictions
    mj_copyData(data_temp_, model_, data_); // Save current state
    
    // Set state in temporary data with bounds checking
    unpackStateToData(x, data_temp_);
    
    // Clamp control inputs to actuator limits for numerical stability
    Eigen::VectorXd u_clamped = u;
    for (int i = 0; i < nu_; ++i) {
        if (!std::isfinite(u_clamped(i))) {
            u_clamped(i) = 0.0;  // Replace non-finite with zero
        }
        
        // Clamp to actuator bounds if available
        if (i < model_->nu) {
            double u_min = model_->actuator_ctrlrange[i * 2];
            double u_max = model_->actuator_ctrlrange[i * 2 + 1];
            if (std::isfinite(u_min) && std::isfinite(u_max)) {
                u_clamped(i) = std::max(u_min, std::min(u_max, u_clamped(i)));
            }
        }
    }
    
    unpackControlToData(u_clamped, data_temp_);
    
    // CRITICAL: Recompute contacts after manual state setting
    mj_forward(model_, data_temp_);
    
    // Check for warnings after forward computation
    if (data_temp_->warning[mjWARN_BADQACC].number > 0) {
        std::cout << "WARNING: MuJoCo QACC warning detected during rollout. "
                  << "Time: " << data_temp_->time 
                  << ", DOF warnings: " << data_temp_->warning[mjWARN_BADQACC].number << std::endl;
        
        // Log problematic state and control for debugging
        std::cout << "  Problematic state (first 10): ";
        for (int i = 0; i < std::min(10, (int)x.size()); ++i) {
            std::cout << x(i) << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Problematic control (first 5): ";
        for (int i = 0; i < std::min(5, (int)u.size()); ++i) {
            std::cout << u(i) << " ";
        }
        std::cout << std::endl;
        
        // Reset the warning counter for clean slate
        data_temp_->warning[mjWARN_BADQACC].number = 0;
    }
    
    // Store values before physics step for diagnostics
    double kinetic_energy_before = data_temp_->energy[0];
    double potential_energy_before = data_temp_->energy[1];
    
    // Step forward in temporary data
    mj_step(model_, data_temp_);
    
    // Check energy conservation as a stability indicator
    double kinetic_energy_after = data_temp_->energy[0];
    double potential_energy_after = data_temp_->energy[1];
    double total_energy_before = kinetic_energy_before + potential_energy_before;
    double total_energy_after = kinetic_energy_after + potential_energy_after;
    double energy_change = std::abs(total_energy_after - total_energy_before);
    
    // Warn if energy change is suspiciously large (indicating instability)
    if (energy_change > 1000.0) {
        std::cout << "WARNING: Large energy change detected in rollout: " << energy_change 
                  << " J. This may indicate numerical instability." << std::endl;
    }
    
    // Extract result from temporary data
    packStateFromData(x_next, data_temp_);
    
    // Final check for numerical issues in resulting state
    if (!x_next.allFinite()) {
        std::cerr << "ERROR: rolloutOneStep produced non-finite state. Using input state with zero velocity." << std::endl;
        x_next = x;
        // Zero out velocities (second half of state vector)
        for (int i = model_->nq; i < nx_; ++i) {
            x_next(i) = 0.0;
        }
    }
    
    // Check for extreme values that suggest instability
    for (int i = 0; i < x_next.size(); ++i) {
        if (std::abs(x_next(i)) > 1e6) {
            std::cout << "WARNING: Extreme state value detected after rollout: x[" << i 
                      << "] = " << x_next(i) << ". Clamping to reasonable bounds." << std::endl;
            x_next(i) = std::copysign(1e6, x_next(i));  // Preserve sign but limit magnitude
        }
    }
    
    // Don't restore - let main simulation continue naturally
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
        double constraint_cost_val = constraintCost(x, u);
        
        return tracking_cost + constraint_cost_val;
    }
    
    Eigen::VectorXd x_err = x - x_ref_full_[t];
    Eigen::VectorXd u_err = u - u_ref_full_[t];
    
    double tracking_cost = 0.5 * (x_err.transpose() * Q_ * x_err + u_err.transpose() * R_ * u_err)(0, 0);
    double constraint_cost_val = constraintCost(x, u);
    
    return tracking_cost + constraint_cost_val;
}

double RobotUtils::terminalCost(const Eigen::VectorXd& x) const {
    if (x_ref_full_.empty()) {
        return 0.0;  // No reference available
    }
    
    // Use last available reference
    Eigen::VectorXd x_err = x - x_ref_full_.back();
    double tracking_cost = 0.5 * (x_err.transpose() * Qf_ * x_err)(0, 0);
    
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
    
    return tracking_cost + constraint_cost_val;
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
        
        ++line_count;
    }
    
    std::cout << "Loaded " << x_ref_full_.size() << " reference states" << std::endl;
    return !x_ref_full_.empty();
}

void RobotUtils::getReferenceWindow(int t0, int N, 
                                    std::vector<Eigen::VectorXd>& x_ref_window,
                                    std::vector<Eigen::VectorXd>& u_ref_window) const {
    x_ref_window.clear();
    u_ref_window.clear();
    
    for (int i = 0; i <= N; ++i) {  // N+1 states, N controls
        int ref_idx = std::min(t0 + i, (int)x_ref_full_.size() - 1);
        x_ref_window.push_back(x_ref_full_[ref_idx]);
        
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
    
    // Enhanced numerical stability settings for MuJoCo solver
    model_->opt.cone = mjCONE_PYRAMIDAL;    // Friction cone for better convergence
    model_->opt.jacobian = mjJAC_SPARSE;    // Sparse Jacobian for efficiency
    model_->opt.solver = mjSOL_NEWTON;      // Newton solver for hard contacts
    model_->opt.iterations = 1000;          // More solver iterations for convergence
    model_->opt.tolerance = 1e-8;           // Tighter tolerance for better accuracy
    
    // Enhanced integrator settings for numerical stability
    model_->opt.integrator = mjINT_RK4;     // 4th-order Runge-Kutta for accuracy
    model_->opt.collision = mjCOL_ALL;      // Enable all collision detection
    
    // Disable problematic features that can cause instability
    model_->opt.disableflags |= mjDSBL_WARMSTART;  // Disable warm start which can be unstable
    
    // Add constraint solver stability settings
    model_->opt.noslip_iterations = 50;     // More iterations for no-slip constraints
    model_->opt.noslip_tolerance = 1e-8;    // Tighter no-slip tolerance
    model_->opt.mpr_iterations = 50;        // More MPR iterations
    model_->opt.mpr_tolerance = 1e-8;       // Tighter MPR tolerance
    
    // Set reasonable bounds on solver parameters to prevent extreme behavior
    model_->opt.apirate = std::min(model_->opt.apirate, 1000.0);  // Limit API rate
    
    // Forward kinematics to compute dependent quantities
    mj_forward(model_, data_);
    
    // Check for any warnings after initialization
    for (int i = 0; i < mjNWARNING; ++i) {
        if (data_->warning[i].number > 0) {
            std::cout << "WARNING: MuJoCo warning " << i << " occurred " 
                      << data_->warning[i].number << " times during initialization." << std::endl;
            // Clear warning for fresh start
            data_->warning[i].number = 0;
        }
    }
    
    std::cout << "Robot initialized to standing pose with enhanced numerical stability settings" << std::endl;
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


// EXTRAS
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

void RobotUtils::scaleRobotMass(double scale_factor) {
    if (model_) {
        for (int i = 0; i < model_->nbody; ++i) {
            model_->body_mass[i] *= scale_factor;
        }
        std::cout << "Scaled robot mass by factor: " << scale_factor << std::endl;
    }
}

// Numerical stability monitoring and recovery functions
bool RobotUtils::checkNumericalStability() const {
    if (!model_ || !data_) return false;
    
    // Check for any active MuJoCo warnings
    bool has_warnings = false;
    for (int i = 0; i < mjNWARNING; ++i) {
        if (data_->warning[i].number > 0) {
            has_warnings = true;
            if (i == mjWARN_BADQACC) {
                std::cout << "STABILITY WARNING: QACC instability detected (" 
                          << data_->warning[i].number << " instances)" << std::endl;
            }
        }
    }
    
    // Check state values for NaN/Inf
    for (int i = 0; i < model_->nq; ++i) {
        if (!std::isfinite(data_->qpos[i])) {
            std::cout << "STABILITY ERROR: Non-finite position qpos[" << i << "] = " 
                      << data_->qpos[i] << std::endl;
            return false;
        }
    }
    
    for (int i = 0; i < model_->nv; ++i) {
        if (!std::isfinite(data_->qvel[i])) {
            std::cout << "STABILITY ERROR: Non-finite velocity qvel[" << i << "] = " 
                      << data_->qvel[i] << std::endl;
            return false;
        }
        
        if (!std::isfinite(data_->qacc[i])) {
            std::cout << "STABILITY ERROR: Non-finite acceleration qacc[" << i << "] = " 
                      << data_->qacc[i] << std::endl;
            return false;
        }
    }
    
    // Check for extreme velocities that might lead to instability
    for (int i = 0; i < model_->nv; ++i) {
        if (std::abs(data_->qvel[i]) > 100.0) {  // 100 rad/s or m/s is quite extreme
            std::cout << "STABILITY WARNING: Extreme velocity qvel[" << i << "] = " 
                      << data_->qvel[i] << std::endl;
            has_warnings = true;
        }
        
        if (std::abs(data_->qacc[i]) > 1000.0) {  // 1000 rad/s² or m/s² is very extreme
            std::cout << "STABILITY WARNING: Extreme acceleration qacc[" << i << "] = " 
                      << data_->qacc[i] << std::endl;
            has_warnings = true;
        }
    }
    
    // Check energy for reasonable values
    double total_energy = data_->energy[0] + data_->energy[1];  // kinetic + potential
    if (!std::isfinite(total_energy) || std::abs(total_energy) > 1e6) {
        std::cout << "STABILITY WARNING: Extreme total energy = " << total_energy << " J" << std::endl;
        has_warnings = true;
    }
    
    return !has_warnings;  // Return true if no warnings detected
}

void RobotUtils::logNumericalState() const {
    if (!model_ || !data_) return;
    
    std::cout << "\n=== NUMERICAL STATE DIAGNOSTICS ===" << std::endl;
    std::cout << "Time: " << data_->time << std::endl;
    
    // Log warning counts
    for (int i = 0; i < mjNWARNING; ++i) {
        if (data_->warning[i].number > 0) {
            std::cout << "Warning " << i << ": " << data_->warning[i].number << " instances" << std::endl;
        }
    }
    
    // Log energy state
    std::cout << "Kinetic Energy: " << data_->energy[0] << " J" << std::endl;
    std::cout << "Potential Energy: " << data_->energy[1] << " J" << std::endl;
    std::cout << "Total Energy: " << (data_->energy[0] + data_->energy[1]) << " J" << std::endl;
    
    // Log extreme state values
    std::cout << "Position extremes: ";
    double pos_min = 1e10, pos_max = -1e10;
    for (int i = 0; i < model_->nq; ++i) {
        pos_min = std::min(pos_min, data_->qpos[i]);
        pos_max = std::max(pos_max, data_->qpos[i]);
    }
    std::cout << "[" << pos_min << ", " << pos_max << "]" << std::endl;
    
    std::cout << "Velocity extremes: ";
    double vel_min = 1e10, vel_max = -1e10;
    for (int i = 0; i < model_->nv; ++i) {
        vel_min = std::min(vel_min, data_->qvel[i]);
        vel_max = std::max(vel_max, data_->qvel[i]);
    }
    std::cout << "[" << vel_min << ", " << vel_max << "]" << std::endl;
    
    std::cout << "Acceleration extremes: ";
    double acc_min = 1e10, acc_max = -1e10;
    for (int i = 0; i < model_->nv; ++i) {
        acc_min = std::min(acc_min, data_->qacc[i]);
        acc_max = std::max(acc_max, data_->qacc[i]);
    }
    std::cout << "[" << acc_min << ", " << acc_max << "]" << std::endl;
    
    // Log contact information
    std::cout << "Active contacts: " << data_->ncon << std::endl;
    if (data_->ncon > 10) {
        std::cout << "WARNING: Unusually high number of contacts detected!" << std::endl;
    }
    
    std::cout << "==================================\n" << std::endl;
}

bool RobotUtils::recoverFromInstability() {
    if (!model_ || !data_) return false;
    
    std::cout << "ATTEMPTING NUMERICAL RECOVERY..." << std::endl;
    
    bool recovery_needed = false;
    
    // Reset non-finite values to reasonable defaults
    for (int i = 0; i < model_->nq; ++i) {
        if (!std::isfinite(data_->qpos[i])) {
            std::cout << "Resetting non-finite qpos[" << i << "] from " << data_->qpos[i] << " to 0.0" << std::endl;
            data_->qpos[i] = (i == 2) ? 1.0432 : ((i == 3) ? 1.0 : 0.0);  // Special handling for base pose
            recovery_needed = true;
        }
    }
    
    for (int i = 0; i < model_->nv; ++i) {
        if (!std::isfinite(data_->qvel[i])) {
            std::cout << "Resetting non-finite qvel[" << i << "] from " << data_->qvel[i] << " to 0.0" << std::endl;
            data_->qvel[i] = 0.0;
            recovery_needed = true;
        }
        
        // Clamp extreme velocities
        if (std::abs(data_->qvel[i]) > 50.0) {
            std::cout << "Clamping extreme qvel[" << i << "] from " << data_->qvel[i];
            data_->qvel[i] = std::copysign(50.0, data_->qvel[i]);
            std::cout << " to " << data_->qvel[i] << std::endl;
            recovery_needed = true;
        }
    }
    
    // Reset control to zero to prevent further instability
    for (int i = 0; i < model_->nu; ++i) {
        data_->ctrl[i] = 0.0;
    }
    
    if (recovery_needed) {
        // Recompute forward dynamics with recovered state
        mj_forward(model_, data_);
        
        // Clear all warnings
        for (int i = 0; i < mjNWARNING; ++i) {
            data_->warning[i].number = 0;
        }
        
        std::cout << "Numerical recovery completed. System state restored." << std::endl;
        return true;
    }
    
    return false;  // No recovery was needed
}
