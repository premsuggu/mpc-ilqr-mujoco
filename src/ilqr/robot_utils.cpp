#include "ilqr/robot_utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <limits>

// Define M_PI (for MSVC/Windows compatibility)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

RobotUtils::RobotUtils() 
    : model_(nullptr), data_(nullptr), data_temp_(nullptr),
      nx_(0), nu_(0), dt_(0.01), w_height_(0.0), w_vel_(0.0), w_joint_vel_(0.0),
      w_joint_limits_(500.0), w_control_limits_(1000.0), w_upright_(0.0), w_balance_(0.0),
      w_pelvis_feet_(0.0), w_walk_(0.0), speed_goal_(0.0),
      left_foot_body_name_("foot_left"), right_foot_body_name_("foot_right"),
      pelvis_body_name_("pelvis"), torso_body_name_("torso"),
      waist_lower_body_name_("waist_lower"),
      linearization_epsilon_(1e-4),
      fd_tolerance_(1e-6), fd_mode_(0),
      instability_debug_enabled_(false), instability_qacc_threshold_(1e5),
      instability_debug_count_(0), instability_debug_limit_(20) {
    const char* dbg = std::getenv("MPC_DEBUG_QACC");
    if (dbg && std::string(dbg) != "0") {
        instability_debug_enabled_ = true;
    }
    const char* thr = std::getenv("MPC_DEBUG_QACC_THRESHOLD");
    if (thr) {
        instability_qacc_threshold_ = std::atof(thr);
        if (!(instability_qacc_threshold_ > 0.0)) {
            instability_qacc_threshold_ = 1e5;
        }
    }
    const char* max_logs = std::getenv("MPC_DEBUG_QACC_MAX_LOGS");
    if (max_logs) {
        instability_debug_limit_ = std::atoi(max_logs);
        if (instability_debug_limit_ <= 0) {
            instability_debug_limit_ = 20;
        }
    }
}

void RobotUtils::configureInstabilityDebug(bool enabled, double qacc_threshold, int max_logs) {
    instability_debug_enabled_ = enabled;
    instability_qacc_threshold_ = (qacc_threshold > 0.0) ? qacc_threshold : 1e5;
    instability_debug_limit_ = (max_logs > 0) ? max_logs : 20;
    instability_debug_count_ = 0;
}

void RobotUtils::logQaccInstabilityIfAny(const mjData* source_data, const char* context) {
    if (!instability_debug_enabled_ || !model_ || !source_data) return;
    if (instability_debug_count_ >= instability_debug_limit_) return;

    bool has_nonfinite = false;
    int worst_dof = -1;
    double worst_abs = 0.0;

    for (int i = 0; i < model_->nv; ++i) {
        const double a = source_data->qacc[i];
        if (!std::isfinite(a)) {
            has_nonfinite = true;
            worst_dof = i;
            worst_abs = std::numeric_limits<double>::infinity();
            break;
        }
        const double aa = std::abs(a);
        if (aa > worst_abs) {
            worst_abs = aa;
            worst_dof = i;
        }
    }

    if (!has_nonfinite && worst_abs < instability_qacc_threshold_) return;

    instability_debug_count_++;

    const int dof = std::max(0, worst_dof);
    const int joint_id = (dof < model_->nv) ? model_->dof_jntid[dof] : -1;
    const int qpos_idx = (joint_id >= 0 && joint_id < model_->njnt) ? model_->jnt_qposadr[joint_id] : -1;
    const double qpos_val = (qpos_idx >= 0 && qpos_idx < model_->nq) ? source_data->qpos[qpos_idx] : std::numeric_limits<double>::quiet_NaN();
    const double qvel_val = (dof < model_->nv) ? source_data->qvel[dof] : std::numeric_limits<double>::quiet_NaN();
    const double qacc_val = (dof < model_->nv) ? source_data->qacc[dof] : std::numeric_limits<double>::quiet_NaN();

    double ctrl_min = 0.0;
    double ctrl_max = 0.0;
    if (model_->nu > 0) {
        ctrl_min = source_data->ctrl[0];
        ctrl_max = source_data->ctrl[0];
        for (int i = 1; i < model_->nu; ++i) {
            ctrl_min = std::min(ctrl_min, static_cast<double>(source_data->ctrl[i]));
            ctrl_max = std::max(ctrl_max, static_cast<double>(source_data->ctrl[i]));
        }
    }

    std::cerr << "[INSTABILITY DEBUG] context=" << context
              << " time=" << source_data->time
              << " ncon=" << source_data->ncon
              << " dof=" << dof
              << " qacc=" << qacc_val
              << " qvel=" << qvel_val
              << " qpos=" << qpos_val
              << " ctrl_range=[" << ctrl_min << ", " << ctrl_max << "]"
              << " base_xyz=[" << source_data->qpos[0] << ", " << source_data->qpos[1] << ", " << source_data->qpos[2] << "]"
              << std::endl;

    if (instability_debug_count_ == instability_debug_limit_) {
        std::cerr << "[INSTABILITY DEBUG] reached log limit (" << instability_debug_limit_
                  << "). Increase MPC_DEBUG_QACC_MAX_LOGS if needed." << std::endl;
    }
}

RobotUtils::~RobotUtils() {
    if (data_temp_) mj_deleteData(data_temp_);
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
}

bool RobotUtils::loadModel(const std::string& xml_path,
                            const std::string& left_foot_name,
                            const std::string& right_foot_name) {
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
    
    // Initialize end-effector body IDs using configured names
    ee_site_ids_.clear();
    int left_foot_id = mj_name2id(model_, mjOBJ_BODY, left_foot_name.c_str());
    int right_foot_id = mj_name2id(model_, mjOBJ_BODY, right_foot_name.c_str());
    
    // Validation: Warn if bodies not found
    if (left_foot_id < 0) {
        std::cerr << "WARNING: Left foot body '" << left_foot_name << "' not found in model!" << std::endl;
    } else {
        ee_site_ids_.push_back(left_foot_id);
        std::cout << "  Left foot: '" << left_foot_name << "' (body ID " << left_foot_id << ")" << std::endl;
    }
    
    if (right_foot_id < 0) {
        std::cerr << "WARNING: Right foot body '" << right_foot_name << "' not found in model!" << std::endl;
    } else {
        ee_site_ids_.push_back(right_foot_id);
        std::cout << "  Right foot: '" << right_foot_name << "' (body ID " << right_foot_id << ")" << std::endl;
    }
    
    std::cout << "Found " << ee_site_ids_.size() << " end-effector bodies" << std::endl;
    
    // CRITICAL: Fail if no end-effectors found
    if (ee_site_ids_.empty()) {
        std::cerr << "ERROR: No end-effector bodies found! Check config.yaml ee_feet names." << std::endl;
        return false;
    }
    
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
    unpackStateToData(x, data_);
}

// Get the robot's current state
void RobotUtils::getState(Eigen::VectorXd& x) const {
    if (!data_) return;
    x.resize(nx_);
    packStateFromData(x, data_);
}

// Set the robot's control input (actuator commands)
void RobotUtils::setControl(const Eigen::VectorXd& u) {
    if (!data_ || u.size() != nu_) {
        std::cerr << "Invalid control size: " << u.size() << " (expected " << nu_ << ")" << std::endl;
        return;
    }
    unpackControlToData(u, data_);
}

// Advance the simulation by one step
void RobotUtils::step() {
    if (!model_ || !data_) return;
    mj_step(model_, data_);
    logQaccInstabilityIfAny(data_, "step");
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
    logQaccInstabilityIfAny(data_temp_, "rollout_mj_forward");
    mj_step(model_, data_temp_);
    logQaccInstabilityIfAny(data_temp_, "rollout_mj_step");
    packStateFromData(x_next, data_temp_);
    // No need to restore original state
}


void RobotUtils::linearizeDynamicsFD(const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                                     Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                                     double eps) {
    if (!model_ || !data_ || !data_temp_) return;
    
    // Use configured FD tolerance (DeepMind MJPC: 1e-6)
    if (eps <= 0.0) {
        eps = fd_tolerance_;
    }
    
    A.resize(nx_, nx_);
    B.resize(nx_, nu_);
    
    // Save original state
    mj_copyData(data_temp_, model_, data_);
    
    // Set state and control in mjData
    unpackStateToData(x, data_);
    unpackControlToData(u, data_);
    
    // Compute forward dynamics at current state
    mj_forward(model_, data_);
    logQaccInstabilityIfAny(data_, "linearize_mj_forward");
    
    // Use MuJoCo's built-in finite difference function (DeepMind MJPC approach)
    // This properly handles quaternions and is more efficient than manual loops
    std::vector<double> A_flat(nx_ * nx_);
    std::vector<double> B_flat(nx_ * nu_);
    
    // Call mjd_transitionFD with configured FD mode
    // fd_mode_: 1 = centered differences, 0 = forward differences
    mjd_transitionFD(model_, data_, eps, static_cast<mjtByte>(fd_mode_),
                     A_flat.data(), B_flat.data(), 
                     nullptr, nullptr);  // C, D not needed (sensor derivatives)
    
    // Map MuJoCo's column-major arrays to Eigen matrices (also column-major by default)
    A = Eigen::Map<Eigen::MatrixXd>(A_flat.data(), nx_, nx_);
    B = Eigen::Map<Eigen::MatrixXd>(B_flat.data(), nx_, nu_);
    
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
        
        // Add height tracking cost
        double height_cost = 0.0;
        if (w_height_ > 0.0 && !height_ref_full_.empty()) {
            Eigen::Vector3d com_current = computeCoM(x);
            int height_ref_idx = std::min(t, (int)height_ref_full_.size() - 1);
            Eigen::Vector3d com_err = com_current - height_ref_full_[height_ref_idx];
            height_cost = 0.5 * w_height_ * com_err.squaredNorm();
        }
        
        return tracking_cost + height_cost;  // No soft constraints!
    }
    
    Eigen::VectorXd x_err = x - x_ref_full_[t];
    Eigen::VectorXd u_err = u - u_ref_full_[t];
    
    double tracking_cost = 0.5 * (x_err.transpose() * Q_ * x_err + u_err.transpose() * R_ * u_err)(0, 0);
    
    // Add height tracking cost
    double height_cost = 0.0;
    if (w_height_ > 0.0 && t < (int)height_ref_full_.size()) {
        Eigen::Vector3d com_current = computeCoM(x);
        Eigen::Vector3d com_err = com_current - height_ref_full_[t];
        height_cost = 0.5 * w_height_ * com_err.squaredNorm();
    }
    
    return tracking_cost + height_cost;  // No soft constraints!
}

double RobotUtils::terminalCost(const Eigen::VectorXd& x) const {
    if (x_ref_full_.empty()) {
        return 0.0;  // No reference available
    }
    
    // Use last available reference
    Eigen::VectorXd x_err = x - x_ref_full_.back();
    double tracking_cost = 0.5 * (x_err.transpose() * Qf_ * x_err)(0, 0);
    
    // Add terminal height tracking cost
    double height_cost = 0.0;
    if (w_height_ > 0.0 && !height_ref_full_.empty()) {
        Eigen::Vector3d com_current = computeCoM(x);
        Eigen::Vector3d com_err = com_current - height_ref_full_.back();
        height_cost = 0.5 * w_height_ * com_err.squaredNorm();
    }
    
    return tracking_cost + height_cost;  // No soft constraints!
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
    height_ref_full_.clear();
    com_vel_ref_full_.clear();
    ee_pos_ref_full_.clear();
    ee_vel_ref_full_.clear();
    
    // Temporary storage for all positions (for velocity computation)
    std::vector<std::vector<double>> all_q_vals;
    std::vector<std::vector<double>> all_v_vals;
    
    std::string q_line, v_line;
    int line_count = 0;
    
    // First pass: load all data
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
        
        all_q_vals.push_back(q_vals);
        all_v_vals.push_back(v_vals);
        ++line_count;
    }
    
    if (all_q_vals.empty()) {
        std::cerr << "No valid reference states loaded" << std::endl;
        return false;
    }
    
    // Second pass: compute EE velocities using proper differentiation
    for (size_t t = 0; t < all_q_vals.size(); ++t) {
        const auto& q_vals = all_q_vals[t];
        const auto& v_vals = all_v_vals[t];
        
        // Create state vector [q; v]
        Eigen::VectorXd x_ref(nx_);
        for (int i = 0; i < model_->nq; ++i) x_ref(i) = q_vals[i];
        for (int i = 0; i < model_->nv; ++i) x_ref(model_->nq + i) = v_vals[i];
        
        x_ref_full_.push_back(x_ref);
        
        // Zero control reference
        u_ref_full_.push_back(Eigen::VectorXd::Zero(nu_));
        
        // Compute CoM and end-effector references for this state
        mjData* temp_data = mj_makeData(model_);
        for (int i = 0; i < model_->nq; ++i) temp_data->qpos[i] = q_vals[i];
        for (int i = 0; i < model_->nv; ++i) temp_data->qvel[i] = v_vals[i];
        mj_forward(model_, temp_data);
        
        // CoM reference
        Eigen::Vector3d com_ref;
        for (int i = 0; i < 3; ++i) {
            com_ref(i) = temp_data->subtree_com[3 + i];
        }
        height_ref_full_.push_back(com_ref);
        
        // CoM velocity reference: use Jacobian method (SEPARATE from position)
        std::vector<mjtNum> jac_com(3 * model_->nv);
        mju_zero(jac_com.data(), 3 * model_->nv);
        mj_jacSubtreeCom(model_, temp_data, jac_com.data(), 0); // Body 0 is the root
        
        Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> 
            J_com(jac_com.data(), 3, model_->nv);
        Eigen::VectorXd qvel = Eigen::Map<const Eigen::VectorXd>(temp_data->qvel, model_->nv);
        Eigen::Vector3d com_vel_ref = J_com * qvel;
        com_vel_ref_full_.push_back(com_vel_ref);
        
        // End-effector position and velocity references
        std::vector<Eigen::Vector3d> ee_pos_refs, ee_vel_refs;
        for (int ee_idx = 0; ee_idx < (int)ee_site_ids_.size(); ++ee_idx) {
            int body_id = ee_site_ids_[ee_idx];
            
            // Position reference (body position)
            Eigen::Vector3d ee_pos;
            for (int i = 0; i < 3; ++i) {
                ee_pos(i) = temp_data->xpos[3 * body_id + i];
            }
            ee_pos_refs.push_back(ee_pos);
            
            // Velocity reference: use Jacobian method (accurate and fast)
            Eigen::MatrixXd jac_pos(3, model_->nv), jac_rot(3, model_->nv);
            mj_jacBody(model_, temp_data, jac_pos.data(), jac_rot.data(), body_id);
            
            Eigen::Vector3d ee_vel = jac_pos * Eigen::Map<const Eigen::VectorXd>(temp_data->qvel, model_->nv);
            ee_vel_refs.push_back(ee_vel);
        }
        ee_pos_ref_full_.push_back(ee_pos_refs);
        ee_vel_ref_full_.push_back(ee_vel_refs);
        
        mj_deleteData(temp_data);
    }
    
    std::cout << "Loaded " << x_ref_full_.size() << " reference states" << std::endl;
    return !x_ref_full_.empty();
}

void RobotUtils::getReferenceWindow(int t0, int N, 
                                    std::vector<Eigen::VectorXd>& x_ref_window,
                                    std::vector<Eigen::VectorXd>& u_ref_window,
                                    std::vector<Eigen::Vector3d>& height_ref_window) const {
    x_ref_window.clear();
    u_ref_window.clear();
    height_ref_window.clear();
    
    for (int i = 0; i <= N; ++i) {  // N+1 states, N controls, N+1 CoM references
        int ref_idx = std::min(t0 + i, (int)x_ref_full_.size() - 1);
        x_ref_window.push_back(x_ref_full_[ref_idx]);
        
        // Add height reference for this timestep
        int height_ref_idx = std::min(t0 + i, (int)height_ref_full_.size() - 1);
        height_ref_window.push_back(height_ref_full_[height_ref_idx]);
        
        if (i < N) {  // Only N controls
            int u_ref_idx = std::min(t0 + i, (int)u_ref_full_.size() - 1);
            u_ref_window.push_back(u_ref_full_[u_ref_idx]);
        }
    }
}

bool RobotUtils::loadContactSchedule(const std::string& contact_path) {
    contact_schedule_.clear();
    
    std::ifstream file(contact_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Failed to open contact schedule file: " << contact_path << std::endl;
        return false;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    // Read each timestep
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<int> contacts;
        
        // Parse comma-separated values
        while (std::getline(ss, token, ',')) {
            try {
                contacts.push_back(std::stoi(token));
            } catch (...) {
                std::cerr << "Warning: Failed to parse contact value: " << token << std::endl;
                continue;
            }
        }
        
        // Validate number of end-effectors matches
        if (!contacts.empty() && contacts.size() != ee_site_ids_.size()) {
            std::cerr << "Warning: Contact schedule has " << contacts.size() 
                      << " end-effectors but model has " << ee_site_ids_.size() << std::endl;
        }
        
        if (!contacts.empty()) {
            contact_schedule_.push_back(contacts);
        }
    }
    
    file.close();
    
    std::cout << "Loaded contact schedule: " << contact_schedule_.size() 
              << " timesteps, " << (contact_schedule_.empty() ? 0 : contact_schedule_[0].size()) 
              << " end-effectors" << std::endl;
    
    return !contact_schedule_.empty();
}

bool RobotUtils::isStance(int ee_idx, int t) const {
    // Bounds checking
    if (t < 0 || t >= (int)contact_schedule_.size()) {
        return true;  // Default to stance if no schedule available
    }
    if (ee_idx < 0 || ee_idx >= (int)contact_schedule_[t].size()) {
        return true;  // Default to stance for invalid indices
    }
    
    return contact_schedule_[t][ee_idx] == 1;
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

Eigen::Vector3d RobotUtils::getEEVelReference(int t, int ee_idx) const {
    if (t >= (int)ee_vel_ref_full_.size() || ee_idx >= (int)ee_vel_ref_full_[t].size()) {
        throw std::runtime_error("Invalid velocity reference index: t=" + std::to_string(t) + 
                                ", ee_idx=" + std::to_string(ee_idx));
    }
    
    return ee_vel_ref_full_[t][ee_idx];
}

Eigen::Vector3d RobotUtils::getCoMVelReference(int t) const {
    if (t >= (int)com_vel_ref_full_.size()) {
        throw std::runtime_error("Invalid CoM velocity reference index: t=" + std::to_string(t));
    }
    
    return com_vel_ref_full_[t];
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

// CONSTRAINT COST FUNCTIONS

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
    // Unpack state directly to specified data using Eigen::Map for efficient memory copy
    // This avoids element-by-element loops and uses optimized BLAS operations
    Eigen::Map<Eigen::VectorXd>(target_data->qpos, model_->nq) = x.head(model_->nq);
    Eigen::Map<Eigen::VectorXd>(target_data->qvel, model_->nv) = x.tail(model_->nv);
}

void RobotUtils::unpackControlToData(const Eigen::VectorXd& u, mjData* target_data) {
    // Clamp controls to actuator limits (DeepMind hard constraints)
    for (int i = 0; i < model_->nu; ++i) {
        double u_clamped = std::clamp(u(i), 
                                      model_->actuator_ctrlrange[i * 2],      // min
                                      model_->actuator_ctrlrange[i * 2 + 1]); // max
        target_data->ctrl[i] = u_clamped;
    }
}

void RobotUtils::packStateFromData(Eigen::VectorXd& x, mjData* source_data) const {
    // Pack state from specified data using Eigen::Map for efficient memory copy
    x.resize(nx_);
    x.head(model_->nq) = Eigen::Map<const Eigen::VectorXd>(source_data->qpos, model_->nq);
    x.tail(model_->nv) = Eigen::Map<const Eigen::VectorXd>(source_data->qvel, model_->nv);
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

Eigen::Vector3d RobotUtils::computeEEPos(const Eigen::VectorXd& x, int ee_idx) const {
    if (ee_idx < 0 || ee_idx >= (int)ee_site_ids_.size()) {
        return Eigen::Vector3d::Zero();
    }
    
    // Set state and compute FK
    const_cast<RobotUtils*>(this)->unpackStateToData(x, data_temp_);
    mj_forward(model_, data_temp_);
    
    // Get body position
    int body_id = ee_site_ids_[ee_idx];
    return Eigen::Vector3d(
        data_temp_->xpos[3*body_id + 0],
        data_temp_->xpos[3*body_id + 1],
        data_temp_->xpos[3*body_id + 2]
    );
}

Eigen::Vector3d RobotUtils::computeEEVel(const Eigen::VectorXd& x, int ee_idx) const {
    if (ee_idx < 0 || ee_idx >= (int)ee_site_ids_.size()) {
        return Eigen::Vector3d::Zero();
    }
    
    // Set state and compute velocities
    const_cast<RobotUtils*>(this)->unpackStateToData(x, data_temp_);
    mj_kinematics(model_, data_temp_);
    
    // Get body linear velocity (cvel stores 6D: angular + linear)
    int body_id = ee_site_ids_[ee_idx];
    return Eigen::Vector3d(
        data_temp_->cvel[6*body_id + 3],  // Linear x
        data_temp_->cvel[6*body_id + 4],  // Linear y
        data_temp_->cvel[6*body_id + 5]   // Linear z
    );
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
        int dof_addr = model_->jnt_dofadr[joint_id];
        
        // qfrc_bias contains gravity + Coriolis + centrifugal forces
        if (dof_addr >= 0 && dof_addr < model_->nv) {
            ugrav(i) = temp_data->qfrc_bias[dof_addr];
        } else {
            ugrav(i) = 0.0;
        }
    }
}

// =========================================================================
// Rerun Visualization Helper Functions
// =========================================================================

Eigen::VectorXd RobotUtils::getJointLowerLimits() const {
    if (!model_) return Eigen::VectorXd();
    
    // Return limits for actual joints (skip base which is freejoint with 7 DOF)
    int num_joints = model_->nq - 7;
    Eigen::VectorXd limits(num_joints);
    
    for (int i = 0; i < num_joints; ++i) {
        int qpos_idx = i + 7;  // Skip base (pos:3 + quat:4)
        
        // Find the joint that corresponds to this qpos index
        int jnt_id = -1;
        for (int j = 0; j < model_->njnt; ++j) {
            if (model_->jnt_qposadr[j] == qpos_idx) {
                jnt_id = j;
                break;
            }
        }
        
        if (jnt_id >= 0) {
            limits(i) = model_->jnt_range[jnt_id * 2];  // Lower limit
        } else {
            limits(i) = -M_PI;  // Default if not found
        }
    }
    
    return limits;
}

Eigen::VectorXd RobotUtils::getJointUpperLimits() const {
    if (!model_) return Eigen::VectorXd();
    
    int num_joints = model_->nq - 7;
    Eigen::VectorXd limits(num_joints);
    
    for (int i = 0; i < num_joints; ++i) {
        int qpos_idx = i + 7;
        
        int jnt_id = -1;
        for (int j = 0; j < model_->njnt; ++j) {
            if (model_->jnt_qposadr[j] == qpos_idx) {
                jnt_id = j;
                break;
            }
        }
        
        if (jnt_id >= 0) {
            limits(i) = model_->jnt_range[jnt_id * 2 + 1];  // Upper limit
        } else {
            limits(i) = M_PI;  // Default
        }
    }
    
    return limits;
}

Eigen::VectorXd RobotUtils::getTorqueLimits() const {
    if (!model_) return Eigen::VectorXd();
    
    Eigen::VectorXd limits(nu_);
    
    for (int i = 0; i < nu_; ++i) {
        // actuator_ctrlrange[i*2+1] is the upper control limit (absolute max torque)
        limits(i) = model_->actuator_ctrlrange[i * 2 + 1];
    }
    
    return limits;
}

std::vector<std::string> RobotUtils::getJointNames() const {
    if (!model_) return std::vector<std::string>();
    
    std::vector<std::string> names;
    names.reserve(nu_);
    
    for (int i = 0; i < nu_; ++i) {
        // Get joint ID from actuator transmission
        int joint_id = model_->actuator_trnid[i * 2];
        
        // Get joint name using MuJoCo API
        const char* name_cstr = mj_id2name(model_, mjOBJ_JOINT, joint_id);
        std::string name = name_cstr ? std::string(name_cstr) : ("joint_" + std::to_string(i));
        
        names.push_back(name);
    }
    
    return names;
}

Eigen::Vector3d RobotUtils::computeCoMVelocity(const Eigen::VectorXd& x) const {
    if (!model_ || !data_temp_) return Eigen::Vector3d::Zero();
    
    // Set state and compute kinematics
    const_cast<RobotUtils*>(this)->unpackStateToData(x, data_temp_);
    mj_forward(model_, data_temp_);
    
    // Compute CoM velocity using mass-weighted average
    double total_mass = 0.0;
    Eigen::Vector3d com_vel = Eigen::Vector3d::Zero();
    
    for (int i = 1; i < model_->nbody; ++i) {
        double body_mass = model_->body_mass[i];
        if (body_mass > 0) {
            total_mass += body_mass;
            // cvel is body-centric velocity [angular(3), linear(3)]
            for (int j = 0; j < 3; ++j) {
                com_vel(j) += body_mass * data_temp_->cvel[i * 6 + 3 + j];
            }
        }
    }
    
    return (total_mass > 0) ? com_vel / total_mass : com_vel;
}

std::vector<Eigen::Vector3d> RobotUtils::getEndEffectorPositions() const {
    if (!model_ || !data_) return std::vector<Eigen::Vector3d>();
    
    std::vector<Eigen::Vector3d> positions;
    positions.reserve(ee_site_ids_.size());
    
    for (int site_id : ee_site_ids_) {
        Eigen::Vector3d pos;
        pos.x() = data_->site_xpos[site_id * 3 + 0];
        pos.y() = data_->site_xpos[site_id * 3 + 1];
        pos.z() = data_->site_xpos[site_id * 3 + 2];
        positions.push_back(pos);
    }
    
    return positions;
}

std::vector<bool> RobotUtils::getContactStates(int time_step) const {
    std::vector<bool> contacts(ee_site_ids_.size(), false);
    
    // If we have a contact schedule loaded, use it
    if (!contact_schedule_.empty() && time_step >= 0 && time_step < contact_schedule_.size()) {
        for (size_t i = 0; i < ee_site_ids_.size() && i < contact_schedule_[time_step].size(); ++i) {
            contacts[i] = (contact_schedule_[time_step][i] == 1);
        }
    }
    // Otherwise, detect from actual MuJoCo contacts
    else if (data_) {
        for (int c = 0; c < data_->ncon; ++c) {
            int geom1 = data_->contact[c].geom1;
            int geom2 = data_->contact[c].geom2;
            
            // Check if either geom is attached to an end-effector site
            for (size_t i = 0; i < ee_site_ids_.size(); ++i) {
                int body_id = model_->site_bodyid[ee_site_ids_[i]];
                
                // Check if contact involves this body
                int body1 = model_->geom_bodyid[geom1];
                int body2 = model_->geom_bodyid[geom2];
                
                if (body1 == body_id || body2 == body_id) {
                    contacts[i] = true;
                    break;
                }
            }
        }
    }
    
    return contacts;
}

// World-frame z-position of a named MuJoCo body (for Pelvis/Feet cost evaluation)
double RobotUtils::computeBodyZPos(const Eigen::VectorXd& x, const std::string& body_name) const {
    int body_id = mj_name2id(model_, mjOBJ_BODY, body_name.c_str());
    if (body_id < 0) {
        std::cerr << "WARNING: Body '" << body_name << "' not found in model" << std::endl;
        return 0.0;
    }
    const_cast<RobotUtils*>(this)->unpackStateToData(x, data_temp_);
    mj_kinematics(model_, data_temp_);
    return data_temp_->xpos[3 * body_id + 2];
}

// World-frame x-axis (column 0) of a named body's rotation matrix.
// MuJoCo xmat is row-major 3x3: column 0 = [R[0], R[3], R[6]]
Eigen::Vector3d RobotUtils::computeBodyXAxis(const Eigen::VectorXd& x, const std::string& body_name) const {
    int body_id = mj_name2id(model_, mjOBJ_BODY, body_name.c_str());
    if (body_id < 0) {
        std::cerr << "WARNING: Body '" << body_name << "' not found in model" << std::endl;
        return Eigen::Vector3d(1, 0, 0);  // fallback: world x-axis
    }
    const_cast<RobotUtils*>(this)->unpackStateToData(x, data_temp_);
    mj_kinematics(model_, data_temp_);
    const mjtNum* R = data_temp_->xmat + 9 * body_id;  // row-major 3x3
    return Eigen::Vector3d(R[0], R[3], R[6]);  // column 0 = body x-axis in world frame
}

// World-frame xy subtree CoM linear velocity of a named body.
// DeepMind: "waist_lower_subcomvel" = subtreelinvel sensor on waist_lower body.
// MuJoCo computes data->subtreelinvel[3*body_id : 3*body_id+3] in mj_kinematics.
Eigen::Vector2d RobotUtils::computeSubtreeLinVel2d(const Eigen::VectorXd& x, const std::string& body_name) const {
    int body_id = mj_name2id(model_, mjOBJ_BODY, body_name.c_str());
    if (body_id < 0) {
        std::cerr << "WARNING: Body '" << body_name << "' not found for subtreeLinVel" << std::endl;
        // Fallback: use base linear velocity (torso approx)
        return x.segment(model_->nq, 2);
    }
    const_cast<RobotUtils*>(this)->unpackStateToData(x, data_temp_);
    mj_kinematics(model_, data_temp_);
    // subtree_linvel: world-frame CoM velocity of subtree rooted at body_id
    const mjtNum* v = data_temp_->subtree_linvel + 3 * body_id;
    return Eigen::Vector2d(v[0], v[1]);
}