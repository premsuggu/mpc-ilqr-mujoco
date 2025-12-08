#include "rerun/rerun_logger.hpp"
#include <iostream>

RerunLogger::RerunLogger(const std::string& app_name)
    : rec_(app_name), initialized_(false) {
}

RerunLogger::~RerunLogger() {
    if (initialized_) {
        rec_.flush_blocking();
    }
}

bool RerunLogger::initialize() {
    rec_.set_thread_local();
    
    // Always spawn viewer
    auto err = rec_.spawn();
    if (err.is_ok()) {
        std::cout << "[Rerun] Viewer spawned successfully" << std::endl;
        initialized_ = true;
        return true;
    } else {
        std::cerr << "[Rerun] Failed to spawn viewer: " << err.description << std::endl;
        return false;
    }
}

void RerunLogger::setTime(int step, double sim_time_sec) {
    if (!initialized_) return;
    rec_.set_time_sequence("step", step);
    rec_.set_time_seconds("sim_time", sim_time_sec);
}

// === BASE STATE ===
void RerunLogger::logBaseState(
    const Eigen::VectorXd& x_current,
    const Eigen::VectorXd* x_ref,
    int nq) {
    
    if (!initialized_) return;
    using namespace rerun;
    
    // Extract base position and velocity
    double base_x = x_current[0];
    double base_y = x_current[1];
    double base_z = x_current[2];
    double base_vx = x_current[nq + 0];
    double base_vy = x_current[nq + 1];
    double base_vz = x_current[nq + 2];
    
    // Log position: [x, y, z] as actual values
    rec_.log("base/position/x", archetypes::Scalars(base_x));
    rec_.log("base/position/y", archetypes::Scalars(base_y));
    rec_.log("base/position/z", archetypes::Scalars(base_z));
    
    // Log velocity: [vx, vy, vz] as actual values
    rec_.log("base/velocity/vx", archetypes::Scalars(base_vx));
    rec_.log("base/velocity/vy", archetypes::Scalars(base_vy));
    rec_.log("base/velocity/vz", archetypes::Scalars(base_vz));
    
    // Log references with "_ref" suffix for automatic styling
    if (x_ref) {
        rec_.log("base/position/x_ref", archetypes::Scalars((*x_ref)[0]));
        rec_.log("base/position/y_ref", archetypes::Scalars((*x_ref)[1]));
        rec_.log("base/position/z_ref", archetypes::Scalars((*x_ref)[2]));
        
        rec_.log("base/velocity/vx_ref", archetypes::Scalars((*x_ref)[nq + 0]));
        rec_.log("base/velocity/vy_ref", archetypes::Scalars((*x_ref)[nq + 1]));
        rec_.log("base/velocity/vz_ref", archetypes::Scalars((*x_ref)[nq + 2]));
    }
}

// === JOINTS ===
void RerunLogger::logJoints(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& q_lower,
    const Eigen::VectorXd& q_upper,
    const std::vector<std::string>& joint_names,
    const Eigen::VectorXd* q_ref) {
    
    if (!initialized_) return;
    using namespace rerun;
    
    // Skip first 7 (base: pos(3) + quat(4)), log actual joints starting from index 7
    int start_idx = 7;
    
    // Joint grouping based on H1 structure:
    // Left leg: indices 0-4, Right leg: 5-9, Torso: 10, Left arm: 11-14, Right arm: 15-18
    
    for (int i = start_idx; i < q.size(); ++i) {
        int joint_idx = i - start_idx;  // Actuator index (0-18)
        
        if (joint_idx >= joint_names.size()) continue;
        
        std::string full_name = joint_names[joint_idx];
        
        // Determine grouping path based on joint index
        std::string group_path;
        std::string short_name;
        
        if (joint_idx >= 0 && joint_idx <= 4) {
            // Left leg
            group_path = "joints/legs/left/";
            if (full_name.find("hip_yaw") != std::string::npos) short_name = "hip_yaw";
            else if (full_name.find("hip_roll") != std::string::npos) short_name = "hip_roll";
            else if (full_name.find("hip_pitch") != std::string::npos) short_name = "hip_pitch";
            else if (full_name.find("knee") != std::string::npos) short_name = "knee";
            else if (full_name.find("ankle") != std::string::npos) short_name = "ankle";
            else short_name = full_name;
        } 
        else if (joint_idx >= 5 && joint_idx <= 9) {
            // Right leg
            group_path = "joints/legs/right/";
            if (full_name.find("hip_yaw") != std::string::npos) short_name = "hip_yaw";
            else if (full_name.find("hip_roll") != std::string::npos) short_name = "hip_roll";
            else if (full_name.find("hip_pitch") != std::string::npos) short_name = "hip_pitch";
            else if (full_name.find("knee") != std::string::npos) short_name = "knee";
            else if (full_name.find("ankle") != std::string::npos) short_name = "ankle";
            else short_name = full_name;
        }
        else if (joint_idx == 10) {
            // Torso
            group_path = "joints/torso/";
            short_name = "yaw";
        }
        else if (joint_idx >= 11 && joint_idx <= 14) {
            // Left arm
            group_path = "joints/arms/left/";
            if (full_name.find("shoulder_pitch") != std::string::npos) short_name = "shoulder_pitch";
            else if (full_name.find("shoulder_roll") != std::string::npos) short_name = "shoulder_roll";
            else if (full_name.find("shoulder_yaw") != std::string::npos) short_name = "shoulder_yaw";
            else if (full_name.find("elbow") != std::string::npos) short_name = "elbow";
            else short_name = full_name;
        }
        else if (joint_idx >= 15 && joint_idx <= 18) {
            // Right arm
            group_path = "joints/arms/right/";
            if (full_name.find("shoulder_pitch") != std::string::npos) short_name = "shoulder_pitch";
            else if (full_name.find("shoulder_roll") != std::string::npos) short_name = "shoulder_roll";
            else if (full_name.find("shoulder_yaw") != std::string::npos) short_name = "shoulder_yaw";
            else if (full_name.find("elbow") != std::string::npos) short_name = "elbow";
            else short_name = full_name;
        }
        else {
            group_path = "joints/other/";
            short_name = full_name;
        }
        
        // Log joint position (actual value)
        rec_.log(group_path + short_name, archetypes::Scalars(q[i]));
        
        // Log reference position if provided
        if (q_ref && i < q_ref->size()) {
            rec_.log(group_path + short_name + "_ref", archetypes::Scalars((*q_ref)[i]));
        }
        
        // Log limits as static horizontal lines (only once)
        static bool limits_logged = false;
        if (!limits_logged && joint_idx < q_lower.size()) {
            rec_.log_static(group_path + short_name + "_lower_limit", 
                           archetypes::Scalars(q_lower[joint_idx]));
            rec_.log_static(group_path + short_name + "_upper_limit", 
                           archetypes::Scalars(q_upper[joint_idx]));
        }
    }
    
    static bool limits_logged = false;
    limits_logged = true;
}

// === COM ===
void RerunLogger::logCoM(
    const Eigen::Vector3d& com_pos,
    const Eigen::Vector3d& com_vel,
    const Eigen::Vector3d* com_pos_ref,
    const Eigen::Vector3d* com_vel_ref) {
    
    if (!initialized_) return;
    using namespace rerun;
    
    // Log CoM position: [x, y, z] actual
    rec_.log("com/position/x", archetypes::Scalars(com_pos.x()));
    rec_.log("com/position/y", archetypes::Scalars(com_pos.y()));
    rec_.log("com/position/z", archetypes::Scalars(com_pos.z()));
    
    // Log CoM velocity: [vx, vy, vz] actual
    rec_.log("com/velocity/vx", archetypes::Scalars(com_vel.x()));
    rec_.log("com/velocity/vy", archetypes::Scalars(com_vel.y()));
    rec_.log("com/velocity/vz", archetypes::Scalars(com_vel.z()));
    
    // Log references with "_ref" suffix
    if (com_pos_ref) {
        rec_.log("com/position/x_ref", archetypes::Scalars(com_pos_ref->x()));
        rec_.log("com/position/y_ref", archetypes::Scalars(com_pos_ref->y()));
        rec_.log("com/position/z_ref", archetypes::Scalars(com_pos_ref->z()));
    }
    
    if (com_vel_ref) {
        rec_.log("com/velocity/vx_ref", archetypes::Scalars(com_vel_ref->x()));
        rec_.log("com/velocity/vy_ref", archetypes::Scalars(com_vel_ref->y()));
        rec_.log("com/velocity/vz_ref", archetypes::Scalars(com_vel_ref->z()));
    }
}

// === END EFFECTORS ===
void RerunLogger::logEndEffectors(
    const std::vector<Eigen::Vector3d>& ee_positions,
    const std::vector<bool>& contact_states,
    const std::vector<Eigen::Vector3d>* ee_pos_ref) {
    
    if (!initialized_) return;
    using namespace rerun;
    
    std::vector<std::string> ee_names = {"left_foot", "right_foot"};
    
    for (size_t i = 0; i < ee_positions.size() && i < ee_names.size(); ++i) {
        const std::string& name = ee_names[i];
        
        // Log position: [x, y, z] actual
        rec_.log("feet/" + name + "/position/x", archetypes::Scalars(ee_positions[i].x()));
        rec_.log("feet/" + name + "/position/y", archetypes::Scalars(ee_positions[i].y()));
        rec_.log("feet/" + name + "/position/z", archetypes::Scalars(ee_positions[i].z()));
        
        // Log contact state as separate scalar (0/1 time-series)
        rec_.log("feet/contact/" + name, 
                archetypes::Scalars(contact_states[i] ? 1.0 : 0.0));
        
        // Log reference positions with "_ref" suffix
        if (ee_pos_ref && i < ee_pos_ref->size()) {
            rec_.log("feet/" + name + "/position/x_ref", archetypes::Scalars((*ee_pos_ref)[i].x()));
            rec_.log("feet/" + name + "/position/y_ref", archetypes::Scalars((*ee_pos_ref)[i].y()));
            rec_.log("feet/" + name + "/position/z_ref", archetypes::Scalars((*ee_pos_ref)[i].z()));
        }
    }
}

// === TORQUES ===
void RerunLogger::logTorques(
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& u_limits,
    const std::vector<std::string>& joint_names) {
    
    if (!initialized_) return;
    using namespace rerun;
    
    // Torque grouping matches joint grouping:
    // Left leg: 0-4, Right leg: 5-9, Torso: 10, Left arm: 11-14, Right arm: 15-18
    
    for (int i = 0; i < u.size(); ++i) {
        if (i >= joint_names.size()) continue;
        
        std::string full_name = joint_names[i];
        
        // Determine grouping path based on actuator index
        std::string group_path;
        std::string short_name;
        
        if (i >= 0 && i <= 4) {
            // Left leg
            group_path = "torques/legs/left/";
            if (full_name.find("hip_yaw") != std::string::npos) short_name = "hip_yaw";
            else if (full_name.find("hip_roll") != std::string::npos) short_name = "hip_roll";
            else if (full_name.find("hip_pitch") != std::string::npos) short_name = "hip_pitch";
            else if (full_name.find("knee") != std::string::npos) short_name = "knee";
            else if (full_name.find("ankle") != std::string::npos) short_name = "ankle";
            else short_name = full_name;
        } 
        else if (i >= 5 && i <= 9) {
            // Right leg
            group_path = "torques/legs/right/";
            if (full_name.find("hip_yaw") != std::string::npos) short_name = "hip_yaw";
            else if (full_name.find("hip_roll") != std::string::npos) short_name = "hip_roll";
            else if (full_name.find("hip_pitch") != std::string::npos) short_name = "hip_pitch";
            else if (full_name.find("knee") != std::string::npos) short_name = "knee";
            else if (full_name.find("ankle") != std::string::npos) short_name = "ankle";
            else short_name = full_name;
        }
        else if (i == 10) {
            // Torso
            group_path = "torques/torso/";
            short_name = "yaw";
        }
        else if (i >= 11 && i <= 14) {
            // Left arm
            group_path = "torques/arms/left/";
            if (full_name.find("shoulder_pitch") != std::string::npos) short_name = "shoulder_pitch";
            else if (full_name.find("shoulder_roll") != std::string::npos) short_name = "shoulder_roll";
            else if (full_name.find("shoulder_yaw") != std::string::npos) short_name = "shoulder_yaw";
            else if (full_name.find("elbow") != std::string::npos) short_name = "elbow";
            else short_name = full_name;
        }
        else if (i >= 15 && i <= 18) {
            // Right arm
            group_path = "torques/arms/right/";
            if (full_name.find("shoulder_pitch") != std::string::npos) short_name = "shoulder_pitch";
            else if (full_name.find("shoulder_roll") != std::string::npos) short_name = "shoulder_roll";
            else if (full_name.find("shoulder_yaw") != std::string::npos) short_name = "shoulder_yaw";
            else if (full_name.find("elbow") != std::string::npos) short_name = "elbow";
            else short_name = full_name;
        }
        else {
            group_path = "torques/other/";
            short_name = full_name;
        }
        
        // Log torque value
        rec_.log(group_path + short_name, archetypes::Scalars(u[i]));
        
        // Log limits as static horizontal reference lines (only once, both +limit and -limit)
        static bool limits_logged = false;
        if (!limits_logged && i < u_limits.size()) {
            rec_.log_static(group_path + short_name + "_limit_upper", 
                           archetypes::Scalars(u_limits[i]));
            rec_.log_static(group_path + short_name + "_limit_lower", 
                           archetypes::Scalars(-u_limits[i]));
        }
    }
    
    static bool limits_logged = false;
    limits_logged = true;
}
