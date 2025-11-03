#include "nlp/nlp_config.hpp"
#include "nlp/nlp_utils.hpp"
#include "nlp/nlp_solver.hpp"
#include "nlp/mpc_utils.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <iostream>

using namespace nlp;

/**
 * @brief Load robot model
 */
pinocchio::Model loadRobotModel(const std::string& urdf_path) {
    std::cout << "Loading robot model..." << std::endl;
    
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model);
    
    std::cout << "Robot model loaded:" << std::endl;
    std::cout << "  nq = " << model.nq << std::endl;
    std::cout << "  nv = " << model.nv << std::endl;
    std::cout << "  nu = " << (model.nv - 6) << "\n" << std::endl;
    
    return model;
}

/**
 * @brief Load reference trajectories from CSV files
 */
References loadReferences(const NLPConfig& config, const pinocchio::Model& model) {
    std::cout << "Loading reference trajectories..." << std::endl;
    
    References refs;
    
    // Load CSV data
    Eigen::MatrixXd q_ref_mat = loadCSV(config.q_ref_path);
    Eigen::MatrixXd v_ref_mat = loadCSV(config.v_ref_path);
    
    int num_samples = q_ref_mat.rows();
    
    // Prepare Pinocchio data for FK/CoM computations
    pinocchio::Data data(model);
    
    // Process each timestep
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd q = q_ref_mat.row(i);
        Eigen::VectorXd v = v_ref_mat.row(i);
        
        refs.q_ref.push_back(q);
        refs.v_ref.push_back(v);
        
        // Compute CoM
        refs.com_ref.push_back(computeCoM(model, data, q));
        
        // Compute EE positions
        std::vector<Eigen::Vector3d> ee_positions;
        for (const auto& ee_name : config.ee_names) {
            ee_positions.push_back(computeEEPosition(model, data, q, ee_name));
        }
        refs.ee_pos_ref.push_back(ee_positions);
    }
    
    std::cout << "References loaded: " << num_samples << " samples\n" << std::endl;
    
    return refs;
}

/**
 * @brief Save MPC results to CSV files
 */
void saveResults(const MPCResults& results, const std::string& output_dir) {
    std::cout << "Saving results to " << output_dir << "..." << std::endl;
    
    saveTrajectoryCSV(output_dir + "nlp_q_optimal.csv", results.q_trajectory);
    saveTrajectoryCSV(output_dir + "nlp_v_optimal.csv", results.v_trajectory);
    saveTrajectoryCSV(output_dir + "nlp_u_optimal.csv", results.u_trajectory);
    
    std::cout << "Results saved successfully!\n" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "==========================================================\n";
    std::cout << "  NLP-based MPC for Humanoid Robot\n";
    std::cout << "  Contact-Implicit Trajectory Optimization\n";
    std::cout << "==========================================================\n\n";
    
    // Load configuration
    NLPConfig config;
    std::cout << "Configuration:\n";
    std::cout << "  Horizon: N = " << config.N << std::endl;
    std::cout << "  Time step: dt = " << config.dt << " s" << std::endl;
    std::cout << "  MPC steps: " << config.num_mpc_steps << std::endl;
    std::cout << "  URDF: " << config.urdf_path << "\n" << std::endl;
    
    // Load robot model
    pinocchio::Model model = loadRobotModel(config.urdf_path);
    
    // Load reference trajectories
    References refs = loadReferences(config, model);
    
    // Create NLP solver
    std::cout << "Creating NLP solver..." << std::endl;
    NLPSolver solver(config, model);
    std::cout << "NLP solver created\n" << std::endl;
    
    // Create MPC orchestrator
    MPCUtils mpc(config, model, refs, solver);
    
    // Run MPC loop
    MPCResults results = mpc.run();
    
    // Save results
    if (results.success) {
        saveResults(results, config.output_dir);
        std::cout << "✓ MPC completed successfully!" << std::endl;
    } else {
        std::cerr << "✗ MPC terminated early (" << results.num_steps_completed 
                  << "/" << config.num_mpc_steps << " steps)" << std::endl;
        return 1;
    }
    
    return 0;
}
