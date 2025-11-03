#include "nlp/nlp_utils.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace nlp {

Eigen::MatrixXd loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    file.close();
    
    if (data.empty()) {
        throw std::runtime_error("Empty CSV file: " + filename);
    }
    
    int rows = data.size();
    int cols = data[0].size();
    
    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    
    return matrix;
}

void saveTrajectoryCSV(const std::string& filename, 
                       const std::vector<Eigen::VectorXd>& trajectory) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    for (const auto& vec : trajectory) {
        for (int i = 0; i < vec.size(); ++i) {
            file << vec(i);
            if (i < vec.size() - 1) file << ",";
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Saved trajectory to " << filename << std::endl;
}

Eigen::Vector3d computeCoM(const pinocchio::Model& model,
                           pinocchio::Data& data,
                           const Eigen::VectorXd& q) {
    pinocchio::centerOfMass(model, data, q);
    return data.com[0];
}

Eigen::Vector3d computeEEPosition(const pinocchio::Model& model,
                                  pinocchio::Data& data,
                                  const Eigen::VectorXd& q,
                                  const std::string& frame_name) {
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::FrameIndex frame_id = model.getFrameId(frame_name);
    return data.oMf[frame_id].translation();
}

Eigen::VectorXd integrateState(const pinocchio::Model& model,
                                const Eigen::VectorXd& x_current,
                                const Eigen::VectorXd& a,
                                double dt) {
    int nq = model.nq;
    int nv = model.nv;
    
    Eigen::VectorXd q_current = x_current.head(nq);
    Eigen::VectorXd v_current = x_current.tail(nv);
    
    Eigen::VectorXd v_next = v_current + dt * a;
    
    Eigen::VectorXd q_next(nq);
    pinocchio::integrate(model, q_current, v_next * dt, q_next);
    
    Eigen::VectorXd x_next(nq + nv);
    x_next.head(nq) = q_next;
    x_next.tail(nv) = v_next;
    
    return x_next;
}

} // namespace nlp
