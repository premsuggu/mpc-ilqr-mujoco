#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <string>
#include <vector>

namespace nlp {

/**
 * @brief Load CSV file into Eigen matrix
 * @param filename Path to CSV file
 * @return Loaded matrix
 */
Eigen::MatrixXd loadCSV(const std::string& filename);

/**
 * @brief Save trajectory to CSV file
 * @param filename Path to output file
 * @param trajectory Vector of state vectors
 */
void saveTrajectoryCSV(const std::string& filename, 
                       const std::vector<Eigen::VectorXd>& trajectory);

/**
 * @brief Compute center of mass position
 * @param model Pinocchio model
 * @param data Pinocchio data
 * @param q Configuration
 * @return CoM position (3D vector)
 */
Eigen::Vector3d computeCoM(const pinocchio::Model& model,
                           pinocchio::Data& data,
                           const Eigen::VectorXd& q);

/**
 * @brief Compute end-effector position
 * @param model Pinocchio model
 * @param data Pinocchio data
 * @param q Configuration
 * @param frame_name Name of end-effector frame
 * @return EE position (3D vector)
 */
Eigen::Vector3d computeEEPosition(const pinocchio::Model& model,
                                  pinocchio::Data& data,
                                  const Eigen::VectorXd& q,
                                  const std::string& frame_name);

/**
 * @brief Integrate state forward using Pinocchio
 * @param model Pinocchio model
 * @param x_current Current state [q; v]
 * @param a Acceleration
 * @param dt Time step
 * @return Next state [q_next; v_next]
 */
Eigen::VectorXd integrateState(const pinocchio::Model& model,
                                const Eigen::VectorXd& x_current,
                                const Eigen::VectorXd& a,
                                double dt);

} // namespace nlp
