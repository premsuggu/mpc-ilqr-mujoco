#!/usr/bin/env python3
"""
Generate trajectories for DeepMind humanoid from CMU keyframes.
Outputs CSV files compatible with our MPC pipeline.
No external dependencies required (stdlib only).
"""

import xml.etree.ElementTree as ET
import os

# Output directory
OUTPUT_DIR = "data/dm_humanoid"

def parse_cmu_keyframes(xml_path):
    """Parse CMU keyframe XML and extract qpos values."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    keyframe_elem = root.find('keyframe')
    if keyframe_elem is None:
        raise ValueError(f"No <keyframe> element found in {xml_path}")
    
    qpos_list = []
    qvel_list = []
    
    for key in keyframe_elem.findall('key'):
        # Get qpos (required)
        qpos_str = key.get('qpos')
        if qpos_str:
            qpos = [float(x) for x in qpos_str.split()]
            qpos_list.append(qpos)
        
        # Get qvel (optional)
        qvel_str = key.get('qvel')
        if qvel_str:
            qvel = [float(x) for x in qvel_str.split()]
            qvel_list.append(qvel)
    
    return qpos_list, qvel_list if qvel_list else None

def save_trajectory(filename, data, description=""):
    """Save trajectory as CSV."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        for row in data:
            f.write(','.join(f'{v:.6f}' for v in row) + '\n')
    print(f"Saved {description}: {filepath} ({len(data)} rows x {len(data[0])} cols)")

def generate_tpose_trajectory(num_frames=501):
    """
    Generate a T-pose trajectory for the DM humanoid.
    T-pose: standing upright with arms extended horizontally.
    
    DM humanoid qpos structure (28 values):
    - Position [0:3]: x, y, z
    - Quaternion [3:7]: qw, qx, qy, qz (MuJoCo convention)
    - Abdomen [7:10]: z, y, x rotations
    - Right leg [10:16]: hip_x, hip_z, hip_y, knee, ankle_y, ankle_x
    - Left leg [16:22]: same
    - Right arm [22:25]: shoulder1, shoulder2, elbow
    - Left arm [25:28]: same
    """
    # T-pose configuration (28 values)
    tpose = [0.0] * 28
    tpose[0:3] = [0.0, 0.0, 1.282]  # Position (standing height)
    tpose[3:7] = [1.0, 0.0, 0.0, 0.0]  # Quaternion (identity = upright)
    # Abdomen: neutral
    tpose[7:10] = [0.0, 0.0, 0.0]
    # Legs: neutral standing
    tpose[10:16] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Right leg
    tpose[16:22] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Left leg
    # Arms extended horizontally (T-pose)
    # shoulder2 controls arm abduction, set to ~-1.5 for horizontal
    tpose[22:25] = [0.0, -1.5, 0.0]  # Right arm
    tpose[25:28] = [0.0, -1.5, 0.0]  # Left arm
    
    # Create trajectory (repeat the same pose)
    return [tpose[:] for _ in range(num_frames)]

def generate_squat_trajectory(num_frames=501):
    """
    Generate a squat pose trajectory from DM humanoid's built-in keyframe.
    """
    # Squat configuration from humanoid.xml (28 values)
    squat = [0.0] * 28
    squat[0:3] = [0.0, 0.0, 0.596]  # Lower height
    squat[3:7] = [0.988015, 0.0, 0.154359, 0.0]  # Slight forward lean
    squat[7:10] = [0.0, 0.4, 0.0]  # Abdomen bent
    squat[10:16] = [-0.25, -0.5, -2.5, -2.65, -0.8, 0.56]  # Right leg
    squat[16:22] = [-0.25, -0.5, -2.5, -2.65, -0.8, 0.56]  # Left leg
    squat[22:28] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Arms neutral
    
    return [squat[:] for _ in range(num_frames)]

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("DM Humanoid Trajectory Generation")
    print("=" * 60)
    
    # 1. Generate T-pose trajectory
    print("\n[1] Generating T-pose trajectory...")
    tpose_traj = generate_tpose_trajectory(num_frames=501)
    save_trajectory("q_tpose.csv", tpose_traj, "T-pose trajectory")
    
    # 2. Generate squat trajectory
    print("\n[2] Generating squat trajectory...")
    squat_traj = generate_squat_trajectory(num_frames=501)
    save_trajectory("q_squat.csv", squat_traj, "Squat trajectory")
    
    # 3. Parse CMU walking keyframes
    print("\n[3] Parsing CMU walking keyframes...")
    cmu_walk_path = "others/mujoco_mpc/mjpc/tasks/humanoid/tracking/keyframes/CMU-CMU-137-137_40_poses.xml"
    
    if os.path.exists(cmu_walk_path):
        try:
            qpos_walk, qvel_walk = parse_cmu_keyframes(cmu_walk_path)
            print(f"  Parsed {len(qpos_walk)} walking keyframes")
            if qpos_walk:
                print(f"  qpos dimension: {len(qpos_walk[0])}")
            
            # Save walking trajectory
            save_trajectory("q_walking.csv", qpos_walk, "CMU walking trajectory")
            
            # Generate zero velocities (CMU data doesn't have consistent qvel)
            nv = 27  # DM humanoid nv
            v_zeros = [[0.0] * nv for _ in range(len(qpos_walk))]
            save_trajectory("v_walking.csv", v_zeros, "Walking velocities (zeros)")
                
        except Exception as e:
            print(f"  Error parsing CMU keyframes: {e}")
    else:
        print(f"  CMU keyframes not found at: {cmu_walk_path}")
    
    # 4. Simpler CMU trajectory
    cmu_simple_path = "others/mujoco_mpc/mjpc/tasks/humanoid/tracking/keyframes/CMU-CMU-02-02_04_poses.xml"
    if os.path.exists(cmu_simple_path):
        print("\n[4] Parsing simpler CMU trajectory...")
        try:
            qpos_simple, _ = parse_cmu_keyframes(cmu_simple_path)
            print(f"  Parsed {len(qpos_simple)} keyframes")
            save_trajectory("q_walk_simple.csv", qpos_simple, "Simple walk trajectory")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Generated trajectories in:", OUTPUT_DIR)
    print("=" * 60)
    
    # Summary
    print("\nFiles created:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.csv'):
            path = os.path.join(OUTPUT_DIR, f)
            with open(path) as fp:
                lines = len(fp.readlines())
            print(f"  - {f} ({lines} rows)")

if __name__ == "__main__":
    main()
