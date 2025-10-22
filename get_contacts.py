"""
Script to get contact schedule using Mujoco

This script loads the walking csv data and generates the contact_standing/walking.csv.

Usage:
    python get_contacts.py
    python get_contacts.py --input data/q_standing.csv --output data/contact_standing.csv
    python get_contacts.py --input data/q_ref_pinocchio.csv --pinocchio-convention
"""

import sys
import numpy as np
import pandas as pd
import mujoco


def convert_pinocchio_to_mujoco(pinocchio_state):
    """
    Convert Pinocchio convention to MuJoCo convention.
    Pinocchio: [x, y, z, qx, qy, qz, qw, joints...]
    MuJoCo:    [x, y, z, qw, qx, qy, qz, joints...]
    """
    mujoco_state = pinocchio_state.copy()
    
    if pinocchio_state.ndim == 1:
        # Single state
        if len(pinocchio_state) >= 7:
            mujoco_state[3] = pinocchio_state[6]  # qw
            mujoco_state[4] = pinocchio_state[3]  # qx
            mujoco_state[5] = pinocchio_state[4]  # qy
            mujoco_state[6] = pinocchio_state[5]  # qz
    else:
        # Batch of states
        if pinocchio_state.shape[1] >= 7:
            mujoco_state[:, 3] = pinocchio_state[:, 6]  # qw
            mujoco_state[:, 4] = pinocchio_state[:, 3]  # qx
            mujoco_state[:, 5] = pinocchio_state[:, 4]  # qy
            mujoco_state[:, 6] = pinocchio_state[:, 5]  # qz
    
    return mujoco_state

def main():
    # Parse command line arguments
    csv_path = "/home/prem/mujoco_mpc/data/q_standing.csv"
    output_path = "/home/prem/mujoco_mpc/data/contact_standing.csv"
    pinocchio_convention = False  # By default, assume MuJoCo convention
    
    # Simple argument parsing
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            if sys.argv[i] == '--input' and i + 1 < len(sys.argv):
                csv_path = sys.argv[i + 1]
            elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
                output_path = sys.argv[i + 1]
            elif sys.argv[i] in ['--pinocchio-convention', '--pinocchio', '-p']:
                pinocchio_convention = True
            elif sys.argv[i] in ['-h', '--help']:
                print("Usage: python get_contacts.py [--input INPUT.csv] [--output OUTPUT.csv] [--pinocchio-convention]")
                print("\nOptions:")
                print("  --input, -i              Input CSV file path")
                print("  --output, -o             Output CSV file path")
                print("  --pinocchio-convention   Convert from Pinocchio convention (qx,qy,qz,qw) to MuJoCo (qw,qx,qy,qz)")
                print("  --help, -h               Show this help message")
                print("\nDefaults:")
                print("  --input  data/q_ref2.csv")
                print("  --output data/contact_walking.csv")
                print("  Convention: MuJoCo (use --pinocchio-convention if input is in Pinocchio format)")
                return
    
    model_path = "/home/prem/mujoco_mpc/robots/h1_description/mjcf/scene.xml"
    
    # Load model and data
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Load reference trajectory (no header)
    q_ref = pd.read_csv(csv_path, header=None)
    num_timesteps = len(q_ref)
    
    print(f"Loaded {num_timesteps} timesteps from {csv_path}")
    print(f"q_ref shape: {q_ref.shape}")
    print(f"Convention: {'Pinocchio -> MuJoCo conversion enabled' if pinocchio_convention else 'MuJoCo (no conversion)'}")
    
    # Convert from Pinocchio to MuJoCo convention if needed
    if pinocchio_convention:
        print("\nConverting quaternion convention from Pinocchio to MuJoCo...")
        q_ref_values = q_ref.values
        q_ref_converted = convert_pinocchio_to_mujoco(q_ref_values)
        q_ref = pd.DataFrame(q_ref_converted)
        print("  Pinocchio: [x, y, z, qx, qy, qz, qw, joints...]")
        print("  MuJoCo:    [x, y, z, qw, qx, qy, qz, joints...]")
        print("  ✓ Conversion complete")
    
    # Find foot geom IDs by checking ankle bodies
    left_foot_geoms = []
    right_foot_geoms = []
    
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name and 'left_ankle' in body_name:
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    left_foot_geoms.append(geom_id)
        elif body_name and 'right_ankle' in body_name:
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    right_foot_geoms.append(geom_id)
    
    print(f"\nFoot geom IDs:")
    print(f"  Left foot geoms: {left_foot_geoms}")
    print(f"  Right foot geoms: {right_foot_geoms}")
    
    # Initialize contact schedule
    # Format: [left_foot_contact, right_foot_contact] for each timestep
    contact_schedule = np.zeros((num_timesteps, 2), dtype=int)
    
    # Process each timestep
    print(f"\nProcessing {num_timesteps} timesteps...")
    for t in range(num_timesteps):
        # Set configuration
        data.qpos[:] = q_ref.iloc[t].values
        
        # Compute forward kinematics to detect contacts
        mujoco.mj_forward(model, data)
        
        # Check for contacts with left and right foot
        left_contact = False
        right_contact = False
        
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check if either geom is a foot geom
            # Contact distance < 0 means penetration (in contact)
            if contact.dist < 0.001:  # Small positive threshold for contact
                if geom1 in left_foot_geoms or geom2 in left_foot_geoms:
                    left_contact = True
                if geom1 in right_foot_geoms or geom2 in right_foot_geoms:
                    right_contact = True
        
        # Store contact state (1 = in contact, 0 = no contact)
        contact_schedule[t, 0] = 1 if left_contact else 0
        contact_schedule[t, 1] = 1 if right_contact else 0
        
        # Print progress every 50 timesteps
        if (t + 1) % 50 == 0:
            print(f"  Processed {t+1}/{num_timesteps} timesteps")
    
    # Save contact schedule to CSV
    contact_df = pd.DataFrame(contact_schedule, columns=['left_foot', 'right_foot'])
    contact_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Contact schedule saved to {output_path}")
    print(f"  Shape: {contact_df.shape}")
    
    # Print summary statistics
    left_contact_count = np.sum(contact_schedule[:, 0])
    right_contact_count = np.sum(contact_schedule[:, 1])
    both_contact_count = np.sum(np.all(contact_schedule == 1, axis=1))
    no_contact_count = np.sum(np.all(contact_schedule == 0, axis=1))
    
    print(f"\nContact Statistics:")
    print(f"  Left foot in contact:  {left_contact_count}/{num_timesteps} timesteps ({100*left_contact_count/num_timesteps:.1f}%)")
    print(f"  Right foot in contact: {right_contact_count}/{num_timesteps} timesteps ({100*right_contact_count/num_timesteps:.1f}%)")
    print(f"  Both feet in contact:  {both_contact_count}/{num_timesteps} timesteps ({100*both_contact_count/num_timesteps:.1f}%)")
    print(f"  No contact (flight):   {no_contact_count}/{num_timesteps} timesteps ({100*no_contact_count/num_timesteps:.1f}%)")
    
    # Print first few rows as preview
    print(f"\nFirst 10 timesteps of contact schedule:")
    print(contact_df.head(10).to_string(index=True))


if __name__ == "__main__":
    main()