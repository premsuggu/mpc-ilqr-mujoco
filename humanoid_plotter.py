#!/usr/bin/env python3
"""
Humanoid MPC Results Plotter
Plots the essential tracking variables for humanoid locomotion:
- X, Y, Z position tracking
- Orientation (quaternion) tracking  
- Key joint angles (if needed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_humanoid_data():
    """Load humanoid MPC results and reference data"""
    results_dir = "/home/prem/mujoco_mpc/results"
    data_dir = "/home/prem/mujoco_mpc/data"
    
    # Load optimal trajectory (actual)
    q_optimal_path = os.path.join(results_dir, "q_optimal.csv")
    if not os.path.exists(q_optimal_path):
        print(f"Error: {q_optimal_path} not found. Run humanoid MPC first.")
        return None, None, None
    
    # Load reference trajectory  
    q_ref_path = os.path.join(data_dir, "q_standing.csv")
    if not os.path.exists(q_ref_path):
        print(f"Error: {q_ref_path} not found.")
        return None, None, None
    
    # Read actual trajectory
    df_actual = pd.read_csv(q_optimal_path)
    print(f"Loaded actual trajectory: {len(df_actual)} steps")
    
    # Read reference trajectory (no header, just values)
    ref_data = pd.read_csv(q_ref_path, header=None)
    print(f"Loaded reference trajectory: {len(ref_data)} steps")
    
    # Extract time vector from actual data
    if 'time_sec' in df_actual.columns:
        time_actual = df_actual['time_sec'].values
    else:
        # Assume 20Hz MPC (dt=0.02)
        time_actual = np.arange(len(df_actual)) * 0.02
    
    return df_actual, ref_data, time_actual

def plot_humanoid_tracking():
    """Plot humanoid tracking performance focusing on essential variables"""
    
    print("Analyzing humanoid MPC results...")
    df_actual, ref_data, time_actual = load_humanoid_data()
    
    if df_actual is None:
        return
    
    # Create time vector for reference (match the actual data length)
    n_steps = min(len(df_actual), len(ref_data))
    time_ref = time_actual[:n_steps]
    
    # Humanoid state structure (based on H1 robot):
    # q[0:3] = base position (x, y, z)
    # q[3:7] = base quaternion (w, x, y, z) 
    # q[7:] = joint angles
    
    # Extract position data (most critical for locomotion)
    actual_x = df_actual['q_0'].values[:n_steps]
    actual_y = df_actual['q_1'].values[:n_steps] 
    actual_z = df_actual['q_2'].values[:n_steps]
    
    ref_x = ref_data.iloc[:n_steps, 0].values
    ref_y = ref_data.iloc[:n_steps, 1].values  
    ref_z = ref_data.iloc[:n_steps, 2].values
    
    # Extract quaternion data (orientation)
    actual_qw = df_actual['q_3'].values[:n_steps]
    actual_qx = df_actual['q_4'].values[:n_steps]
    actual_qy = df_actual['q_5'].values[:n_steps]
    actual_qz = df_actual['q_6'].values[:n_steps]
    
    ref_qw = ref_data.iloc[:n_steps, 3].values
    ref_qx = ref_data.iloc[:n_steps, 4].values
    ref_qy = ref_data.iloc[:n_steps, 5].values
    ref_qz = ref_data.iloc[:n_steps, 6].values
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Humanoid MPC: Reference vs Actual Trajectory', fontsize=16, fontweight='bold')
    
    # Position tracking (X, Y, Z)
    ax1 = axes[0, 0]
    ax1.plot(time_ref, ref_x, 'b--', linewidth=2, label='Reference X', alpha=0.8)
    ax1.plot(time_actual[:n_steps], actual_x, 'r-', linewidth=2, label='Actual X')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('X Position Tracking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(time_ref, ref_y, 'b--', linewidth=2, label='Reference Y', alpha=0.8)
    ax2.plot(time_actual[:n_steps], actual_y, 'r-', linewidth=2, label='Actual Y')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Y Position Tracking')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(time_ref, ref_z, 'b--', linewidth=2, label='Reference Z', alpha=0.8)
    ax3.plot(time_actual[:n_steps], actual_z, 'r-', linewidth=2, label='Actual Z')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('Z Position (Height) Tracking')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Orientation tracking (quaternion magnitude for simplicity)
    ax4 = axes[1, 1]
    # Plot quaternion w component (most important for upright orientation)
    ax4.plot(time_ref, ref_qw, 'b--', linewidth=2, label='Reference qw', alpha=0.8)
    ax4.plot(time_actual[:n_steps], actual_qw, 'r-', linewidth=2, label='Actual qw')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Quaternion W')
    ax4.set_title('Orientation Tracking (qw)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/home/prem/mujoco_mpc/results/humanoid_tracking_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Humanoid tracking plot saved to: {output_path}")
    
    # Calculate and print tracking errors
    print("\n--- Humanoid Tracking Performance ---")
    
    # Position errors
    x_error = np.abs(actual_x - ref_x)
    y_error = np.abs(actual_y - ref_y) 
    z_error = np.abs(actual_z - ref_z)
    
    print(f"X Position Error - Mean: {np.mean(x_error):.4f}m, Max: {np.max(x_error):.4f}m, RMS: {np.sqrt(np.mean(x_error**2)):.4f}m")
    print(f"Y Position Error - Mean: {np.mean(y_error):.4f}m, Max: {np.max(y_error):.4f}m, RMS: {np.sqrt(np.mean(y_error**2)):.4f}m")
    print(f"Z Position Error - Mean: {np.mean(z_error):.4f}m, Max: {np.max(z_error):.4f}m, RMS: {np.sqrt(np.mean(z_error**2)):.4f}m")
    
    # Overall 3D position error
    pos_error_3d = np.sqrt(x_error**2 + y_error**2 + z_error**2)
    print(f"3D Position Error - Mean: {np.mean(pos_error_3d):.4f}m, Max: {np.max(pos_error_3d):.4f}m, RMS: {np.sqrt(np.mean(pos_error_3d**2)):.4f}m")
    
    # Orientation error (quaternion difference)
    quat_error = np.abs(actual_qw - ref_qw)  # Simplified - just w component
    print(f"Orientation Error (qw) - Mean: {np.mean(quat_error):.4f}, Max: {np.max(quat_error):.4f}, RMS: {np.sqrt(np.mean(quat_error**2)):.4f}")
    
    # Time range info
    print(f"Time range: {time_actual[0]:.3f}s to {time_actual[n_steps-1]:.3f}s")
    print(f"Simulation steps: {n_steps}")
    
    plt.show()

def plot_tracking_errors():
    """Plot tracking errors over time"""
    
    df_actual, ref_data, time_actual = load_humanoid_data()
    if df_actual is None:
        return
    
    n_steps = min(len(df_actual), len(ref_data))
    time_ref = time_actual[:n_steps]
    
    # Calculate position errors
    x_error = df_actual['q_0'].values[:n_steps] - ref_data.iloc[:n_steps, 0].values
    y_error = df_actual['q_1'].values[:n_steps] - ref_data.iloc[:n_steps, 1].values
    z_error = df_actual['q_2'].values[:n_steps] - ref_data.iloc[:n_steps, 2].values
    
    # Plot errors
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle('Humanoid MPC: Tracking Errors Over Time', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(time_ref, x_error, 'r-', linewidth=2)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Error (m)')
    axes[0, 0].set_title('X Position Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_ref, y_error, 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Y Error (m)')
    axes[0, 1].set_title('Y Position Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(time_ref, z_error, 'b-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Z Error (m)')
    axes[1, 0].set_title('Z Position (Height) Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3D position error magnitude
    pos_error_3d = np.sqrt(x_error**2 + y_error**2 + z_error**2)
    axes[1, 1].plot(time_ref, pos_error_3d, 'm-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('3D Error (m)')
    axes[1, 1].set_title('3D Position Error Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save error plot
    error_output_path = "/home/prem/mujoco_mpc/results/humanoid_tracking_errors.png"
    plt.savefig(error_output_path, dpi=300, bbox_inches='tight')
    print(f"Humanoid tracking errors plot saved to: {error_output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Humanoid MPC Results Analysis")
    print("=" * 40)
    
    # Main tracking plot
    plot_humanoid_tracking()
    
    # Error analysis plot  
    plot_tracking_errors()
    
    print("\nAnalysis complete!")
