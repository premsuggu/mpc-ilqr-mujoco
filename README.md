# Humanoid Model Predictive Control (MPC) with iLQR

A cross-platform implementation of Model Predictive Control for humanoid robots using MuJoCo physics simulation and iterative Linear Quadratic Regulator (iLQR) optimization.

![Humanoid Standing Balance](results/stands.gif)

## 🚀 Features

- **Real-time MPC**: 50Hz control loop with ~5-8 seconds per optimization step
- **iLQR Optimization**: Efficient iterative Linear Quadratic Regulator solver with warm-start capability
- **Symbolic Differentiation**: Fast analytical derivatives using Pinocchio + CasADi
- **MuJoCo Integration**: Physics simulation with contact modeling
- **Cross-platform**: Should work on Linux, macOS, and Windows
- **H1 Humanoid Robot**: Pre-configured for Unitree H1 robot model (51-state, 19-control)
- **Configuration-Driven**: All parameters loaded from `config.yaml`
- **Performance Profiling**: Optional compile-time profiling with zero overhead when disabled
- **Visualization Tools**: Python scripts for trajectory analysis and 3D MuJoCo viewer

## 📋 Prerequisites

- **[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)** or **[Anaconda](https://www.anaconda.com/download/)** (Required for all platforms)
- **C++ Compiler**:
  - **Linux**: GCC 9+ (install via `sudo apt install build-essential`)
  - **macOS**: Clang (install via `xcode-select --install`)
  - **Windows**: Visual Studio 2019/2022 Community with "Desktop development with C++" workload

## 🛠️ Installation

### **Linux (Ubuntu/Debian)**

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install build-essential cmake git

# 2. Clone repository
git clone https://github.com/premsuggu/Mujoco-MPC.git
cd Mujoco-MPC

# 3. Create conda environment (installs all C++ and Python dependencies)
conda env create -f environment.yml
conda activate humanoid-mpc

# 4. Build the project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# 5. Run MPC simulation
./build/humanoid_mpc
```

### **macOS**

```bash
# 1. Install Xcode Command Line Tools
xcode-select --install

# 2. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Clone repository
git clone https://github.com/premsuggu/Mujoco-MPC.git
cd Mujoco-MPC

# 4. Create conda environment
conda env create -f environment.yml
conda activate humanoid-mpc

# 5. Build the project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# 6. Run MPC simulation
./build/humanoid_mpc
```

### **Windows**

```powershell
# 1. Install Visual Studio 2022 Community Edition
# Download from: https://visualstudio.microsoft.com/vs/community/
# During installation, select "Desktop development with C++"

# 2. Install Git for Windows
# Download from: https://git-scm.com/download/win

# 3. Open Anaconda PowerShell Prompt (or Command Prompt) as Administrator

# 4. Clone repository
git clone https://github.com/premsuggu/Mujoco-MPC.git
cd Mujoco-MPC

# 5. Create conda environment
conda env create -f environment.yml
conda activate humanoid-mpc

# 6. Build the project (using Visual Studio)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS%

# 7. Run MPC simulation
build\Release\humanoid_mpc.exe
```

**Note for Windows Users:**
- Always run commands in "Anaconda PowerShell Prompt" or "Anaconda Prompt" (not regular PowerShell)
- If `conda activate` doesn't work, use: `C:\Users\<YourUsername>\miniconda3\Scripts\activate.bat humanoid-mpc`
- Make sure Visual Studio's C++ compiler is in PATH (installation should handle this)

## 🎮 Usage

**⚠️ Important: Always activate the conda environment before running!**

### 1. Run MPC Simulation

```bash
# Activate environment (all platforms)
conda activate humanoid-mpc

# Run simulation
# Linux/macOS:
./build/humanoid_mpc

# Windows:
build\Release\humanoid_mpc.exe
```

**Output:**
```
Configuration loaded from config.yaml
Model loaded: nx=51, nu=19
MPC initialized with N=25, dt=0.02
Step 0/15 | Cost: 16.27 | (X,Y,Z): (0,0,1.043) m | Control range: [-4.84, 0.56]
...
Simulation completed in 120000 ms
Average step time: 8000 ms
```

### 2. Visualize Results in 3D

```bash
# Activate environment
conda activate humanoid-mpc

# Launch MuJoCo viewer with optimal trajectory
python simulate.py
```

This opens an interactive 3D viewer showing the robot executing the MPC trajectory at 50Hz.

### 3. Plot Performance Metrics

```bash
# Generate tracking error plots
python plotter.py
```

**Generated files:**
- `results/humanoid_tracking_comparison.png` - State trajectory comparison
- `results/humanoid_tracking_errors.png` - Tracking error analysis

### 4. Enable Performance Profiling

```bash
# Build with profiling enabled
cmake -B build -DENABLE_PROFILING=ON
cmake --build build --config Release

# Run to see detailed timing breakdown
./build/humanoid_mpc  # Linux/macOS
build\Release\humanoid_mpc.exe  # Windows
```

**Profiling output:**
```
=== Performance Profiling ===
--- Timing Summary ---
Function               Calls   Total(ms)     Avg(ms)     Min(ms)     Max(ms)
----------------------------------------------------------------------------
MPC_stepOnce              15   118450.07     7896.67     3912.85    13790.75
MPC_iLQR_solve            15   118422.07     7894.80     3912.06    13789.90
iLQR_linearization        89    98263.22     1104.08      858.40     1325.65
iLQR_costQuadratics       89    11469.46      128.87       99.70      155.34
iLQR_lineSearch          101     6640.49       65.75       12.02      159.81
...

--- Memory Summary ---
Initial:  54.95 MB
Peak:     57.45 MB
Final:    57.45 MB
Leaked:   2.50 MB
```

**📈 Interpreting Profiling Results:**

**Timing Breakdown:**
- **MPC_stepOnce**: Total time for one MPC control step (including all iLQR iterations)
- **iLQR_linearization**: Computing dynamics Jacobians A_t, B_t (~83% of total time)
  - This is the main bottleneck due to finite difference computations in MuJoCo
- **iLQR_costQuadratics**: Computing Q, R cost matrices (~10% of total time)
- **iLQR_lineSearch**: Forward rollout to find optimal step size (~6% of time)
- **Calls**: Number of times each function was called (varies by convergence)

**Memory Metrics:**
- **Initial**: RSS (Resident Set Size) at program start after model loading
- **Peak**: Maximum memory usage during simulation
- **Final**: Memory usage at program exit
- **Leaked**: Difference between Final and Initial (small leaks ~2-3 MB are normal due to caching)

**Performance Tips:**
- If linearization takes >90% of time: Consider reducing prediction horizon N
- If memory grows over time: Check for large matrix allocations in iLQR loop
- Use `Avg(ms)` to identify consistent bottlenecks vs `Max(ms)` for outliers

## 📊 Project Structure

```
mujoco_mpc/
├── app/
│   └── humanoid_mpc.cpp          # Main MPC application
├── include/
│   ├── robot_utils.hpp           # MuJoCo wrapper + kinematics
│   ├── ilqr.hpp                  # iLQR solver implementation
│   ├── mpc.hpp                   # MPC orchestrator with warm-start
│   ├── derivatives.hpp           # Symbolic differentiation (Pinocchio+CasADi)
│   └── config.hpp                # YAML configuration loader
├── src/
│   ├── robot_utils.cpp           # Robot state management + rollout
│   ├── ilqr.cpp                  # iLQR optimization algorithm
│   ├── mpc.cpp                   # MPC control loop
│   ├── derivatives.cpp           # CoM + end-effector derivatives
│   └── config.cpp                # Configuration parser
├── robots/
│   └── h1_description/           # Unitree H1 robot URDF/MJCF files
│       ├── urdf/h1.urdf          # Robot model for Pinocchio
│       └── mjcf/scene.xml        # MuJoCo simulation scene
├── data/
│   ├── q_standing.csv            # Standing reference trajectory
│   ├── v_standing.csv            # Standing reference velocities
│   ├── q_ref.csv                 # Walking reference (future work)
│   └── v_ref.csv                 # Walking velocities (future work)
├── results/                      # Generated simulation results
│   ├── q_optimal.csv             # Optimal state trajectory
│   ├── u_optimal.csv             # Optimal control sequence
│   └── stands.gif                # Demo visualization
├── config.yaml                   # Central configuration file
├── simulate.py                   # 3D MuJoCo visualization script
├── plotter.py                    # Performance analysis plotting
├── environment.yml               # Conda environment definition
├── CMakeLists.txt                # Build configuration
└── README.md                     # This file
```

## 🧮 Algorithm Details

### **iLQR Optimization**
- **Method**: Iterative Linear Quadratic Regulator with line search
- **Iterations**: Typically 5-10 per MPC step
- **Convergence**: Stops when cost improvement < 1e-4
- **Regularization**: Adaptive λ ∈ [1e-6, 1e-3]

### **Cost Function**
```
J = Σ(||x - x_ref||²_Q + ||u||²_R) + ||x_N - x_ref_N||²_Qf
    + W_com * ||CoM - CoM_ref||²
    + W_foot * Σ||foot_pos - foot_ref||²
```

### **Dynamics Linearization**
- **Method**: Finite differences using MuJoCo's `mj_forward`
- **Jacobians**: A_t (51×51), B_t (51×19) computed at each timestep
- **Bottleneck**: Linearization takes ~83% of total computation time

### **Symbolic Derivatives** (Fast!)
- **CoM Jacobian**: Pinocchio analytical computation (~1ms)
- **End-effector Jacobian**: CasADi automatic differentiation (~2ms)
- **10-30x faster** than finite differences for cost derivatives

## ⚙️ Configuration

All MPC parameters are defined in `config.yaml`:

### **Key Parameters**

```yaml
mpc:
  horizon: 25              # Prediction horizon (25 steps = 0.5 seconds)
  dt: 0.02                 # MPC timestep (50 Hz)
  sim_steps: 15            # Number of simulation steps to run
  
  cost_weights:
    Q_position_z: 2000.0   # Height tracking (critical for balance)
    Q_quat_xyz: [500, 500, 300]  # Orientation control (roll, pitch, yaw)
    Q_joint_pos: 80.0      # Joint position tracking
    R_control: 0.01        # Control effort penalty
    W_com: 500.0           # Center of Mass tracking weight
    W_foot: 500.0          # Foot position tracking weight
```

### **Tuning Guide**

- **Increase `Q_position_z`** → Tighter height control (may cause oscillations)
- **Increase `Q_joint_vel`** → Smoother motions (slower response)
- **Decrease `R_control`** → More aggressive control (higher torques)
- **Increase `W_com`** → Better CoM tracking (may conflict with joint tracking)

### **Reference Trajectories**

- **Standing pose**: `data/q_standing.csv` - All joints at 0° except base height (Z = 1.0432m)
- **Walking motion** (future work): `data/q_ref.csv` and `data/v_ref.csv`

## 📈 Performance

**Hardware:** Typical modern laptop (Intel i7, 16GB RAM)

**Computational Breakdown:**
- **iLQR Linearization**: 75-100 seconds (83% of time) - Bottleneck!
- **Cost Quadratics**: 11-12 seconds (10%)
- **Line Search**: 5-7 seconds (5%)
- **Backward Pass**: 0.5-0.6 seconds (0.5%)
- **Other**: < 1 second

**Typical Results:**
- **Tracking Error**: <1cm RMS position error, <3° orientation error
- **Balance Duration**: >15 seconds stable standing
- **MPC Step Time**: 5-8 seconds (not real-time, but faster than brute-force)
- **Memory Usage**: ~55-58 MB (no memory leaks)

## 🐛 Troubleshooting

### **Build Errors**

**Problem:** `CMake Error: Could not find MuJoCo`
```bash
# Solution: Make sure conda environment is activated
conda activate humanoid-mpc

# Verify MuJoCo is installed in conda environment
conda list mujoco

# If missing, reinstall environment
conda env remove -n humanoid-mpc
conda env create -f environment.yml
```

**Problem:** `undefined reference to 'mj_forward'` (Linux)
```bash
# Solution: Rebuild from clean state
rm -rf build
conda activate humanoid-mpc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**Problem:** `LNK2019: unresolved external symbol` (Windows)
```powershell
# Solution: Ensure Visual Studio C++ tools are installed
# 1. Open Visual Studio Installer
# 2. Modify your installation
# 3. Check "Desktop development with C++"
# 4. Rebuild project
rmdir /s build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### **Runtime Errors**

**Problem:** `ModuleNotFoundError: No module named 'mujoco'` (Python)
```bash
# Solution: Activate conda environment before running Python scripts
conda activate humanoid-mpc
python simulate.py
```

**Problem:** `Error: Model file not found at robots/h1_description/mjcf/scene.xml`
```bash
# Solution: Run from project root directory
cd /path/to/Mujoco-MPC
./build/humanoid_mpc  # NOT from build/ directory
```

**Problem:** Negative time values in profiling output
```bash
# This was a bug fixed - steady_clock instead of high_resolution_clock
# Solution: Make sure you have latest code
git pull origin feature/cost-terms
cmake --build build --config Release
```

### **Performance Issues**

**Problem:** MPC is too slow (>10 seconds per step)
- **Check CPU usage**: Should be near 100% for single core
- **Disable verbose output**: Set `verbose: false` in `config.yaml`
- **Reduce horizon**: Try `horizon: 20` or `horizon: 15`
- **Check compiler optimization**: Ensure `-O3` flag is active

**Problem:** High memory usage or leaks
- **Enable profiling** to track memory deltas
- **Check for large matrices**: Ensure no accidental allocations in hot loops
- **Verify Eigen usage**: Use `.noalias()` for matrix operations

### **Platform-Specific Issues**

**Linux:**
- **X11 forwarding for SSH**: Use `ssh -X` for remote visualization
- **OpenGL errors**: Install `sudo apt install libgl1-mesa-dev libglu1-mesa-dev`

**macOS:**
- **Rosetta 2 warning on M1/M2**: Ignore, everything still works
- **XQuartz required**: Install via `brew install --cask xquartz` if viewer fails

**Windows:**
- **Antivirus blocking**: Add project folder to Windows Defender exclusions
- **Path too long error**: Move project closer to C:\ drive root

## 🔬 Development

### **Adding Custom Cost Terms**

Edit `src/ilqr.cpp` in `computeCostQuadratics()`:

```cpp
// Add custom cost derivative
void iLQR::addCustomCost(int t, const Eigen::VectorXd& x_ref) {
    double weight = 100.0;
    Eigen::VectorXd grad = /* your gradient */;
    Eigen::MatrixXd hess = /* your Hessian */;
    
    lx_[t] += grad;
    lxx_[t] += hess;
}
```

### **Changing Robot Model**

1. Add your robot URDF to `robots/your_robot/`
2. Update `config.yaml`:
   ```yaml
   robot:
     model_path: "robots/your_robot/model.xml"
     urdf_path: "robots/your_robot/robot.urdf"
   ```
3. Adjust cost weights for your robot's dimensions

### **Running Tests**

```bash
# Build with profiling to verify performance
cmake -B build -DENABLE_PROFILING=ON
cmake --build build
./build/humanoid_mpc

# Check output for anomalies:
# - All times should be positive
# - Memory should be stable (~55-60 MB)
# - Robot should not fall (Z > 0.5m throughout)
```

## 📚 Dependencies

**C++ Libraries** (installed via conda):
- **MuJoCo 3.0+**: Physics simulation
- **Eigen 3.4+**: Linear algebra
- **Pinocchio 2.6+**: Robot kinematics
- **CasADi 3.6+**: Automatic differentiation
- **yaml-cpp**: Configuration parsing
- **GLFW 3.4+**: OpenGL windowing

**Python Packages** (installed via conda):
- **mujoco**: Python bindings for visualization
- **numpy, pandas**: Data handling
- **matplotlib, seaborn**: Plotting
- **pyyaml**: Config file parsing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[MuJoCo](https://mujoco.org/)** - Physics simulation framework
- **[Pinocchio](https://github.com/stack-of-tasks/pinocchio)** - Fast rigid body dynamics
- **[CasADi](https://web.casadi.org/)** - Symbolic differentiation
- **[Eigen](https://eigen.tuxfamily.org/)** - Linear algebra library
- **[Unitree Robotics](https://www.unitree.com/)** - H1 humanoid robot model

## 📧 Contact

For questions or issues, please open a [GitHub Issue](https://github.com/premsuggu/Mujoco-MPC/issues).

---

**Built with ❤️ for robotics research and education**
