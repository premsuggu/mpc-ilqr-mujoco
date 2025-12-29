[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/premsuggu/mpc-ilqr-mujoco)
# Humanoid Model Predictive Control (MPC) with iLQR

A cross-platform implementation of Model Predictive Control for humanoid robots using MuJoCo physics simulation and iterative Linear Quadratic Regulator (iLQR) optimization.

![Humanoid Standing Balance](results/T_h1.gif)

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
```

```bash
# 2. Clone repository
git clone https://github.com/premsuggu/Mujoco-MPC.git
cd Mujoco-MPC
```

```bash
# 3. Create conda environment (installs all C++ and Python dependencies)
conda env create -f environment.yml
conda activate humanoid-mpc
```

```bash
# 4. Build the project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

```bash
# 5. Run MPC simulation
./build/humanoid_mpc
```


### **macOS**

```bash
# 1. Install Xcode Command Line Tools
xcode-select --install
```

```bash
# 2. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

```bash
# 3. Clone repository
git clone https://github.com/premsuggu/Mujoco-MPC.git
cd Mujoco-MPC
```
```bash
# 4. Create conda environment
conda env create -f environment.yml
conda activate humanoid-mpc
```
```bash
# 5. Build the project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```
```bash
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
```
```powershell
# 4. Clone repository
git clone https://github.com/premsuggu/Mujoco-MPC.git
cd Mujoco-MPC
```
```powershell
# 5. Create conda environment
conda env create -f environment.yml
conda activate humanoid-mpc
```
```powershell
# 6. Build the project (using Visual Studio)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS%
```
```powershell
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
Simulation completed in 24728 ms
Average step time: 2472.80 ms
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
MPC_computeControl        10        0.01        0.00        0.00        0.00
MPC_extractReference      10        0.05        0.01        0.00        0.01
MPC_iLQR_solve            10    24720.71     2472.07     1228.29     3067.83
MPC_stepOnce              10    24726.70     2472.67     1228.46     3072.27
MPC_warmStart             10        5.75        0.57        0.13        4.41
iLQR_backwardPass         90      206.22        2.29        2.05        3.67
...

--- Memory Summary ---
Initial:  404.76 MB
Peak:     407.26 MB
Final:    407.26 MB
```

**📈 Interpreting Profiling Results:**

**Timing Breakdown:**
- **MPC_stepOnce**: Total time for one MPC control step (including all iLQR iterations)
- **iLQR_linearization**: Computing dynamics Jacobians A_t, B_t 
  - This is the main bottleneck due to finite difference computations in MuJoCo
- **iLQR_costQuadratics**: Computing Q, R cost matrices
- **iLQR_lineSearch**: Forward rollout to find optimal step size
- **Calls**: Number of times each function was called

**Memory Metrics:**
- **Initial**: RSS (Resident Set Size) at program start after model loading
- **Peak**: Maximum memory usage during simulation
- **Final**: Memory usage at program exit
- **Leaked**: Difference between Final and Initial (small leaks ~2-3 MB are expected)

## 📊 Project Structure

This project implements **two separate MPC pipelines**:
1. **iLQR-based MPC** (`humanoid_mpc`) - Fast gradient-based optimization
2. **NLP-based MPC** (`nlp_mpc`) - Direct trajectory optimization using IPOPT

```
mujoco_mpc/
├── main/
│   ├── humanoid_mpc.cpp          # iLQR pipeline entry point
│   └── main_nlp.cpp              # NLP pipeline entry point
├── include/
│   ├── ilqr/                     # iLQR MPC implementation
│   │   ├── robot_utils.hpp       # MuJoCo wrapper + dynamics
│   │   ├── ilqr.hpp              # iLQR solver algorithm
│   │   ├── mpc.hpp               # MPC orchestrator with warm-start
│   │   ├── derivatives.hpp       # Symbolic differentiation (Pinocchio+CasADi)
│   │   └── config.hpp            # YAML configuration loader
│   └── nlp/                      # NLP MPC implementation (best practices)
│       ├── nlp_config.hpp        # Configuration structs (weights, options)
│       ├── nlp_utils.hpp         # Utilities (CSV I/O, Pinocchio helpers)
│       ├── sym_utils.hpp         # Symbolic expressions (costs, dynamics, constraints)
│       ├── mpc_utils.hpp         # MPC orchestration (References, ContactSchedule, loop)
│       └── nlp_solver.hpp        # IPOPT solver interface
├── src/
│   ├── ilqr/                     # iLQR implementation files
│   │   ├── robot_utils.cpp       # Robot state management + rollout
│   │   ├── ilqr.cpp              # iLQR optimization algorithm
│   │   ├── mpc.cpp               # MPC control loop
│   │   ├── derivatives.cpp       # CoM + end-effector derivatives
│   │   └── config.cpp            # Configuration parser
│   └── nlp/                      # NLP implementation files
│       ├── nlp_utils.cpp         # Utility implementations
│       ├── sym_utils.cpp         # Symbolic expression building
│       ├── nlp_solver.cpp        # IPOPT solver setup and callbacks
│       └── mpc_utils.cpp         # MPC loop orchestration
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
├── config.yaml                   # iLQR pipeline configuration
├── simulate.py                   # 3D MuJoCo visualization script
├── plotter.py                    # Performance analysis plotting
├── environment.yml               # Conda environment definition
├── CMakeLists.txt                # Build configuration (builds both pipelines)
└── README.md                     # This file
```

## 🧮 Algorithm Details

### **iLQR Optimization**
- **Method**: Iterative Linear Quadratic Regulator with line search
- **Iterations**: 1 in most of the cases. 
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

### **Symbolic Derivatives**
- **CoM Jacobian & Hessian**: Pinocchio analytical computation
- **End-effector Jacobian & Hessian**: CasADi automatic differentiation

<details>
<summary><h2>🔬 NLP-Based MPC Pipeline (Previous Implementation)</h2></summary>

The project includes an alternative **NLP-based MPC** implementation that uses direct trajectory optimization with the IPOPT solver. This is a cleaner, more modular architecture following software engineering best practices.

### **Architecture Overview**

The NLP pipeline is organized into separate modules with clear responsibilities:

- **nlp_config.hpp**: Configuration structs (CostWeights, SolverOptions, NLPConfig)
- **nlp_utils**: Reusable utilities (CSV I/O, Pinocchio helpers)
- **sym_utils**: ALL symbolic expressions (costs + dynamics + constraints)
- **mpc_utils**: MPC orchestration (References, ContactSchedule, MPCResults, loop logic)
- **nlp_solver**: The IPOPT interface layer

### **Build & Run NLP Pipeline**

```bash
# Activate conda environment
conda activate humanoid-mpc

# Build NLP executable (already built with main build, but can rebuild separately)
cd build

# Linux/macOS:
make nlp_mpc -j8

# Windows (use cmake --build):
cmake --build . --config Release --target nlp_mpc

# Run NLP MPC
./nlp_mpc           # Linux/macOS
Release/nlp_mpc.exe # Windows
```

**Note for Windows users**: The `make` command doesn't work with Visual Studio's MSBuild system. Always use `cmake --build` instead.

### **NLP Algorithm Details**

**Method**: Direct trajectory optimization using IPOPT (Interior Point Optimizer)
- **Variables**: Full trajectory {x₀, u₀, x₁, u₁, ..., x_N}
- **Constraints**: 
  - Dynamics: x_{t+1} = f(x_t, u_t) using semi-implicit Euler integration
  - Torque limits: τ = M(q)a + C(q,v) + g(q) ≤ τ_max
- **Cost**: Quadratic tracking cost + control effort penalty
- **Solver**: IPOPT with analytical gradients from CasADi

### **Extending NLP Pipeline**

**Add custom cost term** in `sym_utils.cpp`:
```cpp
// Add to buildStageCost() function
casadi::SX custom_cost = /* your expression */;
total_cost += custom_cost;
```

**Add constraint** in `sym_utils.cpp`:
```cpp
// Add to buildDynamicsFunctions()
casadi::SX constraint = /* your constraint expression */;
constraint_fns_["my_constraint"] = casadi::Function("my_constraint", {q, v, u}, {constraint});
```

**Modify solver options** in `nlp_config.hpp`:
```cpp
struct SolverOptions {
    int max_iter = 100;  // Change IPOPT iterations
    double tol = 1e-6;   // Change convergence tolerance
    // ... add more options
};
```

</details>

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


### **Reference Trajectories**

- **Standing pose**: `data/q_standing.csv` - All joints at 0° except base height (Z = 1.0432m for unitree h1)
- **Walking motion** (for future use): `data/q_ref.csv` and `data/v_ref.csv`
- **Others**: Reference trajectories in `.h5` and `.npz` format

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
<!--

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
-->

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
# - Memory should be stable
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