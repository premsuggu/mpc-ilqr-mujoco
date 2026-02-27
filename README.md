[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/premsuggu/mpc-ilqr-mujoco)
# Humanoid Model Predictive Control (MPC) with iLQR

A cross-platform implementation of Model Predictive Control for humanoid robots using MuJoCo physics simulation and iterative Linear Quadratic Regulator (iLQR) optimization.

![Humanoid Standing Balance](results/T_h1.gif)

## 🚀 Features

- **Real-time MPC**: 50Hz control loop with ~5-8 seconds per optimization step
- **iLQR Optimization**: Efficient iterative Linear Quadratic Regulator solver with warm-start capability
- **Log-Scale Line Search**: Dynamic alpha generation (DeepMind MJPC style) for aggressive stepping
- **Robust Cost Functions**: 6 configurable norm types (Quadratic, L22, L2, Cosh, SmoothAbs2Loss, Rectify)
- **Symbolic Differentiation**: Fast analytical derivatives using Pinocchio + CasADi
- **MuJoCo Integration**: Physics simulation with contact modeling
- **Cross-platform**: Works on Linux, macOS, and Windows
- **Multi-Robot Support**: Model-agnostic design supporting multiple humanoid models (H1, DeepMind Humanoid, etc.)
- **Configuration-Driven**: All parameters and robot-specific settings loaded from `config.yaml`
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
  - Uses symbolic differentiation (CasADi + Pinocchio) for analytical derivatives
- **iLQR_costQuadratics**: Computing Q, R cost matrices with norm derivatives
- **iLQR_backwardPass**: Computing feedback gains via Riccati recursion
- **iLQR_lineSearch**: Forward rollout to find optimal step size (log-scale alphas)
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
│   │   ├── cost.hpp              # Residual-based cost function interface
│   │   ├── norm.hpp              # Robust norm types (6 types: Quadratic, L22, L2, Cosh, SmoothAbs2Loss, Rectify)
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
│   │   ├── cost.cpp              # Cost residual computation (8 cost terms)
│   │   ├── norm.cpp              # Norm function implementations (numerical + symbolic)
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
- **Method**: Iterative Linear Quadratic Regulator with log-scale line search
- **Iterations**: 1 in most cases (warm-start from previous solution)
- **Convergence**: Stops when cost improvement < 1e-4
- **Regularization**: Adaptive λ ∈ [1e-6, 1e6] (control regularization on Quu)
- **Line Search**: Log-scale from 1.0 to min_step (default: 10 steps, min=1e-3)
  - Alphas generated dynamically: α = exp(log(min) + i·Δ) for i = 0...N-1
  - Tried in descending order (1.0 → min) for aggressive stepping
  - First improvement accepted (Armijo backtracking)

### **Cost Function**
Residual-based cost formulation with configurable robust norms:
```
J = Σ_t [ W_state·ρ(r_state) + W_ctrl·ρ(r_ctrl) 
        + W_com_pos·ρ(r_com_pos) + W_com_vel·ρ(r_com_vel)
        + W_ee_pos·ρ(r_ee_pos) + W_ee_vel·ρ(r_ee_vel)
        + W_upright·ρ(r_upright) + W_balance·ρ(r_balance) ]
```

**Supported Norm Types** (ρ: ℝⁿ → ℝ):
- **Quadratic**: `0.5 · rᵀr` (standard least squares)
- **L22**: `p² · (√(rᵀr/p² + 1) - 1)` (smoothed L2)
- **L2**: `p · √(rᵀr)` (linear growth for outliers)
- **Cosh**: `p² · (cosh(√(rᵀr)/p) - 1)` (smooth near zero)
- **SmoothAbs2Loss**: Smooth transition from quadratic to linear
- **Rectify**: `p · max(0, r)` (one-sided penalty)

Each cost term has configurable norm type and parameters (p, q) via `config.yaml`.

### **Dynamics Linearization**
- **Method**: Symbolic differentiation with CasADi + Pinocchio
- **Jacobians**: A_t (51×51), B_t (51×19) computed analytically
- **Cost Derivatives**: Symbolic computation of gradients and Hessians
  - State/Control costs: Analytical derivatives through norm functions
  - CoM costs: Pinocchio for CoM Jacobian/Hessian w.r.t. q
  - End-effector costs: CasADi automatic differentiation
- **Advantage**: More accurate and faster than finite differences

### **Symbolic Derivatives Architecture**
The `symDerivatives` class provides analytical derivatives for all cost terms:
- **CoM Derivatives**: Pinocchio's `getJacobianComFromRootJoint()` and `computeJointKinematicHessians()`
- **End-Effector Derivatives**: CasADi symbolic expressions with automatic differentiation
- **Norm Derivatives**: Analytical gradients/Hessians for all 6 robust norm types
- **Cost Linearization**: Symbolic lx, lu, lxx, luu, lux computed without numerical approximation

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
  horizon: 23              # Prediction horizon (23 steps = 0.46 seconds at 50 Hz)
  dt: 0.02                 # MPC timestep (50 Hz)
  sim_steps: 50            # Number of simulation steps to run
  
  cost_weights:
    # State tracking weights (exact DeepMind MJPC values)
    Q_position_z: 5.0      # Height tracking
    Q_quat_xyz: [5.0, 5.0, 5.0]  # Orientation control
    Q_joint_pos: 0.025     # Joint posture (small regularization)
    R_control: 0.1         # Control effort penalty
    W_com_pos: 5.0         # CoM position/balance (capture point)
    W_com_vel: 0.625       # CoM velocity tracking
    W_foot: 1.0            # Foot position tracking
    W_foot_vel: 1.0        # Foot velocity tracking
    W_upright: 5.0         # Torso upright orientation
    w_balance: 5.0         # Balance cost weight
  
  # Norm types for robust cost functions (DeepMind MJPC exact values)
  # Available: quadratic (0), l22 (1), l2 (2), cosh (3), smooth_abs_2_loss (7), rectify (8)
  norm_types:
    balance:               # L22 norm (smooth near zero, quadratic growth)
      type: 1
      p: 0.02
      q: 4.0
    upright:               # L2 norm (linear growth for outliers)
      type: 2
      p: 0.01
    com_velocity:          # SmoothAbs2Loss (smooth quadratic-to-linear transition)
      type: 7
      p: 0.2
      q: 4.0
    com_position:          # SmoothAbs2Loss
      type: 7
      p: 0.1
      q: 4.0
    ee_position_foot_left:  # Rectify (one-sided penalty)
      type: 8
      p: 0.05
    ee_velocity_foot_left:  # SmoothAbs2Loss
      type: 7
      p: 0.2
      q: 4.0
  
  # iLQR solver settings
  ilqr_settings:
    max_iterations: 10                # Maximum iLQR iterations
    tolerance: 1.0e-4                 # Convergence tolerance
    initial_regularization: 1.0e-3    # Starting lambda value
    reg_min: 1.0e-6                   # Min regularization
    reg_max: 1.0e6                    # Max regularization
    reg_increase_factor: 10.0         # Lambda *= factor (poor improvement)
    reg_decrease_factor: 10.0         # Lambda /= factor (good improvement)
    trust_region_good: 0.5            # Improvement ratio threshold for "good"
    trust_region_poor: 0.25           # Improvement ratio threshold for "poor"
    
    # Log-scale line search (DeepMind MJPC style)
    num_line_search_steps: 30         # Number of alpha candidates
    min_linesearch_step: 1.0e-3       # Minimum step size (log-scale from 1.0 to min)
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

The system is **model-agnostic**. To add a new robot:

1. **Add your robot files** to `robots/your_robot/`
2. **Update `config.yaml`**:
   ```yaml
   robot:
     name: your_robot
     model_path: "robots/your_robot/model.xml"
     urdf_path: "robots/your_robot/robot.urdf"
     ee_feet:
       left_feet_ee: "left_foot_body_name"   # Body name in your MJCF/URDF
       right_feet_ee: "right_foot_body_name"  # Body name in your MJCF/URDF
   ```
3. **Generate reference trajectories** (q_ref.csv, v_ref.csv, contact_schedule.csv)
4. **Adjust cost weights** for your robot's dimensions (optional)

**Supported Models**:
- ✅ Unitree H1 (bodies: `left_ankle_link`, `right_ankle_link`)
- ✅ DeepMind Humanoid (bodies: `foot_left`, `foot_right`)
- ✅ Any humanoid with MuJoCo MJCF/URDF format

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

This work was inspired by and references concepts from **Google DeepMind's [MuJoCo MPC (MJPC)](https://github.com/google-deepmind/mujoco_mpc)**, a general-purpose predictive control framework. While our implementation takes a different approach (specialized iLQR for humanoids with analytical derivatives), we acknowledge MJPC as a valuable reference for MPC design patterns and contact modeling strategies.

### Core Dependencies

- **[MuJoCo](https://mujoco.org/)** ([Todorov et al., 2012](https://ieeexplore.ieee.org/document/6386109)) - Advanced physics simulation engine
  ```
  @inproceedings{todorov2012mujoco,
    title={MuJoCo: A physics engine for model-based control},
    author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
    booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
    pages={5026--5033},
    year={2012},
    organization={IEEE}
  }
  ```

- **[Pinocchio](https://github.com/stack-of-tasks/pinocchio)** ([Carpentier et al., 2019](https://hal.science/hal-01866228)) - Rigid body dynamics library
  ```
  @inproceedings{carpentier2019pinocchio,
    title={The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives},
    author={Carpentier, Justin and Saurel, Guilhem and Buondonno, Gabriele and Mirabel, Joseph and Lamiraux, Florent and Stasse, Olivier and Mansard, Nicolas},
    booktitle={2019 IEEE/SICE International Symposium on System Integration (SII)},
    pages={614--619},
    year={2019},
    organization={IEEE}
  }
  ```

- **[CasADi](https://web.casadi.org/)** ([Andersson et al., 2019](https://link.springer.com/article/10.1007/s12532-018-0139-4)) - Symbolic framework for automatic differentiation
  ```
  @article{andersson2019casadi,
    title={CasADi: a software framework for nonlinear optimization and optimal control},
    author={Andersson, Joel AE and Gillis, Joris and Horn, Greg and Rawlings, James B and Diehl, Moritz},
    journal={Mathematical Programming Computation},
    volume={11},
    number={1},
    pages={1--36},
    year={2019},
    publisher={Springer}
  }
  ```

- **[Eigen](https://eigen.tuxfamily.org/)** - High-performance C++ linear algebra library

### Robot Models

- **[Unitree H1](https://www.unitree.com/)** - Humanoid robot platform
- **[DeepMind Humanoid](https://github.com/google-deepmind/dm_control)** - Simulated humanoid from DM Control Suite
  ```
  @article{tunyasuvunakool2020dm_control,
    title={dm\_control: Software and tasks for continuous control},
    author={Tunyasuvunakool, Saran and Muldal, Alistair and Doron, Yotam and Liu, Siqi and Bohez, Steven and Merel, Josh and Erez, Tom and Lillicrap, Timothy and Heess, Nicolas and Tassa, Yuval},
    journal={Software Impacts},
    volume={6},
    pages={100022},
    year={2020},
    publisher={Elsevier}
  }
  ```

### Algorithmic References

- **iLQR Algorithm**: [Li & Todorov, 2004](https://ieeexplore.ieee.org/document/1389084)
  ```
  @inproceedings{li2004iterative,
    title={Iterative linear quadratic regulator design for nonlinear biological movement systems},
    author={Li, Weiwei and Todorov, Emanuel},
    booktitle={First International Conference on Informatics in Control, Automation and Robotics},
    pages={222--229},
    year={2004}
  }
  ```

- **Differential Dynamic Programming (DDP)**: [Mayne, 1966](https://www.sciencedirect.com/science/article/pii/S1474667017699666); [Jacobson & Mayne, 1970](https://www.sciencedirect.com/science/article/pii/B9780123724403500038)

### Additional Tools

- **yaml-cpp** - YAML configuration parser
- **GLFW** - OpenGL windowing and input library

## 📧 Contact

For questions or issues, please open a [GitHub Issue](https://github.com/premsuggu/Mujoco-MPC/issues).

---