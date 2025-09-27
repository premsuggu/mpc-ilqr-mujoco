# Humanoid Model Predictive Control (MPC)

A cross-platform implementation of Model Predictive Control for humanoid robots using MuJoCo physics simulation and iterative Linear Quadratic Regulator (iLQR).

![MPC Demo](docs/mpc_demo.gif) <!-- Add a demo gif later -->

## 🚀 Features

- **Real-time MPC**: 50Hz control loop for humanoid balance and locomotion
- **iLQR Optimization**: Efficient iterative Linear Quadratic Regulator solver
- **MuJoCo Integration**: High-fidelity physics simulation using MuJoCo
- **Cross-platform**: Works on Windows, Linux, and macOS
- **H1 Humanoid Robot**: Pre-configured for Unitree H1 robot model
- **Visualization Tools**: Python scripts for trajectory analysis and 3D visualization

## 📋 Prerequisites

- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download/)
- [Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/community/) with C++ build tools **(Windows only)**

## 🛠️ Setup & Installation

### **Quick Start (Linux/macOS)**
```bash
# One-liner setup for Linux users
git clone https://github.com/premsuggu/mpc-ilqr-mujoco.git && cd mpc-ilqr-mujoco && conda env create -f environment.yml && conda activate humanoid-mpc && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/humanoid_mpc
```

### **Step-by-Step Setup (All Platforms)**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/premsuggu/mpc-ilqr-mujoco.git
   cd mpc-ilqr-mujoco
   ```

2. **Create conda environment** (installs all C++ and Python dependencies):
   ```bash
   conda env create -f environment.yml
   conda activate humanoid-mpc
   ```

3. **Build the C++ code**:
   ```bash
   # Make sure conda environment is activated
   conda activate humanoid-mpc
   
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```

## 🎮 Usage

**⚠️ Important: Always activate the conda environment first!**

```bash
# Standard conda activation (Linux/macOS/Windows)
conda activate humanoid-mpc

# Windows PowerShell alternative (if above doesn't work)
C:\Users\{username}\miniconda3\Scripts\activate.bat humanoid-mpc
```

### 1. Run MPC Simulation

```bash
# Windows
build\Release\humanoid_mpc.exe

# Linux/macOS
./build/humanoid_mpc
```

### 2. Visualize Results

```bash
# 3D MuJoCo visualization
python simulate.py

# Plot tracking performance
python plotter.py
```

### 3. Generated Files

The simulation creates:
- `results/q_optimal.csv`: Optimal state trajectory
- `results/u_optimal.csv`: Optimal control inputs
- `results/humanoid_tracking_comparison.png`: Performance plots
- `results/humanoid_tracking_errors.png`: Error analysis

## 📊 Project Structure

```
humanoid-mpc/
├── app/
│   └── humanoid_mpc.cpp          # Main application
├── include/
│   ├── robot_utils.hpp           # MuJoCo wrapper
│   ├── ilqr.hpp                  # iLQR solver
│   └── mpc.hpp                   # MPC orchestrator
├── src/
│   ├── robot_utils.cpp
│   ├── ilqr.cpp
│   └── mpc.cpp
├── robots/
│   └── h1_description/           # H1 robot URDF/MJCF files
├── data/
│   ├── q_standing.csv            # Standing reference trajectory
│   └── v_standing.csv            # Standing reference velocities
│   └── q_ref.csv                 # Walking reference trajectory
│   └── v_ref.csv                 # Walking reference velocities
├── results/                      # Generated trajectory files
├── simulate.py                   # 3D visualization script
├── plotter.py                    # Analysis and plotting script
├── environment.yml               # Conda environment with all dependencies
├── CMakeLists.txt                # Build configuration
└── README.md                     # This file
```

## 🧮 Algorithm Overview

### Dependencies (installed via conda)
**C++ Libraries:**
- MuJoCo 3.3+ (physics simulation)
- Eigen 3.4+ (linear algebra)  
- GLFW 3.4+ (OpenGL windowing)
- CMake 4.1+ (build system)

**Python Packages:**
- mujoco (Python bindings)
- numpy, pandas, scipy (scientific computing)
- matplotlib, seaborn, plotly (visualization)

### Model Predictive Control (MPC)
- **Horizon**: 25 steps (0.5 seconds at 50Hz)
- **Cost Function**: Quadratic penalties on state deviation and control effort
- **Constraints**: Joint limits and control bounds
- **Solver**: Iterative Linear Quadratic Regulator (iLQR)

### Cost Function Weights
```cpp
// Position tracking (critical for balance)
Q(0,0) = 100.0;   // X position
Q(1,1) = 100.0;   // Y position  
Q(2,2) = 2000.0;  // Z position (height)

// Orientation (preventing falls)
Q(3,3) = 500.0;   // Quaternion W
Q(4,4) = 500.0;   // Quaternion X (roll)
Q(5,5) = 500.0;   // Quaternion Y (pitch)
Q(6,6) = 300.0;   // Quaternion Z (yaw)

// Velocity damping (preventing oscillations)
Q_vel *= 150.0;   // Joint velocities
```

## 🔧 Configuration

### Simulation Parameters
Edit `app/humanoid_mpc.cpp`:
```cpp
const double dt = 0.02;          // MPC frequency (50Hz)
const int N = 25;                // Prediction horizon 
const int sim_steps = 20;        // Simulation length
const double physics_dt = 0.02;  // Physics timestep
```

### Cost Function Tuning
Adjust weights in `app/humanoid_mpc.cpp`:
- Increase position weights for tighter tracking
- Increase velocity weights to reduce oscillations
- Adjust control effort weights for smoother controls

### Reference Trajectories
- The csv files `data/q_standing.csv` and `data/v_standing.csv`are reference for a standing pose, All joints at 0° except base height (Z = 1.0432m).
- While the csv files `q_ref.csv` and `v_ref.csv` are for walking motion.

## 📈 Performance
Run the `simulate.py` and `plotter.py` files to see the results.

**Typical Results:**
- **Tracking Error**: <1cm RMS position error
- **Balance Time**: >20 seconds of stable standing
- **Computational Speed**: ~2.6ms per MPC step (real-time capable)
- **Convergence**: iLQR typically converges in 3-5 iterations

## 🐛 Troubleshooting

<details>
<summary>Common Issues and Solutions</summary>

## 🐛 Troubleshooting

<details>
<summary>Common Issues and Solutions</summary>

**Build Errors:**
```bash
# Clean and rebuild (Linux/macOS)
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Clean and rebuild (Windows)
rmdir /s build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Missing C++ Compiler:**
- **Windows**: Install Visual Studio 2022 Community (required)
- **Linux**: `sudo apt install build-essential` (Ubuntu/Debian)
- **macOS**: `xcode-select --install`

**Python Import Errors (ModuleNotFoundError: No module named 'mujoco'):**
```bash
# Make sure conda environment is activated first
conda activate humanoid-mpc

# Verify MuJoCo is installed
conda list mujoco

# If not installed, recreate environment
conda env remove -n humanoid-mpc
conda env create -f environment.yml
conda activate humanoid-mpc
```

**Conda Environment Issues:**
```bash
# Update conda and retry (all platforms)
conda update conda
conda env create -f environment.yml --force

# Alternative activation (Linux/macOS only)
source activate humanoid-mpc
```

**MuJoCo Visualization Issues:**
- **Linux**: Install X11 forwarding for SSH: `ssh -X username@hostname`
- **Windows**: Update graphics drivers through Device Manager
- **macOS**: Install XQuartz if needed: `brew install --cask xquartz`

**MuJoCo Model Loading Error:**
- Ensure you're running from the project root directory
- Check that `robots/h1_description/mjcf/scene.xml` exists

</details>

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MuJoCo](https://mujoco.org/) for the physics simulation
- [Eigen3](https://eigen.tuxfamily.org/) for linear algebra
- [Unitree H1](https://www.unitree.com/h1) for the robot model

---
