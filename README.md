# Autonomous Vehicle System

A complete autonomous vehicle simulation system that implements the full stack of perception, planning, and control functionalities.

## System Overview

This system is a modular autonomous vehicle simulation platform containing core modules such as environment perception, path planning, motion control, and visualization. The system is implemented in Python, supports dual-mode perception using deep learning (YOLOv8) and traditional computer vision, and provides a complete simulation environment with a real-time visualization interface.

## Project Structure

```
.
├── car_system.py              # System core module (all base classes)
├── main_demo.py              # Main demonstration program
├── test_modules.py           # Testing and evaluation module
├── visualization.py          # Real-time visualization module
├── yolov8_detector.py        # YOLOv8 deep learning perception module
├── unified_system.py         # Unified system (integrated version)
├── run_visualization.py      # Visualization launcher
├── test_import.py           # Environment check script
├── requirements.txt          # Dependency package list
└── README.md                # This document
```

## Core Functions

### 1. Environment Perception Module
- **Traditional Method**: Edge detection + contour analysis based on OpenCV
- **Deep Learning Method**: Object detection based on YOLOv8 (pedestrians, vehicles, traffic signs, etc.)
- **Dual-mode Switching**: Automatically selects the optimal detector based on runtime environment
- **Distance Estimation**: Estimates obstacle distance based on object size and camera model

### 2. Path Planning Module
- **Improved A* Algorithm**: 8-direction search, heuristic weighting optimization
- **Path Smoothing**: Straight-line feasibility check, redundant point removal
- **Obstacle Handling**: Obstacle grid map, safe distance penalty
- **Real-time Replanning**: Supports dynamic obstacle environments

### 3. Motion Control Module
- **Layered PID Control**: Independent steering and speed control
- **Feedforward Compensation**: Steering pre-compensation based on path curvature
- **Adaptive Adjustment**: Dynamic look-ahead distance and speed control
- **Steering Limits**: Steering rate limiting ensures smoothness

### 4. Simulation & Visualization
- **Vehicle Dynamics Simulation**: Bicycle model, physical constraints
- **Real-time Visualization**: Multi-view display of system status
- **Performance Monitoring**: Records processing time, tracking error, and other metrics
- **Result Saving**: Automatically saves test results and charts

## Quick Start

### Environment Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python test_import.py
```

### Running Demonstrations

#### Method 1: Run Complete Demo (Recommended)

```bash
python main_demo.py
```

The system will automatically detect YOLOv8 availability and run the corresponding demonstration mode.

#### Method 2: Run Simple Demo

```bash
python unified_system.py
```

This is a simplified version of the system, suitable for quickly understanding basic functionality.

#### Method 3: Run Real-time Visualization Demo

```bash
python run_visualization.py
```

Launches an interactive visualization interface to observe the system's real-time operation status.

### Running Tests

#### Module Unit Tests

```bash
# Test perception module
python -c "from test_modules import test_perception; test_perception()"

# Test planning module
python -c "from test_modules import test_planning; test_planning()"

# Test control module
python -c "from test_modules import test_control; test_control()"
```

#### Comprehensive Performance Test

```bash
python -c "from test_modules import run_comprehensive_test; run_comprehensive_test()"
```

This test will evaluate the performance of all modules and generate a detailed test report.

## Detailed Usage Instructions

### 1. System Configuration

All system parameters are centrally managed through the `SystemConfig` class. Main configuration items include:

- **Perception Parameters**: Confidence threshold, YOLO model path
- **Control Parameters**: PID gains, look-ahead distance, steering limits
- **Planning Parameters**: Grid size, A* algorithm parameters, obstacle radius
- **Path Parameters**: Maximum path points, path smoothing parameters

### 2. Core Module Description

#### 2.1 Perception Module

The system supports two perception modes:
- **Traditional Mode**: Uses OpenCV for edge detection and contour analysis
- **YOLOv8 Mode**: Uses deep learning for object detection (requires installing the ultralytics package)

The system automatically detects YOLOv8 availability and selects the corresponding perception mode.

#### 2.2 Path Planning Module

Based on an improved A* algorithm with the following features:
- 8-direction movement search
- Heuristic function weighting optimization
- Early exit mechanism (returns early when close to the goal)
- Path smoothing

The planning module converts detected obstacles into a 2D grid map, then searches for the optimal path from the start point to the goal point.

#### 2.3 Motion Control Module

Uses a layered control architecture:
- **High-level**: Path tracking, calculates desired steering angle
- **Low-level**: PID control, implements steering and speed control
- **Adaptive**: Dynamically adjusts parameters based on path curvature and tracking error

#### 2.4 Simulation Module

Based on the bicycle model (Ackermann steering geometry):
- Considers vehicle kinematic constraints
- Simulates steering response delay
- Considers physical limits (maximum acceleration, minimum turning radius)

### 3. Demonstration Modes

The system provides three main demonstration modes:

#### 3.1 Simple Demo
- From a fixed start point to a single goal point
- Demonstrates the basic perception-planning-control process
- Generates running trajectory and performance data

#### 3.2 Advanced Demo
- Multi-waypoint navigation
- Complex obstacle scenarios
- Demonstrates adaptive control strategies

#### 3.3 Obstacle Avoidance Demo
- Specifically tests obstacle avoidance capabilities
- Counts avoidance attempts and success rate
- Evaluates path planning quality

### 4. Visualization Features

The system provides rich visualization features:

#### 4.1 Real-time Visualization
- Main View: Displays car, obstacles, path, and goal points
- Perception View: Displays raw image and detection boxes
- Planning View: Displays grid map and planned path
- Control View: Displays control signal change curves
- Performance View: Displays system performance metrics

#### 4.2 Result Saving
- Automatically saves test images
- Generates performance reports (JSON format)
- Creates system performance overview charts

### 5. Performance Monitoring

Built-in performance monitoring module records:
- Perception processing time
- Path planning time
- Control tracking error
- Number of target detections

## Technical Features

### 1. Modular Design
Each module is independently implemented with clear interfaces, easy to extend and maintain.

### 2. Configurable Parameters
All key parameters are centrally managed through configuration files, facilitating debugging and optimization.

### 3. Dual-mode Perception
Supports both traditional CV and deep learning perception methods, adapting to different hardware conditions.

### 4. Complete Simulation Environment
Provides realistic vehicle dynamics simulation, supporting algorithm verification and performance evaluation.

### 5. Rich Visualization
Multi-view real-time display of system status, facilitating algorithm debugging and result analysis.

## File Description

### Core Files

1. **car_system.py**
   - Contains all base classes: SystemConfig, ObstacleDetector, PathPlanner, CarController, CarSimulator, SystemMonitor
   - Provides system component creation and verification functions
   - Includes a complete system testing framework

2. **main_demo.py**
   - Main demonstration program, provides multiple demonstration modes
   - Contains a complete perception-planning-control loop
   - Supports interactive operation and automated testing

3. **test_modules.py**
   - Module unit testing and performance evaluation
   - Generates detailed test reports and visualization charts
   - Includes comprehensive performance testing functionality

### Auxiliary Files

4. **visualization.py**
   - Real-time visualization class definitions
   - Multi-view display of system status
   - Supports interactive demonstrations

5. **yolov8_detector.py**
   - YOLOv8 deep learning detector implementation
   - Includes model loading, inference, and result processing
   - Provides model failure recovery mechanisms

6. **unified_system.py**
   - Simplified system version, suitable for quick demonstrations
   - Contains minimal implementation of all core functionalities

## Testing and Verification

### 1. Unit Testing
Each module has corresponding test functions to verify basic functionality:
- Perception module: Tests obstacle detection accuracy and distance estimation
- Planning module: Tests path search success rate and planning time
- Control module: Tests path tracking precision and stability

### 2. Integration Testing
Verifies the collaborative capability of each module:
- Complete workflow testing
- End-to-end performance evaluation
- System stability testing

### 3. Performance Testing
Evaluates system performance under different scenarios:
- Simple scenarios: Straight lines, gentle curves
- Complex scenarios: Multiple obstacles, narrow passages
- Extreme scenarios: Sharp curves, continuous steering

## Usage Examples

### Example 1: Creating System Components

```python
from car_system import create_system_components

# Create all system components
detector, planner, controller, simulator, monitor = create_system_components()

# Set goal position
goal_position = [10, 10]

# Run control loop
for step in range(100):
    # 1. Perception
    image = generate_test_image()
    detections = detector.detect(image)

    # 2. Planning
    planner.update_obstacle_grid(detections, simulator.position)
    path = planner.astar_search(simulator.position, goal_position)

    # 3. Control
    throttle, steering = controller.update_control(
        simulator.position, simulator.heading, path
    )

    # 4. Execution
    simulator.update(throttle, steering)
```

### Example 2: Running Performance Tests

```python
from test_modules import run_comprehensive_test

# Run comprehensive performance test
results = run_comprehensive_test()

# View test results
print(f"Perception module score: {results['scores']['perception']:.1f}/10")
print(f"Planning module score: {results['scores']['planning']:.1f}/10")
print(f"Control module score: {results['scores']['control']:.1f}/10")
print(f"Overall score: {results['scores']['overall']:.1f}/10")
```

## Troubleshooting

### Common Issue 1: YOLOv8 fails to load

**Symptom**: System prompts "YOLOv8 not available" or model loading fails

**Solution**:
```bash
# Reinstall ultralytics package
pip uninstall ultralytics -y
pip install ultralytics

# Or manually download the model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Common Issue 2: OpenCV import error

**Symptom**: ImportError: No module named 'cv2'

**Solution**:
```bash
# Reinstall OpenCV
pip uninstall opencv-python -y
pip install opencv-python
```

### Common Issue 3: PID controller error

**Symptom**: ImportError: No module named 'simple_pid'

**Solution**:
```bash
# Install simple-pid package
pip install simple-pid
```

### Common Issue 4: Insufficient memory

**Symptom**: System runs slowly or crashes

**Solution**:
- Reduce grid map size (modify SystemConfig.GRID_SIZE)
- Reduce maximum path points (modify SystemConfig.MAX_PATH_POINTS)
- Disable unnecessary visualization features

## Development Extensions

### 1. Adding New Perception Algorithms
Inherit from the ObstacleDetector base class and implement the detect method:
```python
class NewDetector(ObstacleDetector):
    def __init__(self, ...):
        # Initialize parameters

    def detect(self, image):
        # Implement detection logic
        # Return detections list
```

### 2. Modifying Control Strategy
Modify the update_control method of the CarController class to implement new control algorithms.

### 3. Adding New Test Scenarios
Add new test functions in test_modules.py to generate different path and obstacle configurations.

### 4. Extending Visualization Features
Inherit from the CarSystemVisualizer class to add new views or display content.

## Performance Metrics

Main system performance metrics include:

1. **Perception Performance**
   - Detection accuracy: >85%
   - Processing time: <50ms (traditional method <15ms)
   - Distance estimation error: <20%

2. **Planning Performance**
   - Planning success rate: >88% (simple scenes 100%)
   - Planning time: <40ms (simple scenes <10ms)
   - Path quality: efficiency >0.75

3. **Control Performance**
   - Tracking error: <0.7m (straight lines <0.25m)
   - Goal arrival rate: >95%
   - Control stability: No significant oscillation

## System Requirements

### Hardware Requirements
- CPU: Intel Core i5 or equivalent performance
- RAM: 8GB (16GB recommended)
- Storage: At least 2GB available space

### Software Requirements
- Operating System: Windows 10/11, Linux Ubuntu 18.04+, macOS 10.15+
- Python Version: Python 3.8 or higher
- Necessary Dependencies: See requirements.txt

### Recommended Configuration
- GPU: NVIDIA GPU (for accelerating YOLOv8 inference)
- RAM: 16GB
- Python Environment: Use conda or venv for isolated environment

## References

1. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement.
2. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths.
3. Thrun, S., Montemerlo, M., & Dahlkamp, H. (2006). Stanley: The robot that won the DARPA Grand Challenge.

## Changelog

### v1.0.0 (Initial Version)
- Implements basic perception-planning-control framework
- Supports dual-mode perception (traditional CV and YOLOv8)
- Provides complete simulation and visualization environment
- Includes performance testing and evaluation tools

---

*Note: This system is for academic research purposes only. Do not use for actual vehicle control.*