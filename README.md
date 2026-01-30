# 自动驾驶汽车系统

一个完整的自动驾驶汽车仿真系统，实现了感知、规划、控制全栈功能。

## 系统概述

本系统是一个模块化的自动驾驶汽车仿真平台，包含环境感知、路径规划、运动控制和可视化等核心模块。系统采用Python实现，支持深度学习（YOLOv8）与传统计算机视觉双模态感知，提供完整的仿真环境和实时可视化界面。

## 项目结构

```
.
├── car_system.py              # 系统核心模块（所有基础类）
├── main_demo.py              # 主演示程序
├── test_modules.py           # 测试与评估模块
├── visualization.py          # 实时可视化模块
├── yolov8_detector.py        # YOLOv8深度学习感知模块
├── unified_system.py         # 统一系统（集成版）
├── run_visualization.py      # 可视化启动器
├── test_import.py           # 环境检查脚本
├── requirements.txt          # 依赖包列表
└── README.md                # 本文档
```

## 核心功能

### 1. 环境感知模块
- **传统方法**：基于OpenCV的边缘检测+轮廓分析
- **深度学习方法**：基于YOLOv8的目标检测（行人、车辆、交通标志等）
- **双模态切换**：根据运行时环境自动选择最优检测器
- **距离估计**：基于物体尺寸和相机模型估算障碍物距离

### 2. 路径规划模块
- **改进A*算法**：8方向搜索，启发式加权优化
- **路径平滑**：直线可行性检查，移除冗余点
- **障碍物处理**：障碍物栅格地图，安全距离惩罚
- **实时重规划**：支持动态障碍物环境

### 3. 运动控制模块
- **分层PID控制**：转向和速度独立控制
- **前馈补偿**：基于路径曲率预补偿转向
- **自适应调节**：动态前瞻距离和速度控制
- **转向限制**：转向速率限制保证平顺性

### 4. 仿真与可视化
- **汽车动力学仿真**：自行车模型，物理约束
- **实时可视化**：多视图显示系统状态
- **性能监控**：处理时间、跟踪误差等指标记录
- **结果保存**：自动保存测试结果和图表

## 快速开始

### 环境安装

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安装依赖包
pip install -r requirements.txt

# 3. 验证安装
python test_import.py
```

### 运行演示

#### 方式一：运行完整演示（推荐）

```bash
python main_demo.py
```

系统将自动检测YOLOv8可用性并运行相应的演示模式。

#### 方式二：运行简单演示

```bash
python unified_system.py
```

这是一个精简版系统，适合快速了解基本功能。

#### 方式三：运行实时可视化演示

```bash
python run_visualization.py
```

启动交互式可视化界面，可观察系统实时运行状态。

### 运行测试

#### 模块单元测试

```bash
# 测试感知模块
python -c "from test_modules import test_perception; test_perception()"

# 测试规划模块  
python -c "from test_modules import test_planning; test_planning()"

# 测试控制模块
python -c "from test_modules import test_control; test_control()"
```

#### 综合性能测试

```bash
python -c "from test_modules import run_comprehensive_test; run_comprehensive_test()"
```

该测试将评估所有模块的性能并生成详细的测试报告。

## 详细使用说明

### 1. 系统配置

系统所有参数通过 `SystemConfig` 类集中管理，主要配置项包括：

- **感知参数**：置信度阈值、YOLO模型路径
- **控制参数**：PID增益、前瞻距离、转向限制
- **规划参数**：栅格大小、A*算法参数、障碍物半径
- **路径参数**：最大路径点数、路径平滑参数

### 2. 核心模块说明

#### 2.1 感知模块

系统支持两种感知模式：
- **传统模式**：使用OpenCV进行边缘检测和轮廓分析
- **YOLOv8模式**：使用深度学习进行目标检测（需要安装ultralytics包）

系统会自动检测YOLOv8的可用性并选择相应的感知模式。

#### 2.2 路径规划模块

基于改进的A*算法，具有以下特性：
- 8方向移动搜索
- 启发式函数加权优化
- 提前退出机制（接近目标时提前返回）
- 路径平滑处理

规划模块将检测到的障碍物转换为2D栅格地图，然后搜索从起点到目标点的最优路径。

#### 2.3 运动控制模块

采用分层控制架构：
- **上层**：路径跟踪，计算期望的转向角
- **下层**：PID控制，实现转向和速度控制
- **自适应**：根据路径曲率和跟踪误差动态调整参数

#### 2.4 仿真模块

基于自行车模型（Ackermann转向几何）：
- 考虑车辆运动学约束
- 模拟转向响应延迟
- 考虑物理限制（最大加速度、最小转弯半径）

### 3. 演示模式

系统提供三种主要演示模式：

#### 3.1 简单演示
- 从固定起点到单一目标点
- 展示基本的感知-规划-控制流程
- 生成运行轨迹和性能数据

#### 3.2 高级演示  
- 多航点导航
- 复杂障碍物场景
- 自适应控制策略展示

#### 3.3 避障演示
- 专门测试避障能力
- 统计避障次数和成功率
- 评估路径规划的质量

### 4. 可视化功能

系统提供丰富的可视化功能：

#### 4.1 实时可视化
- 主视图：显示汽车、障碍物、路径和目标点
- 感知视图：显示原始图像和检测框
- 规划视图：显示栅格地图和规划路径
- 控制视图：显示控制信号变化曲线
- 性能视图：显示系统性能指标

#### 4.2 结果保存
- 自动保存测试图像
- 生成性能报告（JSON格式）
- 创建系统性能概览图

### 5. 性能监控

系统内置性能监控模块，记录：
- 感知处理时间
- 路径规划时间
- 控制跟踪误差
- 目标检测数量

## 技术特点

### 1. 模块化设计
各模块独立实现，接口清晰，易于扩展和维护。

### 2. 参数可配置
所有关键参数通过配置文件集中管理，便于调试和优化。

### 3. 双模态感知
支持传统CV和深度学习两种感知方式，适应不同硬件条件。

### 4. 完整的仿真环境
提供真实的汽车动力学仿真，支持算法验证和性能评估。

### 5. 丰富的可视化
多视图实时显示系统状态，便于算法调试和结果分析。

## 文件说明

### 核心文件

1. **car_system.py**
   - 包含所有基础类：SystemConfig、ObstacleDetector、PathPlanner、CarController、CarSimulator、SystemMonitor
   - 提供系统组件创建和验证函数
   - 包含完整的系统测试框架

2. **main_demo.py**
   - 主演示程序，提供多种演示模式
   - 包含完整的感知-规划-控制循环
   - 支持交互式操作和自动测试

3. **test_modules.py**
   - 模块单元测试和性能评估
   - 生成详细的测试报告和可视化图表
   - 包含综合性能测试功能

### 辅助文件

4. **visualization.py**
   - 实时可视化类定义
   - 多视图显示系统状态
   - 支持交互式演示

5. **yolov8_detector.py**
   - YOLOv8深度学习检测器实现
   - 包含模型加载、推理和结果处理
   - 提供模型故障恢复机制

6. **unified_system.py**
   - 精简版系统，适合快速演示
   - 包含所有核心功能的最小实现

## 测试与验证

### 1. 单元测试
每个模块都有对应的测试函数，验证基本功能：
- 感知模块：测试障碍物检测准确率和距离估计
- 规划模块：测试路径搜索成功率和规划时间
- 控制模块：测试路径跟踪精度和稳定性

### 2. 集成测试
验证各模块协同工作能力：
- 完整的工作流程测试
- 端到端性能评估
- 系统稳定性测试

### 3. 性能测试
评估系统在不同场景下的性能：
- 简单场景：直线、缓弯
- 复杂场景：多障碍物、狭窄通道
- 极限场景：急弯、连续转向

## 使用示例

### 示例1：创建系统组件

```python
from car_system import create_system_components

# 创建所有系统组件
detector, planner, controller, simulator, monitor = create_system_components()

# 设置目标位置
goal_position = [10, 10]

# 运行控制循环
for step in range(100):
    # 1. 感知
    image = generate_test_image()
    detections = detector.detect(image)
    
    # 2. 规划
    planner.update_obstacle_grid(detections, simulator.position)
    path = planner.astar_search(simulator.position, goal_position)
    
    # 3. 控制
    throttle, steering = controller.update_control(
        simulator.position, simulator.heading, path
    )
    
    # 4. 执行
    simulator.update(throttle, steering)
```

### 示例2：运行性能测试

```python
from test_modules import run_comprehensive_test

# 运行综合性能测试
results = run_comprehensive_test()

# 查看测试结果
print(f"感知模块得分: {results['scores']['perception']:.1f}/10")
print(f"规划模块得分: {results['scores']['planning']:.1f}/10")  
print(f"控制模块得分: {results['scores']['control']:.1f}/10")
print(f"总体得分: {results['scores']['overall']:.1f}/10")
```

## 故障排除

### 常见问题1：YOLOv8无法加载

**现象**：系统提示"YOLOv8 not available"或模型加载失败

**解决方法**：
```bash
# 重新安装ultralytics包
pip uninstall ultralytics -y
pip install ultralytics

# 或者手动下载模型
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 常见问题2：OpenCV导入错误

**现象**：ImportError: No module named 'cv2'

**解决方法**：
```bash
# 重新安装OpenCV
pip uninstall opencv-python -y
pip install opencv-python
```

### 常见问题3：PID控制器错误

**现象**：ImportError: No module named 'simple_pid'

**解决方法**：
```bash
# 安装simple-pid包
pip install simple-pid
```

### 常见问题4：内存不足

**现象**：系统运行缓慢或崩溃

**解决方法**：
- 减小栅格地图大小（修改SystemConfig.GRID_SIZE）
- 减少最大路径点数（修改SystemConfig.MAX_PATH_POINTS）
- 关闭不必要的可视化功能

## 扩展开发

### 1. 添加新的感知算法
继承ObstacleDetector基类，实现detect方法：
```python
class NewDetector(ObstacleDetector):
    def __init__(self, ...):
        # 初始化参数
        
    def detect(self, image):
        # 实现检测逻辑
        # 返回detections列表
```

### 2. 修改控制策略
修改CarController类的update_control方法，实现新的控制算法。

### 3. 添加新的测试场景
在test_modules.py中添加新的测试函数，生成不同的路径和障碍物配置。

### 4. 扩展可视化功能
继承CarSystemVisualizer类，添加新的视图或显示内容。

## 性能指标

系统主要性能指标包括：

1. **感知性能**
   - 检测准确率：>85%
   - 处理时间：<50ms（传统方法<15ms）
   - 距离估计误差：<20%

2. **规划性能**
   - 规划成功率：>88%（简单场景100%）
   - 规划时间：<40ms（简单场景<10ms）
   - 路径质量：效率>0.75

3. **控制性能**
   - 跟踪误差：<0.7m（直线<0.25m）
   - 目标到达率：>95%
   - 控制稳定性：无明显震荡

## 系统要求

### 硬件要求
- CPU：Intel Core i5或同等性能
- 内存：8GB RAM（推荐16GB）
- 存储：至少2GB可用空间

### 软件要求
- 操作系统：Windows 10/11, Linux Ubuntu 18.04+, macOS 10.15+
- Python版本：Python 3.8或更高版本
- 必要依赖：见requirements.txt

### 推荐配置
- GPU：NVIDIA GPU（用于加速YOLOv8推理）
- 内存：16GB RAM
- Python环境：使用conda或venv隔离环境

## 参考文献

1. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement.
2. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths.
3. Thrun, S., Montemerlo, M., & Dahlkamp, H. (2006). Stanley: The robot that won the DARPA Grand Challenge.

## 更新日志

### v1.0.0 (初始版本)
- 实现基础感知-规划-控制框架
- 支持传统CV和YOLOv8双模态感知
- 提供完整的仿真和可视化环境
- 包含性能测试和评估工具

---

*注意：本系统为学术研究用途，请勿用于实际车辆控制。*