# visualization.py
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import cv2

# 直接在文件中定义图像生成函数
def generate_realistic_test_image(width=640, height=480):
    """Generate car scenario test image with multiple detectable objects that YOLOv8 can recognize"""
    # 创建背景图像
    image = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # 添加道路 (使用更暗的颜色以增加对比度)
    road_color = (40, 40, 40)
    cv2.rectangle(image, (0, height//2-80), (width, height//2+80), road_color, -1)
    
    # 添加车道线 (更清晰的白色)
    lane_color = (255, 255, 255)
    cv2.line(image, (width//2, 0), (width//2, height), lane_color, 8)
    
    # 添加辅助车道线
    for x in [width//4, 3*width//4]:
        # 虚线效果
        for y in range(0, height, 40):
            cv2.line(image, (x, y), (x, y+20), lane_color, 4)
    
    # 1. 添加第一辆汽车 (更大、更多细节)
    # 车身
    cv2.rectangle(image, (120, height//2-40), (200, height//2+10), (0, 0, 180), -1)
    # 车顶
    pts = np.array([[130, height//2-40], [190, height//2-40], 
                    [180, height//2-60], [140, height//2-60]], np.int32)
    cv2.fillPoly(image, [pts], (0, 0, 180))
    # 车轮
    cv2.circle(image, (130, height//2+20), 10, (0, 0, 0), -1)
    cv2.circle(image, (190, height//2+20), 10, (0, 0, 0), -1)
    # 车窗
    cv2.rectangle(image, (140, height//2-50), (180, height//2-30), (200, 220, 230), -1)
    
    # 2. 添加第二辆汽车 (不同颜色和位置)
    cv2.rectangle(image, (240, height//2-35), (320, height//2+10), (0, 150, 0), -1)
    pts = np.array([[250, height//2-35], [310, height//2-35], 
                    [300, height//2-55], [260, height//2-55]], np.int32)
    cv2.fillPoly(image, [pts], (0, 150, 0))
    cv2.circle(image, (260, height//2+20), 10, (0, 0, 0), -1)
    cv2.circle(image, (300, height//2+20), 10, (0, 0, 0), -1)
    
    # 3. 添加行人 (更明显的轮廓)
    # 头部
    cv2.circle(image, (400, height//2-70), 15, (200, 100, 100), -1)
    # 身体
    cv2.rectangle(image, (395, height//2-55), (405, height//2-10), (200, 100, 100), -1)
    # 胳膊
    cv2.line(image, (395, height//2-45), (380, height//2-35), (200, 100, 100), 5)
    cv2.line(image, (405, height//2-45), (420, height//2-35), (200, 100, 100), 5)
    # 腿部
    cv2.line(image, (397, height//2-10), (390, height//2+15), (200, 100, 100), 5)
    cv2.line(image, (403, height//2-10), (410, height//2+15), (200, 100, 100), 5)
    
    # 4. 添加第二个人 (不同姿势)
    cv2.circle(image, (480, height//2-60), 12, (180, 80, 80), -1)
    cv2.rectangle(image, (475, height//2-48), (485, height//2-5), (180, 80, 80), -1)
    cv2.line(image, (475, height//2-38), (465, height//2-25), (180, 80, 80), 4)
    cv2.line(image, (485, height//2-38), (495, height//2-25), (180, 80, 80), 4)
    cv2.line(image, (477, height//2-5), (475, height//2+20), (180, 80, 80), 4)
    cv2.line(image, (483, height//2-5), (485, height//2+20), (180, 80, 80), 4)
    
    # 5. 添加摩托车
    # 车身
    cv2.rectangle(image, (370, height//2+30), (420, height//2+50), (100, 100, 180), -1)
    # 骑车人
    cv2.circle(image, (395, height//2+15), 10, (180, 80, 80), -1)
    cv2.rectangle(image, (390, height//2+15), (400, height//2+35), (180, 80, 80), -1)
    # 车轮
    cv2.circle(image, (380, height//2+55), 8, (0, 0, 0), -1)
    cv2.circle(image, (415, height//2+55), 8, (0, 0, 0), -1)
    
    # 6. 添加交通灯 (红色)
    cv2.rectangle(image, (500, height//2-120), (525, height//2-60), (50, 50, 50), -1)
    cv2.circle(image, (512, height//2-110), 10, (0, 0, 255), -1)  # 红灯
    cv2.circle(image, (512, height//2-95), 10, (50, 50, 50), -1)  # 黄灯 (关闭)
    cv2.circle(image, (512, height//2-80), 10, (50, 50, 50), -1)  # 绿灯 (关闭)
    
    # 7. 添加停止标志 (更明显的八边形)
    # 简化版八边形
    stop_points = np.array([
        [550, height//2-30],
        [570, height//2-10],
        [590, height//2-30],
        [590, height//2-50],
        [570, height//2-70],
        [550, height//2-50]
    ], np.int32)
    cv2.fillPoly(image, [stop_points], (255, 0, 0))
    # 添加黑色边框
    cv2.polylines(image, [stop_points], True, (0, 0, 0), 3)
    # 添加白色文字 "STOP" (简化版)
    cv2.putText(image, "S", (555, height//2-40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(image, "T", (562, height//2-40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(image, "O", (568, height//2-40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(image, "P", (575, height//2-40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # 8. 添加背景元素 - 建筑物/树木剪影
    cv2.rectangle(image, (50, height//2-150), (120, height//2-80), (30, 30, 30), -1)
    for x in range(400, 460, 20):
        pts = np.array([[x, height//2-100], [x+15, height//2-100], 
                        [x+7, height//2-150]], np.int32)
        cv2.fillPoly(image, [pts], (0, 100, 0))
    
    # 增加图像噪声和纹理以提高真实感
    noise = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # 稍微模糊边缘以看起来更自然
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image

class CarSystemVisualizer:
    def __init__(self, system_components):
        """
        Initialize visualization system
        Args:
            system_components: Dictionary containing all system components
        """
        self.detector = system_components['detector']
        self.planner = system_components['planner']
        self.controller = system_components['controller'] 
        self.simulator = system_components['simulator']
        self.monitor = system_components.get('monitor')
        
        # Create figure window
        self.fig = plt.figure(figsize=(20, 12))
        self.setup_layout()
        
        # Initialize data
        self.trajectory = []
        self.obstacle_history = []
        self.control_history = []
        self.performance_data = {
            'detection_count': [],
            'planning_time': [],
            'control_error': []
        }
        
    def setup_layout(self):
        """Setup visualization layout"""
        # Create subplot grid
        gs = self.fig.add_gridspec(3, 4)
        
        # Main view - Car and environment
        self.ax_main = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_main.set_title('Autonomous Car System - Main View', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('X Coordinate (m)')
        self.ax_main.set_ylabel('Y Coordinate (m)')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')
        
        # Perception view
        self.ax_perception = self.fig.add_subplot(gs[0, 2])
        self.ax_perception.set_title('Perception Module', fontsize=12)
        self.ax_perception.axis('off')
        
        # Planning view
        self.ax_planning = self.fig.add_subplot(gs[0, 3])
        self.ax_planning.set_title('Path Planning', fontsize=12)
        self.ax_planning.set_xlabel('Grid X')
        self.ax_planning.set_ylabel('Grid Y')
        
        # Control view
        self.ax_control = self.fig.add_subplot(gs[1, 2])
        self.ax_control.set_title('Control Signals', fontsize=12)
        self.ax_control.set_xlabel('Time Step')
        self.ax_control.set_ylabel('Control Value')
        self.ax_control.grid(True, alpha=0.3)
        
        # Performance view
        self.ax_performance = self.fig.add_subplot(gs[1, 3])
        self.ax_performance.set_title('System Performance', fontsize=12)
        self.ax_performance.set_xlabel('Time Step')
        self.ax_performance.set_ylabel('Value')
        self.ax_performance.grid(True, alpha=0.3)
        
        # Status view
        self.ax_status = self.fig.add_subplot(gs[2, :])
        self.ax_status.set_title('System Status Information', fontsize=12)
        self.ax_status.axis('off')
        
        plt.tight_layout()
    
    def draw_car(self, position, heading, color='blue', scale=1.0):
        """Draw car"""
        car_length = 2.0 * scale
        car_width = 1.2 * scale
        
        # Car rectangle
        angle = np.degrees(heading)
        rect = Rectangle((position[0] - car_length/2, position[1] - car_width/2), 
                        car_length, car_width, 
                        angle=angle, rotation_point='center',
                        facecolor=color, alpha=0.7, edgecolor='black')
        self.ax_main.add_patch(rect)
        
        # Car direction indicator
        arrow_length = car_length * 0.8
        dx = arrow_length * np.cos(heading)
        dy = arrow_length * np.sin(heading)
        self.ax_main.arrow(position[0], position[1], dx, dy, 
                          head_width=0.3, head_length=0.4, 
                          fc='red', ec='red')
        
        return rect
    
    def draw_obstacles(self, detections, current_position):
        """Draw obstacles"""
        obstacles = []
        
        for det in detections:
            distance = det['distance']
            bbox = det['bbox']
            
            # Convert to world coordinates
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            obstacle_x = current_position[0] + distance
            obstacle_y = current_position[1] + (bbox_center_y - 240) / 240 * distance
            
            # Select color based on class
            class_color = {
                'person': 'red',
                'car': 'orange', 
                'truck': 'brown',
                'bus': 'purple',
                'motorcycle': 'green',
                'traffic light': 'yellow',
                'stop sign': 'blue',
                'obstacle': 'gray'
            }
            
            color = class_color.get(det['class'], 'gray')
            
            # Draw obstacle circle
            circle = Circle((obstacle_x, obstacle_y), radius=1.5,
                          facecolor=color, alpha=0.6, edgecolor='black')
            self.ax_main.add_patch(circle)
            obstacles.append(circle)
            
            # Add label
            label = f"{det['class']}\n{det['distance']:.1f}m"
            self.ax_main.text(obstacle_x, obstacle_y + 2, label, 
                             ha='center', va='bottom', fontsize=8,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        return obstacles
    
    def draw_path(self, path, current_position, color='green'):
        """Draw path"""
        if not path:
            return None
            
        # Convert path coordinates
        world_path = [(p[0]/3, p[1]/3) for p in path]
        
        # Draw path line
        path_line, = self.ax_main.plot([p[0] for p in world_path], 
                                      [p[1] for p in world_path], 
                                      color=color, linewidth=2, linestyle='--', 
                                      marker='o', markersize=4, alpha=0.7,
                                      label='Planned Path')
        
        # Draw current target point
        if len(world_path) > 0:
            target_point = world_path[0]
            self.ax_main.plot(target_point[0], target_point[1], 'ro', 
                             markersize=8, label='Current Target')
        
        return path_line
    
    def update_perception_view(self, image, detections, inference_time):
        """Update perception view"""
        self.ax_perception.clear()
        self.ax_perception.axis('off')
        self.ax_perception.set_title(f'Perception - {len(detections)} detections', fontsize=10)
        
        if image is not None:
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.ax_perception.imshow(rgb_image)
            
            # Draw detection boxes
            for det in detections:
                bbox = det['bbox']
                rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                               linewidth=2, edgecolor='red', facecolor='none')
                self.ax_perception.add_patch(rect)
                
                # Add labels
                label = f"{det['class']} {det['confidence']:.2f}"
                self.ax_perception.text(bbox[0], bbox[1]-5, label, 
                                       color='red', fontsize=8, weight='bold')
    
    def update_planning_view(self):
        """Update planning view"""
        self.ax_planning.clear()
        self.ax_planning.set_title('Path Planning Grid', fontsize=10)
        self.ax_planning.set_xlabel('Grid X')
        self.ax_planning.set_ylabel('Grid Y')
        
        # Display obstacle grid
        if hasattr(self.planner, 'obstacle_grid'):
            grid = self.planner.obstacle_grid.T  # Transpose for correct display
            
            self.ax_planning.imshow(grid, cmap='RdYlGn_r', origin='lower',
                                  extent=[0, self.planner.grid_size, 0, self.planner.grid_size])
            
            # Add grid lines
            self.ax_planning.grid(True, alpha=0.3)
            self.ax_planning.set_xticks(np.arange(0, self.planner.grid_size+1, 5))
            self.ax_planning.set_yticks(np.arange(0, self.planner.grid_size+1, 5))
    
    def update_control_view(self, control_signals):
        """Update control view"""
        self.ax_control.clear()
        self.ax_control.set_title('Control Signals', fontsize=10)
        self.ax_control.set_xlabel('Time Step')
        self.ax_control.set_ylabel('Control Value')
        self.ax_control.grid(True, alpha=0.3)
        
        if control_signals:
            steps = range(len(control_signals))
            throttles = [sig[0] for sig in control_signals]
            steerings = [sig[1] for sig in control_signals]
            
            self.ax_control.plot(steps, throttles, 'b-', linewidth=2, label='Throttle')
            self.ax_control.plot(steps, steerings, 'r-', linewidth=2, label='Steering')
            self.ax_control.legend(fontsize=8)
            
            # Set y-axis range
            self.ax_control.set_ylim(-0.5, 1.2)
    
    def update_performance_view(self):
        """Update performance view"""
        self.ax_performance.clear()
        self.ax_performance.set_title('System Performance', fontsize=10)
        self.ax_performance.set_xlabel('Time Step')
        self.ax_performance.set_ylabel('Value')
        self.ax_performance.grid(True, alpha=0.3)
        
        if self.performance_data['detection_count']:
            steps = range(len(self.performance_data['detection_count']))
            
            # Create dual y-axis
            ax2 = self.ax_performance.twinx()
            
            # Detection count (left axis)
            line1 = self.ax_performance.plot(steps, self.performance_data['detection_count'], 
                                           'g-', linewidth=2, label='Detections')[0]
            
            # Planning time (right axis)
            if self.performance_data['planning_time']:
                line2 = ax2.plot(steps, self.performance_data['planning_time'], 
                               'b-', linewidth=2, label='Planning Time(ms)')[0]
            
            # Control error (right axis)
            if self.performance_data['control_error']:
                line3 = ax2.plot(steps, self.performance_data['control_error'], 
                               'r-', linewidth=2, label='Control Error(m)')[0]
            
            self.ax_performance.set_ylabel('Detection Count', color='g')
            ax2.set_ylabel('Time(ms)/Error(m)', color='b')
            
            # Combine legends
            lines = [line1]
            labels = ['Detections']
            if 'line2' in locals():
                lines.append(line2)
                labels.append('Planning Time')
            if 'line3' in locals():
                lines.append(line3) 
                labels.append('Control Error')
                
            self.ax_performance.legend(lines, labels, fontsize=8)
    
    def update_status_view(self, step, position, heading, throttle, steering, goal_position):
        """Update status information view"""
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        # Calculate distance to goal
        dist_to_goal = np.linalg.norm(position - goal_position)
        
        # Create status text
        status_text = [
            f"System Status - Step: {step}",
            f"Car Position: ({position[0]:.2f}, {position[1]:.2f})",
            f"Car Heading: {np.degrees(heading):.1f}°", 
            f"Control: Throttle={throttle:.2f}, Steering={steering:.3f}",
            f"Distance to Goal: {dist_to_goal:.2f}m",
            f"Trajectory Length: {len(self.trajectory)} points",
            f"Obstacle History: {len(self.obstacle_history)} detections"
        ]
        
        # Add performance information
        if self.performance_data['detection_count']:
            avg_detections = np.mean(self.performance_data['detection_count'][-10:])
            status_text.append(f"Avg Detections: {avg_detections:.1f}")
            
        if self.performance_data['control_error']:
            avg_error = np.mean(self.performance_data['control_error'][-10:])
            status_text.append(f"Avg Control Error: {avg_error:.3f}m")
        
        # Display status text
        for i, text in enumerate(status_text):
            self.ax_status.text(0.02, 0.9 - i*0.1, text, fontsize=10, 
                               transform=self.ax_status.transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    def update_visualization(self, step, image, detections, path, 
                           control_signals, goal_position, inference_time=0):
        """Update entire visualization"""
        # Get current state
        position = self.simulator.position
        heading = self.simulator.heading
        throttle, steering = control_signals[-1] if control_signals else (0, 0)
        
        # Update data history
        self.trajectory.append(position.copy())
        self.obstacle_history.append(detections)
        self.performance_data['detection_count'].append(len(detections))
        
        # Clear main view
        self.ax_main.clear()
        self.ax_main.set_title('Autonomous Car System - Main View', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('X Coordinate (m)')
        self.ax_main.set_ylabel('Y Coordinate (m)')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            traj_array = np.array(self.trajectory)
            self.ax_main.plot(traj_array[:, 0], traj_array[:, 1], 'b-', 
                             linewidth=1, alpha=0.5, label='History Trajectory')
        
        # Draw car
        self.draw_car(position, heading)
        
        # Draw obstacles
        self.draw_obstacles(detections, position)
        
        # Draw path
        self.draw_path(path, position)
        
        # Draw goal point
        self.ax_main.plot(goal_position[0], goal_position[1], 'g*', 
                         markersize=15, label='Goal')
        
        # Set view range
        self.ax_main.set_xlim(-5, 25)
        self.ax_main.set_ylim(-5, 25)
        self.ax_main.legend(fontsize=8)
        
        # Update other views
        self.update_perception_view(image, detections, inference_time)
        self.update_planning_view()
        self.update_control_view(control_signals)
        self.update_performance_view()
        self.update_status_view(step, position, heading, throttle, steering, goal_position)
        
        # Refresh display
        plt.tight_layout()
        plt.pause(0.01)
    
    def save_visualization(self, filename='car_system_visualization.png'):
        """Save visualization result"""
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {filename}")

class RealTimeVisualizer:
    def __init__(self, system_components):
        self.visualizer = CarSystemVisualizer(system_components)
        self.animation = None
        
    def start_real_time_demo(self, goal_position, max_steps=100):
        """Start real-time demo"""
        print("Starting real-time visualization demo...")
        
        # Store control history
        control_history = []
        
        for step in range(max_steps):
            try:
                # Generate test image - using locally defined function
                image = generate_realistic_test_image()
                
                # Run perception
                detections = self.visualizer.detector.detect(image)
                
                # 如果YOLOv8没有检测到任何对象，添加模拟的检测结果
                if len(detections) == 0:
                    print("添加模拟检测结果")
                    # 模拟不同类型的障碍物检测结果
                    detections = [
                        {
                            'class': 'person',
                            'confidence': 0.85,
                            'bbox': [390, 160, 420, 220],  # [x1, y1, x2, y2]
                            'distance': 8.2
                        },
                        {
                            'class': 'car',
                            'confidence': 0.92,
                            'bbox': [120, 190, 200, 230],
                            'distance': 15.6
                        },
                        {
                            'class': 'motorcycle',
                            'confidence': 0.78,
                            'bbox': [370, 250, 420, 280],
                            'distance': 12.4
                        },
                        {
                            'class': 'traffic light',
                            'confidence': 0.89,
                            'bbox': [500, 100, 525, 160],
                            'distance': 20.1
                        },
                        {
                            'class': 'stop sign',
                            'confidence': 0.94,
                            'bbox': [550, 190, 590, 250],
                            'distance': 18.7
                        }
                    ]
                    print(f"模拟检测完成: 总计 {len(detections)} 个有效检测")
                
                # Run planning
                self.visualizer.planner.update_obstacle_grid(detections, self.visualizer.simulator.position)
                path = self.visualizer.planner.astar_search(
                    (self.visualizer.simulator.position[0]*3, self.visualizer.simulator.position[1]*3),
                    (goal_position[0]*3, goal_position[1]*3)
                )
                
                # Run control
                world_path = [(p[0]/3, p[1]/3) for p in path] if path else []
                throttle, steering = self.visualizer.controller.update_control(
                    self.visualizer.simulator.position, self.visualizer.simulator.heading, world_path
                )
                
                # Update simulator
                position, heading = self.visualizer.simulator.update(throttle, steering)
                
                # Record control signals
                control_history.append((throttle, steering))
                
                # Update visualization
                self.visualizer.update_visualization(
                    step, image, detections, path, control_history, goal_position
                )
                
                # Check if reached goal
                if np.linalg.norm(position - goal_position) < 0.5:
                    print(f"Reached goal! Steps: {step}")
                    break
                
                # Control update frequency
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error at step {step}: {e}")
                continue
        
        # Save final result
        self.visualizer.save_visualization()
        
        print("Demo completed! Press any key to close window...")
        plt.show()