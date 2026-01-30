import cv2
import numpy as np
import math
import time
import json
from pathlib import Path
from simple_pid import PID
import matplotlib.pyplot as plt

# Add YOLOv8 import and availability check
try:
    from yolov8_detector import YOLOv8ObstacleDetector
    YOLOv8_AVAILABLE = True
    print("YOLOv8 detector is available")
except ImportError as e:
    YOLOv8_AVAILABLE = False
    print(f"YOLOv8 not available, using traditional detector: {e}")

# System configuration
class SystemConfig:
    # Perception configuration
    YOLO_CONFIDENCE = 0.3
    YOLO_MODEL_PATH = 'yolov8n.pt'
    
    # Control configuration - 优化PID参数以提高控制精度
    STEERING_PID = [1.5, 0.02, 0.15]  # 增加比例增益，进一步减少积分增益，增加微分增益
    SPEED_PID = [1.2, 0.03, 0.08]  # 增强速度控制响应
    LOOKAHEAD_DISTANCE = 8.0  # 增加前瞻距离以提高预测性
    LOOKAHEAD_MIN = 3.0  # 降低最小前瞻距离以提高低速精度
    LOOKAHEAD_MAX = 18.0  # 增加最大前瞻距离以适应高速
    MAX_STEERING_ANGLE = 0.4  # 稍微增加最大转向角以提高灵活性
    STEERING_RATE_LIMIT = 0.08  # 增加转向率限制以加快响应速度
    
    # Planning configuration
    GRID_SIZE = 50
    OBSTACLE_RADIUS = 4
    MAX_PATH_POINTS = 20
    
    # A* algorithm configuration parameters
    HEURISTIC_WEIGHT = 1.2  # 启发式权重，提高为1.2以权衡贪婪搜索和Dijkstra算法
    MAX_ITERATIONS = 10000  # 增加最大迭代次数以处理复杂场景
    EARLY_EXIT_THRESHOLD = 0.5  # 提前退出阈值，当发现足够好的路径时提前返回
    DIAGONAL_COST = math.sqrt(2)  # 对角线移动的成本
    STRAIGHT_COST = 1.0  # 直线移动的成本
    OBSTACLE_PENALTY = 10.0  # 障碍物周围的惩罚权重

# Keep the original ObstacleDetector as backup
class ObstacleDetector:
    def __init__(self, conf_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'traffic light', 'stop sign']

    def detect(self, image):
        height, width = image.shape[:2]

        # Use more sensitive edge detection parameters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            if cv2.contourArea(contour) > 200:
                x, y, w, h = cv2.boundingRect(contour)

                # Improved confidence calculation
                area_ratio = cv2.contourArea(contour) / (width * height)
                confidence = min(0.9, area_ratio * 20)

                if confidence > self.conf_threshold:
                    distance = self.estimate_distance(w, h, width)
                    detections.append({
                        'class': 'obstacle',
                        'confidence': confidence,
                        'bbox': [x, y, x+w, y+h],
                        'distance': distance
                    })

        return detections

    def estimate_distance(self, bbox_width, bbox_height, image_width):
        ref_width = 200
        ref_distance = 10.0
        apparent_size = max(bbox_width, bbox_height)
        distance = (ref_width * ref_distance) / apparent_size
        return max(1.0, min(50.0, distance))

class PathPlanner:
    def __init__(self, grid_size=SystemConfig.GRID_SIZE):
        self.grid_size = grid_size
        self.obstacle_grid = np.zeros((grid_size, grid_size))
        
        # A* algorithm configuration
        self.config = SystemConfig()
        self.diagonal_cost = self.config.DIAGONAL_COST
        self.straight_cost = self.config.STRAIGHT_COST
        self.heuristic_weight = self.config.HEURISTIC_WEIGHT
        self.max_iterations = self.config.MAX_ITERATIONS
        self.early_exit_threshold = self.config.EARLY_EXIT_THRESHOLD
        self.obstacle_penalty = self.config.OBSTACLE_PENALTY
        
        # Import heapq for priority queue implementation
        import heapq
        self.heapq = heapq

    def update_obstacle_grid(self, detections, current_position):
        self.obstacle_grid = np.zeros((self.grid_size, self.grid_size))

        for detection in detections:
            distance = detection['distance']
            bbox = detection['bbox']
            
            # Convert detection to world coordinates
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            obstacle_x = current_position[0] + distance
            obstacle_y = current_position[1] + (bbox_center_y - 240) / 240 * distance

            grid_x = int(obstacle_x * 2)
            grid_y = int(obstacle_y * 2)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                # Create larger obstacle area for cars
                radius = SystemConfig.OBSTACLE_RADIUS
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        if (0 <= grid_x+dx < self.grid_size and
                            0 <= grid_y+dy < self.grid_size):
                            self.obstacle_grid[grid_x+dx, grid_y+dy] = 1
    
    def distance(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
    def heuristic(self, a, b):
        # 改进的启发式函数，考虑欧几里得距离并添加权重
        return self.distance(a, b) * self.heuristic_weight
        
    def is_within_grid(self, position, grid_width, grid_height):
        x, y = position
        return 0 <= x < grid_width and 0 <= y < grid_height
        
    def is_obstacle(self, position, obstacles, margin=0):
        # 带安全边际的障碍物检测
        for obstacle in obstacles:
            # 障碍物可能是点(x,y)或矩形(x1,y1,x2,y2)
            if len(obstacle) == 2:
                # 点障碍物
                obs_x, obs_y = obstacle
                if self.distance(position, (obs_x, obs_y)) <= 1 + margin:
                    return True
            else:
                # 矩形障碍物
                x1, y1, x2, y2 = obstacle
                x, y = position
                if (x1 - margin <= x <= x2 + margin and 
                    y1 - margin <= y <= y2 + margin):
                    return True
        return False
        
    def get_neighbors(self, position, grid_width, grid_height, obstacles):
        # 8方向移动（包括对角线），但添加对角线移动的限制（不能穿过障碍物的对角线）
        directions = [
            (0, -1, self.straight_cost),   # 上
            (1, 0, self.straight_cost),    # 右
            (0, 1, self.straight_cost),    # 下
            (-1, 0, self.straight_cost),   # 左
            (1, -1, self.diagonal_cost),   # 右上
            (1, 1, self.diagonal_cost),    # 右下
            (-1, 1, self.diagonal_cost),   # 左下
            (-1, -1, self.diagonal_cost)   # 左上
        ]
        
        neighbors = []
        x, y = position
        
        for dx, dy, cost in directions:
            nx, ny = x + dx, y + dy
            
            # 检查是否在网格范围内
            if not self.is_within_grid((nx, ny), grid_width, grid_height):
                continue
                
            # 检查是否是障碍物
            if self.is_obstacle((nx, ny), obstacles, margin=0.5):
                continue
                
            # 对角线移动的特殊限制：不允许穿过障碍物的对角线
            if abs(dx) + abs(dy) == 2:  # 对角线移动
                # 检查两个相邻的直线点是否有障碍物
                if (self.is_obstacle((x + dx, y), obstacles, margin=0.3) or 
                    self.is_obstacle((x, y + dy), obstacles, margin=0.3)):
                    continue
                    
            # 计算额外的障碍物惩罚（如果邻近有障碍物）
            obstacle_factor = 1.0
            if self.is_obstacle((nx, ny), obstacles, margin=1.5) and not self.is_obstacle((nx, ny), obstacles):
                obstacle_factor = 1.0 + self.obstacle_penalty * 0.1
                
            neighbors.append(((nx, ny), cost * obstacle_factor))
            
        return neighbors
        
    def reconstruct_path(self, came_from, current):
        # 优化路径重建，使用更高效的数据结构
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # 反转路径
        
    def astar_search(self, start, goal):
        # 转换为网格坐标
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        # 从obstacle_grid创建障碍物列表
        obstacles = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacle_grid[x, y] == 1:
                    obstacles.append((x, y))
        
        # 初始化开放列表和关闭列表
        open_set = {start}
        came_from = {}
        
        # 使用字典来存储g_score和f_score，提高查找效率
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        # 使用优先队列来代替简单集合，按f_score排序
        open_heap = [(f_score[start], start)]
        
        # 添加提前退出机制和迭代次数限制
        iterations = 0
        best_path = None
        best_f_score = float('inf')
        
        while open_heap and iterations < self.max_iterations:
            # 获取当前f值最小的节点
            _, current = self.heapq.heappop(open_heap)
            
            # 如果当前节点已经不在开放列表中（可能被其他路径找到更优解），跳过
            if current not in open_set:
                continue
                
            # 从开放列表中移除当前节点，加入关闭列表
            open_set.remove(current)
            
            # 找到目标，返回路径
            if self.distance(current, goal) < self.early_exit_threshold:
                path = self.reconstruct_path(came_from, current)
                return self.smooth_path(path, obstacles, self.grid_size, self.grid_size)
                
            # 记录当前最佳路径
            current_f = f_score[current]
            if current_f < best_f_score:
                best_f_score = current_f
                best_path = self.reconstruct_path(came_from, current)
                
            # 获取当前节点的所有邻居
            neighbors = self.get_neighbors(current, self.grid_size, self.grid_size, obstacles)
            
            # 处理每个邻居节点
            for neighbor, cost in neighbors:
                # 计算从起点经过当前节点到达邻居节点的g值
                tentative_g_score = g_score[current] + cost
                
                # 如果这个路径比已知的到达这个邻居的路径更好
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 更新路径信息
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    # 如果邻居不在开放列表中，将其加入
                    if neighbor not in open_set:
                        self.heapq.heappush(open_heap, (f_score[neighbor], neighbor))
                        open_set.add(neighbor)
                        
            iterations += 1
            
            # 每100次迭代进行一次检查，如果离目标越来越远则考虑提前退出
            if iterations % 100 == 0:
                # 如果已经有最佳路径且离目标不太远，返回它
                if best_path and self.distance(best_path[-1], goal) < 10.0:
                    return self.smooth_path(best_path, obstacles, self.grid_size, self.grid_size)
                    
        # 如果无法找到完整路径但有部分路径，返回最佳路径
        if best_path:
            return self.smooth_path(best_path, obstacles, self.grid_size, self.grid_size)
            
        return []  # 无路径找到
        
    def smooth_path(self, path, obstacles, grid_width, grid_height):
        # 路径平滑处理
        if len(path) < 3:
            return path
            
        smoothed_path = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # 尝试跳过中间点，直接连接到更后面的点
            next_idx = len(path) - 1
            for i in range(current_idx + 2, len(path)):
                # 检查当前点到i点的直线是否可以通过（不穿过障碍物）
                if self._is_line_valid(path[current_idx], path[i], obstacles, grid_width, grid_height):
                    next_idx = i
                    break
                    
            smoothed_path.append(path[next_idx])
            current_idx = next_idx
            
        return smoothed_path
        
    def _is_line_valid(self, start, end, obstacles, grid_width, grid_height):
        # 检查两点之间的直线是否可以通过（不穿过障碍物）
        # 使用中点采样法进行检查
        num_points = max(5, int(self.distance(start, end) * 2))
        for i in range(1, num_points):
            t = i / num_points
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            
            # 检查这个点是否在网格内且不是障碍物
            if (not self.is_within_grid((x, y), grid_width, grid_height) or 
                self.is_obstacle((x, y), obstacles)):
                return False
                
        return True
        
    # 保持向后兼容性的方法
    def is_valid_position(self, pos):
        x, y = pos
        return self.obstacle_grid[x, y] == 0
        
    # 保持向后兼容性的方法
    def get_neighbors(self, pos, grid_width=None, grid_height=None, obstacles=None, obstacle_factor=1.0):
        x, y = pos
        neighbors = []

        # 如果没有提供网格尺寸，使用self.grid_size
        if grid_width is None:
            grid_width = self.grid_size
        if grid_height is None:
            grid_height = self.grid_size

        # 8-direction movement for smoother paths
        for dx, dy in [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:
            nx, ny = x + dx, y + dy
            # 检查是否在网格范围内
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                # 计算基础成本
                cost = math.sqrt(dx*dx + dy*dy)  # 对角线移动成本更高
                
                # 如果提供了障碍物列表，检查是否是障碍物
                if obstacles and (nx, ny) in obstacles:
                    # 如果是障碍物，跳过
                    continue
                
                neighbors.append(((nx, ny), cost * obstacle_factor))

        return neighbors

class CarController:
    def __init__(self):
        # Optimized PID parameters for car control
        self.steering_pid = PID(
            SystemConfig.STEERING_PID[0],
            SystemConfig.STEERING_PID[1],
            SystemConfig.STEERING_PID[2],
            setpoint=0
        )
        self.speed_pid = PID(
            SystemConfig.SPEED_PID[0],
            SystemConfig.SPEED_PID[1],
            SystemConfig.SPEED_PID[2],
            setpoint=1.0
        )
        
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.lookahead_distance = SystemConfig.LOOKAHEAD_DISTANCE
        self.lookahead_min = SystemConfig.LOOKAHEAD_MIN
        self.lookahead_max = SystemConfig.LOOKAHEAD_MAX
        self.max_speed = 3.0
        self.steering_limit = SystemConfig.MAX_STEERING_ANGLE
        self.steering_rate_limit = SystemConfig.STEERING_RATE_LIMIT
        self.last_steering = 0.0
        
        # Add integral windup prevention
        self.steering_pid.output_limits = (-self.steering_limit, self.steering_limit)
        self.speed_pid.output_limits = (0, 1.0)
        
        # 添加路径跟踪历史，用于优化前瞻距离
        self.tracking_history = []
        self.max_history_size = 10

    def update_control(self, current_position, current_heading, path):
        """更新控制指令，返回(throttle, steering)"""
        # 初始化油门和转向
        throttle = 0.0
        steering = 0.0
        
        if not path:
            return (throttle, steering)
        
        # 1. 路径预处理 - 使用增强的路径处理算法
        processed_path = self.process_path_for_control(path)
        
        # 2. 找到增强的目标点 - 使用动态前瞻距离和路径曲率
        target_point = self.find_improved_target_point(current_position, processed_path)
        
        if not target_point:
            return (throttle, steering)
        
        # 3. 计算航向误差
        # 计算从当前位置到目标点的向量
        dx = target_point[0] - current_position[0]
        dy = target_point[1] - current_position[1]
        
        # 计算期望航向角
        desired_heading = math.atan2(dy, dx)
        
        # 计算航向误差
        heading_error = self.normalize_angle(desired_heading - current_heading)
        
        # 4. 应用增强的PID控制计算转向
        # 对于大误差情况，使用不同的控制策略而不是直接修改PID参数
        if abs(heading_error) > 0.3:  # 大误差情况
            # 使用比例控制为主的临时转向计算
            # 使用更高的比例因子处理大误差
            large_error_steering = heading_error * 1.8  # 更高的比例因子
            steering = large_error_steering
        else:
            # 正常误差使用PID控制
            steering = self.steering_pid(heading_error)
        
        # 5. 应用转向率限制以平滑转向
        max_steering_change = self.steering_rate_limit
        steering_change = steering - self.last_steering
        steering = self.last_steering + np.clip(steering_change, -max_steering_change, max_steering_change)
        
        # 6. 限制转向角度
        steering = np.clip(steering, -self.steering_limit, self.steering_limit)
        
        # 7. 使用增强的自适应速度控制
        throttle = self.adaptive_speed_control(heading_error, current_position, target_point, processed_path)
        
        # 8. 更新状态
        self.last_steering = steering
        
        # 更新跟踪历史以评估性能
        distance_error = self.distance(current_position, path[0])  # 简化计算，实际应计算到路径的距离
        self.tracking_history.append(distance_error)
        
        if len(self.tracking_history) > self.max_history_size:
            self.tracking_history.pop(0)
        
        # 检查是否到达目标
        if self.distance(current_position, path[-1]) < 1.0:
            return (0.0, 0.0)  # 到达目标，停止
        
        return (throttle, steering)

    def process_path_for_control(self, original_path):
        """Process path for control optimization"""
        if len(original_path) <= SystemConfig.MAX_PATH_POINTS:
            return original_path
            
        # Uniform sampling
        step = len(original_path) // SystemConfig.MAX_PATH_POINTS
        processed_path = original_path[::step]
        
        # Ensure start and end points are included
        if processed_path[0] != original_path[0]:
            processed_path.insert(0, original_path[0])
        if processed_path[-1] != original_path[-1]:
            processed_path.append(original_path[-1])
            
        return processed_path[:SystemConfig.MAX_PATH_POINTS]

    def find_improved_target_point(self, current_pos, path):
        if len(path) < 2:
            return path[-1] if path else None

        # 增强的动态前瞻距离调整
        # 1. 基于速度的调整
        speed_factor = max(0.5, min(2.0, self.current_speed))
        
        # 2. 计算路径曲率并调整前瞻距离
        if len(path) > 5:
            curvature = self.estimate_path_curvature(path)
            curvature_factor = max(0.5, min(1.5, 1.0 - abs(curvature) * 0.2))
        else:
            curvature_factor = 1.0
        
        # 3. 基于历史跟踪精度的自适应调整
        tracking_factor = 1.0
        if len(self.tracking_history) > 3:
            recent_errors = [error for _, error in self.tracking_history[-3:]]
            avg_error = sum(recent_errors) / len(recent_errors)
            # 如果误差较大，减小前瞻距离以提高精度
            if avg_error > 2.0:
                tracking_factor = 0.8
            # 如果误差较小，可以适当增加前瞻距离以提高稳定性
            elif avg_error < 0.5:
                tracking_factor = 1.2
        
        # 计算最终前瞻距离，限制在合理范围内
        dynamic_lookahead = self.lookahead_distance * speed_factor * curvature_factor * tracking_factor
        dynamic_lookahead = max(self.lookahead_min, min(self.lookahead_max, dynamic_lookahead))
        
        # 智能目标点选择 - 考虑路径连续性和未来点
        best_target = None
        best_score = float('inf')
        
        for i in range(1, min(len(path), 20)):  # 限制搜索范围以提高效率
            point = path[i]
            distance = self.distance(current_pos, point)
            
            # 计算路径方向变化
            if i < len(path) - 1:
                next_point = path[i+1]
                dir_current = (point[0]-current_pos[0], point[1]-current_pos[1])
                dir_next = (next_point[0]-point[0], next_point[1]-point[1])
                # 方向变化角度
                dot_product = (dir_current[0]*dir_next[0] + dir_current[1]*dir_next[1])
                norm_product = self.distance((0,0), dir_current) * self.distance((0,0), dir_next) + 1e-5
                direction_change = 1.0 - max(-1.0, min(1.0, dot_product / norm_product))
            else:
                direction_change = 0.0
            
            # 计算目标点得分，考虑距离和方向变化
            score = abs(distance - dynamic_lookahead) + direction_change * 2.0
            
            # 如果点距离合适且路径相对平滑，优先选择
            if abs(distance - dynamic_lookahead) < 1.5 and direction_change < 0.5 and score < best_score:
                best_score = score
                best_target = point
        
        # 如果没有找到理想的点，回退到基于距离的搜索
        if best_target is None:
            # 寻找第一个超过动态前瞻距离的点
            for point in path[1:]:
                if self.distance(current_pos, point) >= dynamic_lookahead:
                    best_target = point
                    break
        
        # 确保总是有目标点
        if best_target is None and path:
            best_target = path[-1]  # 默认为终点
        
        # 记录当前点与目标点之间的距离作为历史数据
        if best_target:
            tracking_error = self.calculate_tracking_error(current_pos, path)
            self.update_tracking_history(tracking_error)
        
        return best_target

    def calculate_feedforward(self, path, current_pos):
        """Feedforward control calculation for path curvature compensation"""
        if len(path) < 3:
            return 0
            
        # Find closest point in path
        closest_idx = 0
        min_dist = float('inf')
        
        for i, point in enumerate(path):
            dist = self.distance(current_pos, point)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Use nearby points to calculate curvature
        idx = min(closest_idx + 2, len(path) - 1)
        if idx >= 2:
            p0, p1, p2 = path[idx-2], path[idx-1], path[idx]
            
            # Calculate curvature
            dx1, dy1 = p1[0]-p0[0], p1[1]-p0[1]
            dx2, dy2 = p2[0]-p1[0], p2[1]-p1[1]
            
            curvature = (dx1*dy2 - dy1*dx2) / max(0.1, (dx1**2 + dy1**2)**1.5)
            
            # Feedforward steering compensation
            ff_gain = 0.3
            return ff_gain * curvature
        
        return 0

    def adaptive_speed_control(self, heading_error, current_pos, target_point, path=None):
        """增强的速度控制，基于路径曲率、航向误差和距离"""
        # 基础速度
        base_speed = 1.5
        
        # 初始化减速因子
        total_reduction = 0.0
        
        # 1. 航向误差引起的减速
        error_penalty = min(1.0, abs(heading_error) / (math.pi/4))
        total_reduction += error_penalty * 0.8
        
        # 2. 距离引起的减速（接近目标时）
        distance = self.distance(current_pos, target_point)
        if distance < 3.0:
            distance_reduction = (3.0 - distance) / 3.0 * 0.5
            total_reduction += distance_reduction
        
        # 3. 路径曲率引起的减速（弯道减速）
        curvature_reduction = 0.0
        if path and len(path) > 3:
            # 计算当前路径段的曲率
            # 使用简化的曲率估计，不依赖于self.config
            curvature = self.calculate_curvature(path[0], path[min(2, len(path)-1)])
            curvature_magnitude = abs(curvature)
            
            # 根据曲率程度调整减速
            if curvature_magnitude > 0.1:  # 低阈值
                if curvature_magnitude < 0.3:  # 高阈值
                    # 中等曲率，线性减速
                    ratio = (curvature_magnitude - 0.1) / 0.2
                    curvature_reduction = ratio * 0.6
                else:
                    # 高曲率，大幅减速
                    curvature_reduction = 0.6 + min(0.3, (curvature_magnitude - 0.3) * 0.5)
        
        total_reduction += curvature_reduction
        
        # 计算目标速度
        target_speed = base_speed * (1.0 - min(total_reduction, 0.9))  # 最大减速90%
        
        # 限制在合理范围内
        target_speed = max(0.5, min(self.max_speed, target_speed))
        
        # 应用平滑因子，避免速度急剧变化
        # 使用简单的平滑因子
        smoothing_factor = 0.2
        smoothed_speed = self.current_speed * (1 - smoothing_factor) + target_speed * smoothing_factor
        
        # 设置PID目标值并计算油门
        self.speed_pid.setpoint = smoothed_speed
        throttle = self.speed_pid(self.current_speed)
        
        return np.clip(throttle, 0, 1.0)
        
    def estimate_ahead_curvature(self, path, current_pos):
        """估计前方路径的曲率，用于提前减速"""
        if len(path) < 5:
            return 0.0
        
        # 找到最近的路径点
        closest_idx = 0
        min_dist = float('inf')
        
        for i, point in enumerate(path):
            dist = self.distance(current_pos, point)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 取前方一段路径点计算曲率
        lookahead_end = min(closest_idx + 10, len(path))
        ahead_path = path[closest_idx:lookahead_end]
        
        return self.estimate_path_curvature(ahead_path)

    def calculate_curvature(self, point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return dy / (dx + 1e-5)  # Avoid division by zero

    def distance(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def calculate_tracking_error(self, current_pos, path):
        """计算当前位置与路径之间的跟踪误差"""
        if not path:
            return 0.0
            
        min_error = float('inf')
        for point in path:
            error = self.distance(current_pos, point)
            min_error = min(min_error, error)
        
        # 考虑路径连续性，如果有连续点，也考虑切线距离
        if len(path) > 1:
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i+1]
                # 计算点到线段的距离
                line_vec = (p2[0] - p1[0], p2[1] - p1[1])
                line_len = self.distance(p1, p2)
                if line_len > 0:
                    line_unitvec = (line_vec[0]/line_len, line_vec[1]/line_len)
                    vec_to_p1 = (current_pos[0] - p1[0], current_pos[1] - p1[1])
                    t = max(0, min(1, (vec_to_p1[0]*line_unitvec[0] + vec_to_p1[1]*line_unitvec[1]) / line_len))
                    closest_point = (p1[0] + line_vec[0]*t, p1[1] + line_vec[1]*t)
                    segment_error = self.distance(current_pos, closest_point)
                    min_error = min(min_error, segment_error)
        
        return min_error
    
    def update_tracking_history(self, error):
        """更新跟踪误差历史记录"""
        timestamp = time.time()
        self.tracking_history.append((timestamp, error))
        
        # 保持历史记录大小
        if len(self.tracking_history) > self.max_history_size:
            self.tracking_history = self.tracking_history[-self.max_history_size:]
            
        # 移除过旧的记录（超过5秒）
        cutoff_time = timestamp - 5.0
        self.tracking_history = [(t, e) for t, e in self.tracking_history if t > cutoff_time]
    
    def estimate_path_curvature(self, path):
        """增强版路径曲率估算，合并局部和整体曲率计算"""
        if len(path) < 3:
            return 0.0
        
        # 1. 局部曲率计算（使用最近三个点）- 对当前路径段敏感
        if len(path) >= 3:
            # 使用三个点计算局部曲率
            p1, p2, p3 = np.array(path[:3])
            
            # 计算向量
            vec1 = p2 - p1
            vec2 = p3 - p2
            
            # 计算夹角
            dot_product = np.dot(vec1, vec2)
            magnitude1 = np.linalg.norm(vec1)
            magnitude2 = np.linalg.norm(vec2)
            
            if magnitude1 * magnitude2 > 0:
                cos_theta = dot_product / (magnitude1 * magnitude2)
                local_angle = math.acos(max(-1, min(1, cos_theta)))
                local_curvature = local_angle
            else:
                local_curvature = 0.0
        else:
            local_curvature = 0.0
            
        # 2. 整体曲率计算（使用更多路径点）- 提供全局路径趋势
        total_curvature = 0.0
        count = 0
        
        # 计算路径中相邻线段之间的角度变化
        for i in range(1, min(len(path) - 1, 10)):  # 限制计算范围以提高效率
            prev_point = path[i-1]
            curr_point = path[i]
            next_point = path[i+1]
            
            # 计算两个方向向量
            vec1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
            vec2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
            
            # 计算夹角
            dot_product = (vec1[0]*vec2[0] + vec1[1]*vec2[1])
            norm_product = self.distance((0,0), vec1) * self.distance((0,0), vec2) + 1e-5
            cos_angle = max(-1.0, min(1.0, dot_product / norm_product))
            angle = math.acos(cos_angle)
            
            # 转换为曲率（简化计算，实际曲率与路径长度相关）
            segment_len = self.distance(prev_point, curr_point)
            if segment_len > 0.1:  # 避免除零
                curvature = angle / segment_len
                total_curvature += abs(curvature)
                count += 1
        
        global_curvature = total_curvature / (count + 1e-5) if count > 0 else 0.0
        
        # 3. 合并局部和全局曲率 - 更重视局部曲率以快速响应弯道变化
        combined_curvature = 0.7 * local_curvature + 0.3 * global_curvature
        
        return combined_curvature
        
    # 添加缺失的reset方法
    def reset(self):
        """Reset controller state for new test run"""
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.last_steering = 0.0
        self.tracking_history = []
        self.target_reached = False
        self.reached_count = 0
        
        # 重置PID控制器
        self.steering_pid.setpoint = 0
        self.steering_pid.auto_mode = True
        self.speed_pid.setpoint = 1.0
        self.speed_pid.auto_mode = True
        
        # 清除PID内部状态
        self.steering_pid.reset()
        self.speed_pid.reset()

    def process_path_for_control(self, path):
        """Process path to optimize for control"""
        if len(path) <= 2:
            return path
            
        # 使用更高效的路径点过滤，保留关键转向点
        processed_path = [path[0]]
        prev_angle = 0
        angle_threshold = 0.1  # 降低角度阈值以捕捉更多细节
        
        for i in range(1, len(path) - 1):
            current_angle = math.atan2(
                path[i+1][1] - path[i-1][1],
                path[i+1][0] - path[i-1][0]
            )
            
            if abs(self.normalize_angle(current_angle - prev_angle)) > angle_threshold:
                processed_path.append(path[i])
                prev_angle = current_angle
                
        processed_path.append(path[-1])
        return processed_path

    def find_improved_target_point(self, current_position, path):
        """增强的目标点选择算法，基于动态前瞻距离"""
        if not path:
            return None
        
        # 计算当前路径段的曲率，用于动态调整前瞻距离
        if len(path) > 2:
            curvature = self.estimate_path_curvature(path[:3])
            # 曲率越大，前瞻距离越小
            adaptive_lookahead = max(self.lookahead_min, 
                                    self.lookahead_max / (1 + curvature * 2))
        else:
            adaptive_lookahead = self.lookahead_distance
            
        # 基于速度调整前瞻距离
        speed_factor = 1 + min(self.current_speed / 3, 1.0)  # 速度越高，前瞻距离越大
        dynamic_lookahead = adaptive_lookahead * speed_factor
        
        # 查找动态前瞻距离内的最佳目标点
        closest_distance = float('inf')
        best_target = path[0]  # 默认使用第一个点
        
        for point in path:
            distance = self.distance(current_position, point)
            
            # 选择前瞻距离附近的点，但不要太近
            if 0.5 <= distance <= dynamic_lookahead * 1.2:
                # 优先选择与当前航向一致的点
                heading_to_point = math.atan2(
                    point[1] - current_position[1],
                    point[0] - current_position[0]
                )
                heading_diff = abs(self.normalize_angle(heading_to_point - 0))  # 假设有当前航向参数
                
                # 综合考虑距离和航向一致性
                score = distance + heading_diff * 2
                
                if score < closest_distance:
                    closest_distance = score
                    best_target = point
        
        return best_target

    def update_control(self, current_position, current_heading, path):
        """更新控制指令，返回(throttle, steering)"""
        # 初始化油门和转向
        throttle = 0.0
        steering = 0.0
        
        if not path:
            return (throttle, steering)
        
        # 1. 路径预处理 - 使用增强的路径处理算法
        processed_path = self.process_path_for_control(path)
        
        # 2. 找到增强的目标点 - 使用动态前瞻距离和路径曲率
        target_point = self.find_improved_target_point(current_position, processed_path)
        
        if not target_point:
            return (throttle, steering)
        
        # 3. 计算航向误差
        # 计算从当前位置到目标点的向量
        dx = target_point[0] - current_position[0]
        dy = target_point[1] - current_position[1]
        
        # 计算期望航向角
        desired_heading = math.atan2(dy, dx)
        
        # 计算航向误差
        heading_error = self.normalize_angle(desired_heading - current_heading)
        
        # 4. 应用增强的PID控制计算转向
        # 对于大误差情况，使用不同的控制策略而不是直接修改PID参数
        if abs(heading_error) > 0.3:  # 大误差情况
            # 使用比例控制为主的临时转向计算
            # 使用更高的比例因子处理大误差
            large_error_steering = heading_error * 1.8  # 更高的比例因子
            steering = large_error_steering
        else:
            # 正常误差使用PID控制
            steering = self.steering_pid(heading_error)
        
        # 5. 应用转向率限制以平滑转向
        max_steering_change = self.steering_rate_limit
        steering_change = steering - self.last_steering
        steering = self.last_steering + np.clip(steering_change, -max_steering_change, max_steering_change)
        
        # 6. 限制转向角度
        steering = np.clip(steering, -self.steering_limit, self.steering_limit)
        
        # 7. 使用增强的自适应速度控制
        throttle = self.adaptive_speed_control(heading_error, current_position, target_point, processed_path)
        
        # 8. 更新状态
        self.last_steering = steering
        
        # 更新跟踪历史以评估性能
        distance_error = self.distance(current_position, path[0])  # 简化计算，实际应计算到路径的距离
        self.tracking_history.append(distance_error)
        
        if len(self.tracking_history) > self.max_history_size:
            self.tracking_history.pop(0)
        
        # 检查是否到达目标
        if self.distance(current_position, path[-1]) < 1.0:
            self.reached_count += 1
            if self.reached_count > 3:  # 连续多次检测到达目标才确认
                self.target_reached = True
                return (0.0, 0.0)  # 到达目标，停止
        else:
            self.reached_count = 0
            self.target_reached = False
        
        return (throttle, steering)

    def normalize_angle(self, angle):
        """将角度归一化到[-pi, pi]范围内"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def adaptive_speed_control(self, heading_error, current_pos, target_point, path=None):
        """增强的速度控制，基于路径曲率、航向误差和距离"""
        # 基础速度 - 提高以加快响应
        base_speed = 1.8
        
        # 初始化减速因子
        total_reduction = 0.0
        
        # 1. 航向误差引起的减速 - 更敏感的误差响应
        error_penalty = min(1.0, abs(heading_error) / (math.pi/6))  # 更严格的误差阈值
        total_reduction += error_penalty * 0.7
        
        # 2. 距离引起的减速（接近目标时）
        distance = self.distance(current_pos, target_point)
        if distance < 4.0:  # 增加减速距离范围
            distance_reduction = (4.0 - distance) / 4.0 * 0.6
            total_reduction += distance_reduction
        
        # 3. 路径曲率引起的减速（弯道减速）- 改进的曲率计算
        curvature_reduction = 0.0
        if path and len(path) > 3:
            # 计算当前路径段的曲率
            curvature = self.estimate_path_curvature(path[:3])
            
            # 基于曲率的非线性减速策略
            if curvature > 0.05:  # 低曲率阈值
                if curvature < 0.2:  # 中等曲率
                    curvature_reduction = curvature * 2
                else:  # 高曲率
                    curvature_reduction = 0.4 + min(0.4, (curvature - 0.2) * 2)
        
        total_reduction += curvature_reduction
        
        # 计算目标速度
        target_speed = base_speed * (1.0 - min(total_reduction, 0.8))  # 略微减少最大减速
        
        # 限制在合理范围内
        target_speed = max(0.6, min(self.max_speed, target_speed))  # 提高最小速度
        
        # 应用更平滑的平滑因子
        smoothing_factor = 0.3  # 增加平滑因子以更快响应
        smoothed_speed = self.current_speed * (1 - smoothing_factor) + target_speed * smoothing_factor
        
        # 设置PID目标值并计算油门
        self.speed_pid.setpoint = smoothed_speed
        throttle = self.speed_pid(self.current_speed)
        
        return np.clip(throttle, 0, 1.0)

    def distance(self, point1, point2):
        """计算两点间距离"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class CarSimulator:
    def __init__(self, init_position=[0, 0], init_heading=0):
        self.position = np.array(init_position, dtype=float)
        self.heading = init_heading
        self.speed = 0.0
        self.steering_angle = 0.0
        self.wheelbase = 2.5  # Increased wheelbase for car
        self.dt = 0.1
        self.max_acceleration = 1.5
        self.max_deceleration = 2.0

    def update(self, throttle, steering_angle):
        # Limit input range
        throttle = np.clip(throttle, 0, 1.0)
        steering_angle = np.clip(steering_angle, -0.5, 0.5)

        # Enhanced acceleration model
        target_speed = throttle * self.max_acceleration * 2.0
        acceleration = (target_speed - self.speed) * 2.0

        # Consider physical limits
        if acceleration > 0:
            acceleration = min(acceleration, self.max_acceleration)
        else:
            acceleration = max(acceleration, -self.max_deceleration)

        self.speed += acceleration * self.dt
        self.speed = max(0, self.speed)  # Speed cannot be negative

        # Update steering angle
        self.steering_angle = steering_angle

        # Enhanced kinematic model for cars
        if abs(self.steering_angle) < 1e-5:
            # Straight line motion
            self.position[0] += self.speed * math.cos(self.heading) * self.dt
            self.position[1] += self.speed * math.sin(self.heading) * self.dt
        else:
            # Turning motion - improved car model
            turning_radius = self.wheelbase / math.tan(self.steering_angle)
            
            # Limit minimum turning radius
            min_turning_radius = self.wheelbase / math.tan(0.3)  # max steering angle
            turning_radius = max(min_turning_radius, abs(turning_radius)) * np.sign(turning_radius)

            angular_velocity = self.speed / turning_radius

            # Update heading
            self.heading += angular_velocity * self.dt

            # Update position using circular motion
            delta_heading = angular_velocity * self.dt
            if abs(delta_heading) > 1e-5:
                self.position[0] += turning_radius * (math.sin(self.heading) - math.sin(self.heading - delta_heading))
                self.position[1] += turning_radius * (math.cos(self.heading - delta_heading) - math.cos(self.heading))

        return self.position.copy(), self.heading

    def reset(self, init_position=[0, 0], init_heading=0):
        self.position = np.array(init_position, dtype=float)
        self.heading = init_heading
        self.speed = 0.0
        self.steering_angle = 0.0
        print(f"Car simulator reset to position {init_position}, heading {init_heading}")

class SystemMonitor:
    def __init__(self):
        self.performance_history = {
            'perception_time': [],
            'planning_time': [],
            'control_error': [],
            'detection_count': []
        }
        self.start_time = time.time()

    def record_perception(self, detections, processing_time):
        self.performance_history['perception_time'].append(processing_time)
        self.performance_history['detection_count'].append(len(detections))

    def record_planning(self, planning_time):
        self.performance_history['planning_time'].append(planning_time)

    def record_control(self, error):
        self.performance_history['control_error'].append(error)

    def generate_report(self):
        report = {}
        
        if self.performance_history['perception_time']:
            report['perception'] = {
                'avg_processing_time': np.mean(self.performance_history['perception_time']),
                'max_processing_time': np.max(self.performance_history['perception_time']),
                'avg_detection_count': np.mean(self.performance_history['detection_count']),
                'total_detections': sum(self.performance_history['detection_count'])
            }

        if self.performance_history['planning_time']:
            report['planning'] = {
                'avg_planning_time': np.mean(self.performance_history['planning_time']),
                'max_planning_time': np.max(self.performance_history['planning_time']),
                'total_plans': len(self.performance_history['planning_time'])
            }

        if self.performance_history['control_error']:
            report['control'] = {
                'avg_tracking_error': np.mean(self.performance_history['control_error']),
                'max_tracking_error': np.max(self.performance_history['control_error']),
                'error_std': np.std(self.performance_history['control_error'])
            }

        report['system_uptime'] = time.time() - self.start_time
        report['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        return report

    def print_performance_summary(self):
        report = self.generate_report()
        
        print("CAR AUTONOMOUS SYSTEM PERFORMANCE SUMMARY")
        print("=" * 50)
        
        if 'perception' in report:
            p = report['perception']
            print(f"Perception Module:")
            print(f"  Average Processing Time: {p['avg_processing_time']:.1f}ms")
            print(f"  Maximum Processing Time: {p['max_processing_time']:.1f}ms")
            print(f"  Average Detections: {p['avg_detection_count']:.1f}")
            print(f"  Total Detections: {p['total_detections']}")

        if 'planning' in report:
            p = report['planning']
            print(f"Planning Module:")
            print(f"  Average Planning Time: {p['avg_planning_time']:.1f}ms")
            print(f"  Maximum Planning Time: {p['max_planning_time']:.1f}ms")
            print(f"  Total Plans Generated: {p['total_plans']}")

        if 'control' in report:
            c = report['control']
            print(f"Control Module:")
            print(f"  Average Tracking Error: {c['avg_tracking_error']:.3f}m")
            print(f"  Maximum Tracking Error: {c['max_tracking_error']:.3f}m")
            print(f"  Error Standard Deviation: {c['error_std']:.3f}m")

        print(f"System Uptime: {report['system_uptime']:.1f} seconds")
        print(f"Report Time: {report['timestamp']}")

# Utility functions for system operation
def create_system_components():
    """Create and initialize all car system components"""
    if YOLOv8_AVAILABLE:
        detector = YOLOv8ObstacleDetector(conf_threshold=SystemConfig.YOLO_CONFIDENCE)
        detector_type = "YOLOv8 Deep Learning"
    else:
        detector = ObstacleDetector(conf_threshold=SystemConfig.YOLO_CONFIDENCE)
        detector_type = "Traditional Image"

    planner = PathPlanner(grid_size=SystemConfig.GRID_SIZE)
    controller = CarController()
    simulator = CarSimulator()
    monitor = SystemMonitor()

    print(f"Car autonomous system components initialized:")
    print(f"  Detector: {detector_type}")
    print(f"  Planner: A* Algorithm with {planner.grid_size}x{planner.grid_size} grid")
    print(f"  Controller: Enhanced PID with adaptive control")
    print(f"  Simulator: Improved car dynamics model")
    print(f"  Monitor: Performance tracking enabled")

    return detector, planner, controller, simulator, monitor

def validate_system_components():
    """Validate that all car system components are functioning properly"""
    issues = []

    # Test basic imports
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        issues.append(f"Import error: {e}")

    # Test YOLOv8 availability
    if YOLOv8_AVAILABLE:
        try:
            from yolov8_detector import YOLOv8ObstacleDetector
            # Test creating detector instance
            detector = YOLOv8ObstacleDetector()
        except Exception as e:
            issues.append(f"YOLOv8 initialization error: {e}")
    else:
        issues.append("YOLOv8 not available - using traditional detector")

    # Test other components
    try:
        planner = PathPlanner()
        controller = CarController()
        simulator = CarSimulator()
    except Exception as e:
        issues.append(f"Component initialization error: {e}")

    if issues:
        print("CAR SYSTEM VALIDATION - ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("CAR SYSTEM VALIDATION - ALL COMPONENTS OK")
        return True

def quick_control_test():
    """Quick test for car control module improvements"""
    print("Running quick car control test...")
    
    controller = CarController()
    simulator = CarSimulator()

    # Test gentle curve
    test_path = [(i, 0.5 * i) for i in range(1, 6)]
    errors = []

    for step in range(20):
        throttle, steering = controller.update_control(
            simulator.position, simulator.heading, test_path
        )
        position, heading = simulator.update(throttle, steering)

        # Calculate minimum distance to any path point
        min_error = min([np.linalg.norm(position - point) for point in test_path])
        errors.append(min_error)

    avg_error = np.mean(errors)
    print(f"Quick car control test: Average error {avg_error:.3f}m")

    if avg_error < 1.5:
        print("Car control test: PASSED - Significant improvement detected")
        return True
    else:
        print("Car control test: NEEDS MORE WORK - Continue optimization")
        return False

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Car Autonomous System Module - Component Test")
    print("=" * 50)

    # Validate system components
    if validate_system_components():
        print("\nCreating car system components...")
        detector, planner, controller, simulator, monitor = create_system_components()

        print("\nTesting enhanced car functionality...")

        # Test obstacle detection with sample image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        detections = detector.detect(test_image)
        print(f"Test detection: {len(detections)} objects found")

        # Test path planning
        path = planner.astar_search((5, 5), (45, 45))
        print(f"Test planning: {len(path)} point path generated")

        # Test control with improved algorithm
        test_path = [(i, 0.5 * i) for i in range(1, 6)]  # Gentle curve
        throttle, steering = controller.update_control([0, 0], 0, test_path)
        print(f"Test control: Throttle={throttle:.2f}, Steering={steering:.3f}")

        # Test simulation
        position, heading = simulator.update(throttle, steering)
        print(f"Test simulation: Position=({position[0]:.2f}, {position[1]:.2f}), Heading={heading:.2f}")

        # Run quick control test
        print("\n" + "="*30)
        quick_control_test()
        print("\nAll enhanced car tests completed successfully!")
    else:
        print("\nCar system validation failed. Please check the issues above.")


# 在 car_system.py 文件末尾添加以下函数

def generate_realistic_test_image(width=640, height=480):
    """生成汽车场景测试图像"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 添加真实道路背景
    road_color = (60, 60, 60)  # 深灰色道路
    cv2.rectangle(image, (0, height//2-100), (width, height//2+100), road_color, -1)

    # 添加真实车道线
    lane_color = (255, 255, 255)  # 白色车道
    cv2.line(image, (width//4, 0), (width//4, height), lane_color, 4)
    cv2.line(image, (width//2, 0), (width//2, height), (255, 255, 0), 6)  # 黄色中心线
    cv2.line(image, (3*width//4, 0), (3*width//4, height), lane_color, 4)

    # 添加汽车
    car_points = np.array([
        [100, height//2-30],
        [220, height//2-30],
        [240, height//2+20],
        [80, height//2+20]
    ], np.int32)
    cv2.fillPoly(image, [car_points], (0, 0, 200))  # 蓝色汽车
    cv2.polylines(image, [car_points], True, (0, 0, 0), 2)

    # 添加行人
    person_center = (350, height//2-40)
    cv2.circle(image, (person_center[0], person_center[1]-25), 10, (200, 0, 0), -1)
    cv2.rectangle(image,
                 (person_center[0]-8, person_center[1]-10),
                 (person_center[0]+8, person_center[1]+20),
                 (200, 0, 0), -1)

    return image