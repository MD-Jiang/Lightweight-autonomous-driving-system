import cv2
import numpy as np
import math
import time
import json
from pathlib import Path
from simple_pid import PID
import matplotlib.pyplot as plt

# ==================== 核心汽车系统类 ====================

class ObstacleDetector:
    def __init__(self, conf_threshold=0.3):  # 提高阈值以获得更可靠的检测
        self.conf_threshold = conf_threshold
        self.classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'traffic light', 'stop sign']

    def detect(self, image):
        height, width = image.shape[:2]

        # 使用更敏感的边缘检测参数
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            # 提高面积阈值以避免检测小噪声
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)

                # 改进的置信度计算
                area_ratio = cv2.contourArea(contour) / (width * height)
                confidence = min(0.9, area_ratio * 15)  # 调整置信度系数

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
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.obstacle_grid = np.zeros((grid_size, grid_size))

    def update_obstacle_grid(self, detections, current_position):
        self.obstacle_grid = np.zeros((self.grid_size, self.grid_size))

        for detection in detections:
            distance = detection['distance']
            bbox = detection['bbox']
            
            # 转换检测到世界坐标
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            obstacle_x = current_position[0] + distance
            obstacle_y = current_position[1] + (bbox_center_y - 240) / 240 * distance

            grid_x = int(obstacle_x * 2)
            grid_y = int(obstacle_y * 2)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                # 为汽车创建更大的障碍物区域
                radius = 4
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        if (0 <= grid_x+dx < self.grid_size and 
                            0 <= grid_y+dy < self.grid_size):
                            self.obstacle_grid[grid_x+dx, grid_y+dy] = 1

    def astar_search(self, start, goal):
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        open_set = {start}
        closed_set = set()
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        came_from = {}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

            if current == goal:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if (neighbor in closed_set or 
                    not self.is_valid_position(neighbor)):
                    continue

                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)

        return []  # 未找到路径

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []

        # 8方向移动以获得更平滑的路径
        for dx, dy in [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))

        return neighbors

    def is_valid_position(self, pos):
        x, y = pos
        return self.obstacle_grid[x, y] == 0

    def heuristic(self, a, b):
        # 欧几里得距离以获得更好的路径质量
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def distance(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

class CarController:
    def __init__(self):
        # 优化的汽车控制PID参数
        self.steering_pid = PID(0.8, 0.1, 0.05, setpoint=0)
        self.speed_pid = PID(1.0, 0.1, 0.03, setpoint=1.0)
        
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.lookahead_distance = 4.0
        self.max_speed = 3.0
        self.steering_limit = 0.3
        self.steering_rate_limit = 0.05
        self.last_steering = 0.0

        # 防止积分饱和
        self.steering_pid.output_limits = (-self.steering_limit, self.steering_limit)
        self.speed_pid.output_limits = (0, 1.0)

    def update_control(self, current_position, current_heading, path):
        if len(path) < 2:
            return 0, 0

        # 改进的目标点选择策略
        target_point = self.find_improved_target_point(current_position, path)
        if target_point is None:
            return 0, 0

        # 计算航向误差
        target_heading = math.atan2(
            target_point[1] - current_position[1],
            target_point[0] - current_position[0]
        )
        heading_error = self.normalize_angle(target_heading - current_heading)

        # 改进的转向控制，包含前馈补偿
        steering_ff = self.calculate_feedforward(path, current_position)
        steering_fb = self.steering_pid(heading_error)
        steering_angle = steering_ff + steering_fb

        # 应用转向速率限制
        steering_change = steering_angle - self.last_steering
        if abs(steering_change) > self.steering_rate_limit:
            steering_angle = self.last_steering + np.sign(steering_change) * self.steering_rate_limit

        steering_angle = np.clip(steering_angle, -self.steering_limit, self.steering_limit)
        self.last_steering = steering_angle

        # 基于曲率和跟踪误差的自适应速度控制
        throttle = self.adaptive_speed_control(heading_error, current_position, target_point)

        # 更新内部状态
        self.current_speed = throttle * self.max_speed
        self.current_steering = steering_angle

        return throttle, steering_angle

    def find_improved_target_point(self, current_pos, path):
        if len(path) < 2:
            return path[-1] if path else None

        # 基于速度的动态前瞻距离调整
        speed_factor = max(0.5, min(2.0, self.current_speed))
        dynamic_lookahead = self.lookahead_distance * speed_factor

        # 考虑路径曲率找到最佳目标点
        for i in range(1, len(path)):
            point = path[i]
            distance = self.distance(current_pos, point)

            # 计算路径曲率
            if i > 1:
                prev_point = path[i-1]
                curvature = self.calculate_curvature(prev_point, point)
                # 基于曲率调整前瞻
                curvature_factor = max(0.3, 1.0 - abs(curvature))
                adjusted_lookahead = dynamic_lookahead * curvature_factor
            else:
                adjusted_lookahead = dynamic_lookahead

            if distance >= adjusted_lookahead:
                return point

        return path[-1]  # 默认返回终点

    def calculate_feedforward(self, path, current_pos):
        """前馈控制计算路径曲率补偿"""
        if len(path) < 3:
            return 0

        # 找到最近路径点附近的点计算曲率
        closest_idx = 0
        min_dist = float('inf')
        
        for i, point in enumerate(path):
            dist = self.distance(current_pos, point)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # 使用附近点计算曲率
        idx = min(closest_idx + 2, len(path) - 1)
        if idx >= 2:
            p0, p1, p2 = path[idx-2], path[idx-1], path[idx]
            
            # 计算曲率
            dx1, dy1 = p1[0]-p0[0], p1[1]-p0[1]
            dx2, dy2 = p2[0]-p1[0], p2[1]-p1[1]
            
            curvature = (dx1*dy2 - dy1*dx2) / max(0.1, (dx1**2 + dy1**2)**1.5)
            
            # 前馈转向补偿
            ff_gain = 0.3
            return ff_gain * curvature
        
        return 0

    def adaptive_speed_control(self, heading_error, current_pos, target_point):
        """自适应速度控制"""
        # 基于转向误差和距离的速度调整
        error_penalty = min(1.0, abs(heading_error) / (math.pi/4))
        distance = self.distance(current_pos, target_point)
        
        # 基础速度
        base_speed = 1.5
        
        # 误差越大，速度越慢
        speed_reduction = error_penalty * 0.8
        
        # 距离越近，速度越慢
        if distance < 3.0:
            distance_reduction = (3.0 - distance) / 3.0 * 0.5
            speed_reduction += distance_reduction
            
        target_speed = base_speed * (1.0 - speed_reduction)
        target_speed = max(0.5, min(self.max_speed, target_speed))
        
        self.speed_pid.setpoint = target_speed
        throttle = self.speed_pid(self.current_speed)
        
        return np.clip(throttle, 0, 1.0)

    def calculate_curvature(self, point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return dy / (dx + 1e-5)

    def distance(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def reset(self):
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.last_steering = 0.0
        self.steering_pid.reset()
        self.speed_pid.reset()
        print("汽车控制器重置到初始状态")

class CarSimulator:
    def __init__(self, init_position=[0, 0], init_heading=0):
        self.position = np.array(init_position, dtype=float)
        self.heading = init_heading
        self.speed = 0.0
        self.steering_angle = 0.0
        self.wheelbase = 2.5
        self.dt = 0.1
        self.max_acceleration = 1.5
        self.max_deceleration = 2.0

    def update(self, throttle, steering_angle):
        # 限制输入范围
        throttle = np.clip(throttle, 0, 1.0)
        steering_angle = np.clip(steering_angle, -0.3, 0.3)

        # 改进的加速度模型
        target_speed = throttle * self.max_acceleration * 2.0
        acceleration = (target_speed - self.speed) * 2.0

        # 考虑物理限制
        if acceleration > 0:
            acceleration = min(acceleration, self.max_acceleration)
        else:
            acceleration = max(acceleration, -self.max_deceleration)

        self.speed += acceleration * self.dt
        self.speed = max(0, self.speed)

        # 更新转向角
        self.steering_angle = steering_angle

        # 改进的汽车运动学模型
        if abs(self.steering_angle) < 1e-5:
            # 直线运动
            self.position[0] += self.speed * math.cos(self.heading) * self.dt
            self.position[1] += self.speed * math.sin(self.heading) * self.dt
        else:
            # 转向运动 - 改进的汽车模型
            turning_radius = self.wheelbase / math.tan(self.steering_angle)
            
            # 限制最小转弯半径
            min_turning_radius = self.wheelbase / math.tan(0.3)
            turning_radius = max(min_turning_radius, abs(turning_radius)) * np.sign(turning_radius)

            angular_velocity = self.speed / turning_radius

            # 更新航向
            self.heading += angular_velocity * self.dt

            # 使用圆周运动更新位置
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
        print(f"汽车模拟器重置到位置 {init_position}, 航向 {init_heading}")

class EmergencyMonitor:
    def __init__(self):
        self.error_count = 0
        self.max_errors = 10
        self.last_position = None
        
    def check_control_health(self, error, position, heading):
        """检查控制健康状态"""
        if error > 5.0:
            self.error_count += 1
            print(f"警告: 控制误差过大 {error:.2f}m")
            
        # 检查是否卡住
        if self.last_position is not None:
            movement = self.distance(position, self.last_position)
            if movement < 0.1 and error > 2.0:
                self.error_count += 1
                print("警告: 车辆可能卡住")
                
        self.last_position = position.copy()
            
        if self.error_count >= self.max_errors:
            print("紧急: 控制系统异常，执行安全停止")
            return False
            
        return True
    
    def safe_stop(self, controller, simulator):
        """安全停止程序"""
        print("执行安全停止程序...")
        # 渐进减速
        for i in range(5):
            throttle = max(0, 0.5 - i * 0.1)
            simulator.update(throttle, 0)
            time.sleep(0.1)
        
        controller.reset()
        print("安全停止完成")
        
    def distance(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ==================== 工具函数 ====================

def process_path_for_control(original_path, max_points=20):
    """为控制模块优化路径点"""
    if len(original_path) <= max_points:
        return original_path
        
    # 均匀采样
    step = len(original_path) // max_points
    processed_path = original_path[::step]
    
    # 确保包含起点和终点
    if processed_path[0] != original_path[0]:
        processed_path.insert(0, original_path[0])
    if processed_path[-1] != original_path[-1]:
        processed_path.append(original_path[-1])
        
    return processed_path[:max_points]

def generate_test_image(width=640, height=480):
    """生成汽车场景测试图像"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 添加模拟道路 - 更暗的颜色
    road_color = (180, 180, 180)
    cv2.rectangle(image, (0, height//2-80), (width, height//2+80), road_color, -1)

    # 添加道路中心线 - 更粗更明显
    center_line_color = (0, 255, 255)
    cv2.line(image, (width//2, 0), (width//2, height), center_line_color, 5)

    # 添加清晰可见的模拟障碍物
    obstacles = [
        {'pos': (150, height//2), 'size': (120, 80), 'color': (0, 0, 255), 'label': 'Car'},
        {'pos': (400, height//2-40), 'size': (60, 100), 'color': (255, 0, 0), 'label': 'Person'},
        {'pos': (500, height//2+30), 'size': (100, 60), 'color': (0, 100, 0), 'label': 'Vehicle'},
        {'pos': (300, height//2+50), 'size': (80, 80), 'color': (128, 0, 128), 'label': 'Obstacle'},
    ]

    for obstacle in obstacles:
        x, y = obstacle['pos']
        w, h = obstacle['size']
        color = obstacle['color']

        # 绘制障碍物 - 带边框和填充
        cv2.rectangle(image, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
        cv2.rectangle(image, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 0, 0), 3)

        # 添加标签
        label = obstacle['label']
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # 标签背景
        cv2.rectangle(image,
                    (x-w//2, y-h//2 - text_height - 10),
                    (x-w//2 + text_width, y-h//2),
                    color, -1)
        cv2.rectangle(image,
                    (x-w//2, y-h//2 - text_height - 10),
                    (x-w//2 + text_width, y-h//2),
                    (0, 0, 0), 1)

        # 标签文本
        cv2.putText(image, label,
                   (x-w//2, y-h//2 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # 添加一些噪声和纹理以获得更真实的图像
    noise = np.random.randint(0, 10, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)

    return image

def visualize_detections(image, detections, inference_time):
    """可视化检测结果"""
    result_image = image.copy()

    # 定义不同类别的颜色
    colors = {
        'person': (255, 0, 0),
        'car': (0, 0, 255),
        'vehicle': (0, 255, 0),
        'obstacle': (128, 0, 128)
    }

    # 绘制检测框
    for i, det in enumerate(detections):
        bbox = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        distance = det['distance']

        color = colors.get(class_name, (128, 0, 128))

        # 绘制边界框 - 更粗的线条
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

        # 绘制标签
        label = f"{class_name} {confidence:.2f} {distance:.1f}m"
        font_scale = 0.6
        thickness = 2
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        # 标签背景矩形
        cv2.rectangle(result_image,
                    (bbox[0], bbox[1] - label_size[1] - 10),
                    (bbox[0] + label_size[0], bbox[1]),
                    color, -1)
        cv2.rectangle(result_image,
                    (bbox[0], bbox[1] - label_size[1] - 10),
                    (bbox[0] + label_size[0], bbox[1]),
                    (0, 0, 0), 1)

        # 绘制标签文本
        cv2.putText(result_image, label,
                   (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # 在框内显示索引号
        index_text = str(i+1)
        index_size = cv2.getTextSize(index_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(result_image, index_text,
                   (bbox[0] + 5, bbox[1] + index_size[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 添加性能信息
    info_text = f"Objects detected: {len(detections)}, Time: {inference_time:.1f}ms"
    cv2.putText(result_image, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 如果没有检测到物体显示警告
    if len(detections) == 0:
        warning_text = "No obstacles detected"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = image.shape[0] - 30
        cv2.putText(result_image, warning_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return result_image

# ==================== 测试函数 ====================

def test_perception():
    """完整的汽车感知模块测试"""
    print("=" * 50)
    print("汽车感知模块测试")
    print("=" * 50)
    
    detector = ObstacleDetector(conf_threshold=0.3)

    # 生成测试图像
    print("生成汽车场景合成测试图像...")
    image = generate_test_image()

    # 运行检测
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = (time.time() - start_time) * 1000

    # 显示结果
    print(f"检测结果 (时间: {inference_time:.1f}ms):")
    print("-" * 40)
    
    for i, det in enumerate(detections):
        print(f"目标 {i+1}:")
        print(f" 类别: {det['class']}")
        print(f" 置信度: {det['confidence']:.3f}")
        print(f" 边界框: {det['bbox']}")
        print(f" 距离: {det['distance']:.2f}m")
        print()

    # 可视化结果
    result_image = visualize_detections(image, detections, inference_time)

    # 保存结果
    cv2.imwrite('car_perception_result.jpg', result_image)
    print("结果保存到: car_perception_result.jpg")

    # 性能统计
    if len(detections) > 0:
        avg_confidence = np.mean([det['confidence'] for det in detections])
        avg_distance = np.mean([det['distance'] for det in detections])
        print(f"性能统计:")
        print(f" 平均置信度: {avg_confidence:.3f}")
        print(f" 平均距离: {avg_distance:.2f}m")
        print(f" 检测到物体: {len(detections)}")
    else:
        print("警告: 未检测到障碍物!")
        print("建议: 尝试进一步降低conf_threshold参数")

    return detections

def test_planning():
    """完整的汽车路径规划模块测试"""
    print("\n" + "=" * 50)
    print("汽车路径规划模块测试")
    print("=" * 50)
    
    planner = PathPlanner(grid_size=50)

    # 汽车测试场景配置
    test_scenarios = [
        {
            'name': '简单场景 - 无障碍物',
            'start': (5, 5),
            'goal': (45, 45),
            'obstacles': []
        },
        {
            'name': '复杂场景 - 多障碍物',
            'start': (5, 5),
            'goal': (45, 45),
            'obstacles': [
                {'distance': 2.0, 'bbox': [200, 150, 250, 200]},
                {'distance': 3.5, 'bbox': [300, 250, 350, 300]},
                {'distance': 4.0, 'bbox': [150, 300, 200, 350]},
                {'distance': 2.5, 'bbox': [400, 200, 450, 250]},
            ]
        },
        {
            'name': '挑战场景 - 狭窄通道',
            'start': (5, 25),
            'goal': (45, 25),
            'obstacles': [
                {'distance': 2.0, 'bbox': [200, 100, 250, 200]},
                {'distance': 2.0, 'bbox': [200, 300, 250, 400]},
            ]
        }
    ]

    all_results = []

    for scenario in test_scenarios:
        print(f"测试场景: {scenario['name']}")
        print(f"起点: {scenario['start']}, 终点: {scenario['goal']}")

        # 更新障碍物地图
        planner.update_obstacle_grid(scenario['obstacles'], [0, 0])

        # 运行路径规划
        start_time = time.time()
        path = planner.astar_search(scenario['start'], scenario['goal'])
        planning_time = (time.time() - start_time) * 1000

        # 分析结果
        success = len(path) > 0
        path_length = len(path) if success else 0
        
        print(f"规划结果: {'成功' if success else '失败'}")
        print(f"规划时间: {planning_time:.1f}ms")
        print(f"路径长度: {path_length} 点")

        if success:
            print(f"路径样例: 前3点 {path[:3]}... 后3点 {path[-3:]}")

            # 计算路径效率
            if len(path) > 1:
                start = path[0]
                goal = path[-1]
                straight_line_dist = math.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)
                path_efficiency = straight_line_dist / len(path) if len(path) > 0 else 0
                print(f"路径效率: {path_efficiency:.3f}")
        else:
            print("规划失败: 未找到有效路径")

        # 保存结果
        result = {
            'scenario': scenario['name'],
            'success': success,
            'planning_time': planning_time,
            'path_length': path_length,
            'path': path,
            'obstacles': scenario['obstacles']
        }
        all_results.append(result)

    # 生成综合报告
    successful_tests = [r for r in all_results if r['success']]
    failed_tests = [r for r in all_results if not r['success']]
    
    print(f"\n汽车规划测试报告:")
    print(f" 总场景数: {len(all_results)}")
    print(f" 成功场景: {len(successful_tests)}")
    print(f" 失败场景: {len(failed_tests)}")
    print(f" 成功率: {len(successful_tests)/len(all_results)*100:.1f}%")

    return all_results

def test_control():
    """完整的汽车控制模块测试"""
    print("\n" + "=" * 50)
    print("汽车控制模块测试")
    print("=" * 50)
    
    controller = CarController()
    simulator = CarSimulator(init_position=[0, 0], init_heading=0)
    emergency_monitor = EmergencyMonitor()

    # 增强的测试路径
    test_paths = {
        '直线': [(i, 0) for i in range(1, 11)],
        '缓弯': [(i, 0.1 * i) for i in range(1, 11)],
        '急弯': [(i, 2 * math.sin(0.3 * i)) for i in range(1, 11)],
    }

    all_control_results = []

    for path_name, path in test_paths.items():
        print(f"测试路径: {path_name}")
        print(f"路径点数: {len(path)}")

        # 为每个测试重置控制器和模拟器
        controller.reset()
        simulator.reset(init_position=[0, 0], init_heading=0)
        
        # 处理路径点
        processed_path = process_path_for_control(path, max_points=15)

        trajectory = []
        control_signals = []
        errors = []
        headings = []

        # 增强的控制循环
        max_steps = 50
        target_reached = False

        for step in range(max_steps):
            throttle, steering = controller.update_control(
                simulator.position, simulator.heading, processed_path
            )
            position, heading = simulator.update(throttle, steering)

            # 记录综合数据
            trajectory.append(position.copy())
            control_signals.append((throttle, steering))
            headings.append(heading)

            # 计算到最近路径点的跟踪误差
            if processed_path:
                distances = [np.linalg.norm(position - path_point) for path_point in processed_path]
                min_distance = min(distances)
                closest_point_idx = np.argmin(distances)
                errors.append(min_distance)

            # 健康检查
            if not emergency_monitor.check_control_health(min_distance, position, heading):
                emergency_monitor.safe_stop(controller, simulator)
                break

            # 检查是否到达路径终点
            if np.linalg.norm(position - processed_path[-1]) < 0.3:
                print(f" 到达目标点! 步数: {step + 1}")
                target_reached = True
                break

        # 性能分析
        if errors:
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            final_error = errors[-1] if errors else 0
            error_std = np.std(errors)
            
            print(f" 平均跟踪误差: {avg_error:.3f}m")
            print(f" 最大跟踪误差: {max_error:.3f}m")
            print(f" 最终跟踪误差: {final_error:.3f}m")
            print(f" 误差标准差: {error_std:.3f}m")
            print(f" 目标到达: {'是' if target_reached else '否'}")

        # 保存结果
        result = {
            'path_name': path_name,
            'trajectory': trajectory,
            'control_signals': control_signals,
            'errors': errors,
            'headings': headings,
            'performance': {
                'average_error': float(avg_error) if errors else 0.0,
                'max_error': float(max_error) if errors else 0.0,
                'final_error': float(final_error) if errors else 0.0,
                'error_std': float(error_std) if errors else 0.0,
                'steps': len(trajectory),
                'target_reached': target_reached,
            }
        }
        all_control_results.append(result)

    # 生成控制性能报告
    if all_control_results:
        avg_errors = [r['performance']['average_error'] for r in all_control_results]
        overall_avg_error = np.mean(avg_errors) if avg_errors else 0
        target_reached_count = sum(1 for r in all_control_results if r['performance']['target_reached'])
        
        print(f"\n控制测试完成!")
        print(f" 总体平均误差: {overall_avg_error:.3f}m")
        print(f" 成功率: {target_reached_count}/{len(all_control_results)} 路径完成")

    return all_control_results

def run_comprehensive_test():
    """运行完整的汽车系统综合测试"""
    print("开始汽车系统综合测试...")

    # 1. 测试感知模块
    print("\n>>> 测试汽车感知模块...")
    detections = test_perception()

    # 2. 测试路径规划模块
    print("\n>>> 测试汽车路径规划模块...")
    planning_results = test_planning()

    # 3. 测试控制模块
    print("\n>>> 测试汽车控制模块...")
    control_results = test_control()

    print("\n" + "=" * 50)
    print("汽车系统综合测试完成!")
    print("=" * 50)

    # 生成测试报告
    print("\n汽车系统性能总结:")
    print(f" 感知模块: {len(detections)} 个物体检测到")
    
    successful_plans = len([r for r in planning_results if r['success']])
    print(f" 规划模块: {successful_plans}/{len(planning_results)} 场景成功")

    if control_results and len(control_results) > 0:
        avg_errors = [r['performance']['average_error'] for r in control_results]
        overall_avg_error = np.mean(avg_errors) if avg_errors else 0
        target_reached_count = sum(1 for r in control_results if r['performance']['target_reached'])
        print(f" 控制模块: 平均误差 {overall_avg_error:.3f}m")
        print(f" 控制模块: {target_reached_count}/{len(control_results)} 路径完成")

    # 总体评估
    perception_score = min(10, len(detections) * 3)
    planning_score = min(10, (successful_plans / len(planning_results)) * 10) if planning_results else 0
    
    if control_results:
        if overall_avg_error < 0.5:
            control_score = 10
        elif overall_avg_error < 1.0:
            control_score = 8
        elif overall_avg_error < 2.0:
            control_score = 6
        else:
            control_score = 4
    else:
        control_score = 0

    overall_score = (perception_score + planning_score + control_score) / 3

    print(f"\n总体汽车系统评分: {overall_score:.1f}/10")

    if overall_score >= 8.0:
        print(" 状态: 优秀 - 汽车系统准备就绪")
    elif overall_score >= 6.0:
        print(" 状态: 良好 - 需要少量优化")
    else:
        print(" 状态: 需要改进 - 需要显著改进")

    return {
        'perception': detections,
        'planning': planning_results,
        'control': control_results,
        'scores': {
            'perception': perception_score,
            'planning': planning_score,
            'control': control_score,
            'overall': overall_score
        }
    }

def run_simple_demo():
    """运行简单的汽车避障演示"""
    print("开始汽车自动驾驶演示...")

    # 初始化汽车系统
    detector = ObstacleDetector(conf_threshold=0.3)
    planner = PathPlanner(grid_size=50)
    controller = CarController()
    simulator = CarSimulator(init_position=[2, 2])
    emergency_monitor = EmergencyMonitor()

    # 设置目标
    goal_position = [8, 8]
    
    print("汽车自动驾驶系统初始化完成")
    print(f"起始位置: {simulator.position}")
    print(f"目标位置: {goal_position}")
    print("开始汽车自动驾驶操作...")

    # 运行演示
    for step in range(50):
        if step % 10 == 0:
            print(f"\n--- 步骤 {step + 1} ---")

        # 1. 感知
        image = generate_test_image()
        detections = detector.detect(image)
        
        if step % 10 == 0:
            print(f"感知: {len(detections)} 个障碍物检测到")

        # 2. 规划
        planner.update_obstacle_grid(detections, simulator.position)
        path = planner.astar_search(
            (simulator.position[0]*3, simulator.position[1]*3),
            (goal_position[0]*3, goal_position[1]*3)
        )
        
        world_path = [(p[0]/3, p[1]/3) for p in path] if path else []
        world_path = process_path_for_control(world_path, max_points=15)
        
        if step % 10 == 0 and path:
            print(f"规划: {len(world_path)} 个路径点生成")

        # 3. 控制
        throttle, steering = controller.update_control(
            simulator.position, simulator.heading, world_path
        )

        # 4. 执行
        position, heading = simulator.update(throttle, steering)

        if step % 10 == 0:
            print(f"控制: 油门={throttle:.2f}, 转向={steering:.2f}")
            print(f"状态: 位置=({position[0]:.2f}, {position[1]:.2f})")

        # 健康检查
        tracking_error = np.linalg.norm(position - goal_position)
        if not emergency_monitor.check_control_health(tracking_error, position, heading):
            break

        # 检查是否到达目标
        if np.linalg.norm(position - goal_position) < 0.5:
            print("\n成功: 汽车到达目标位置!")
            break

    final_dist = np.linalg.norm(position - goal_position)
    print(f"\n汽车演示完成! 最终距离目标: {final_dist:.2f} 米")

    return {
        'final_position': position,
        'goal_position': goal_position,
        'final_error': final_dist
    }

def main():
    """汽车自动驾驶系统主菜单"""
    print("汽车自动驾驶系统 v2.0")
    print("城市道路汽车自动驾驶和路径规划系统")

    while True:
        print("\n" + "="*50)
        print("汽车自动驾驶系统")
        print("="*50)
        print("1. 运行简单汽车演示")
        print("2. 测试汽车感知模块")
        print("3. 测试汽车路径规划模块")
        print("4. 测试汽车控制模块")
        print("5. 运行汽车系统综合测试")
        print("6. 退出系统")

        choice = input("\n选择操作 (1-6): ").strip()

        if choice == '1':
            run_simple_demo()
        elif choice == '2':
            test_perception()
        elif choice == '3':
            test_planning()
        elif choice == '4':
            test_control()
        elif choice == '5':
            run_comprehensive_test()
        elif choice == '6':
            print("感谢使用汽车自动驾驶系统! 再见!")
            break
        else:
            print("无效选择，请重试!")

if __name__ == "__main__":
    main()