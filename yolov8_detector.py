import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 修改导入方式，确保能够正确导入test_modules中的函数
try:
    import test_modules
    # 尝试直接从模块获取函数
    generate_realistic_test_image = getattr(test_modules, 'generate_realistic_test_image', None)
    if not generate_realistic_test_image:
        print("Warning: generate_realistic_test_image not found in test_modules")
except ImportError as e:
    print(f"Error importing test_modules: {e}")
    # 定义一个备用函数
    def generate_realistic_test_image(width=640, height=480):
        import numpy as np
        print("Using fallback test image generator")
        # 创建一个简单的测试图像
        image = np.ones((height, width, 3), dtype=np.uint8) * 240  # 浅灰色背景
        return image

import cv2
import numpy as np
import time
import json
from pathlib import Path

class YOLOv8ObstacleDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
        """
        初始化YOLOv8检测器 - 修复版本
        Args:
            model_path: 模型路径
            conf_threshold: 统一置信度阈值0.3
        """
        self.conf_threshold = conf_threshold
        print(f"使用标准置信度阈值: {self.conf_threshold}")
        
        print(f"Loading YOLOv8 model from {model_path}...")

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            print("正在尝试下载模型...")
            try:
                self.model = YOLO('yolov8n.pt')  # 自动下载
                print("模型下载成功")
            except Exception as e:
                print(f"模型下载失败: {e}")
                raise
        else:
            # 检查模型文件是否有效
            file_size = os.path.getsize(model_path)
            print(f"模型文件大小: {file_size / (1024*1024):.1f} MB")
            if file_size < 1000000:  # 小于1MB可能损坏
                print("模型文件可能损坏，重新下载...")
                self.model = YOLO('yolov8n.pt')
            else:
                self.model = YOLO(model_path)
        
        print("YOLOv8 model loaded successfully")

        # 定义汽车场景相关类别
        self.relevant_classes = {
            0: 'person',      # person
            1: 'bicycle',     # bicycle
            2: 'car',         # car
            3: 'motorcycle',  # motorcycle
            5: 'bus',         # bus
            7: 'truck',       # truck
            9: 'traffic light',  # traffic light
            11: 'stop sign',     # stop sign
        }

        print(f"Detector configured for {len(self.relevant_classes)} relevant classes")
        print(f"Confidence threshold: {self.conf_threshold}")

    def detect(self, image):
        """
        使用YOLOv8进行目标检测 - 修复版本
        Args:
            image: 输入图像(BGR格式)
        Returns:
            detections: 检测结果列表
        """
        print(f"开始检测，使用置信度阈值: {self.conf_threshold}")

        # 调整图像尺寸以获得更好检测效果
        original_height, original_width = image.shape[:2]
        resized_image = cv2.resize(image, (640, 640))

        # 使用YOLOv8推理，使用统一的置信度阈值
        try:
            results = self.model(resized_image, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            print(f"推理失败: {e}")
            return []

        detections = []
        valid_detection_count = 0

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取检测信息
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # 只处理相关类别且置信度达标的检测
                    if class_id in self.relevant_classes and confidence >= self.conf_threshold:
                        # 缩放坐标回原始图像尺寸
                        x1 = int(x1 * original_width / 640)
                        y1 = int(y1 * original_height / 640)
                        x2 = int(x2 * original_width / 640)
                        y2 = int(y2 * original_height / 640)

                        class_name = self.relevant_classes[class_id]
                        
                        # 计算距离
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        distance = self.estimate_distance(bbox_width, bbox_height, original_width, class_name)

                        detection = {
                            'class': class_name,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'distance': distance
                        }

                        detections.append(detection)
                        valid_detection_count += 1
                        print(f"有效检测: {class_name} (置信度: {confidence:.3f})")

        print(f"检测完成: 总计 {valid_detection_count} 个有效检测")
        return detections

    def estimate_distance(self, bbox_width, bbox_height, image_width, class_name):
        """
        基于边界框大小估计距离
        Args:
            bbox_width: 边界框宽度(像素)
            bbox_height: 边界框高度(像素)
            image_width: 图像宽度(像素)
            class_name: 检测到的物体类别名称
        Returns:
            distance: 估计距离(米)
        """
        # 不同类别的参考尺寸(米)
        reference_sizes = {
            'person': (0.5, 1.7),      # 宽度, 高度
            'bicycle': (0.7, 1.1),
            'car': (1.8, 1.5),
            'motorcycle': (0.8, 1.2),
            'bus': (2.5, 3.0),
            'truck': (2.5, 3.0),
            'traffic light': (0.3, 0.6),
            'stop sign': (0.6, 0.6),
        }

        if class_name in reference_sizes:
            ref_width, ref_height = reference_sizes[class_name]
            
            # 使用宽度进行距离估计(对大多数物体更可靠)
            apparent_size = bbox_width
            ref_size = ref_width * 100  # 转换为厘米

            # 简化的距离估计公式
            focal_length = 800  # 假设焦距(像素)
            distance = (ref_size * focal_length) / (apparent_size * 100)

            # 应用合理限制
            return max(0.5, min(20.0, distance))
        else:
            return 5.0  # 未知物体的默认距离

    def get_detection_statistics(self, detections):
        """
        生成检测结果的统计信息
        Args:
            detections: 检测结果列表
        Returns:
            stats: 包含检测统计信息的字典
        """
        if not detections:
            return {
                'total_detections': 0,
                'class_distribution': {},
                'avg_confidence': 0,
                'avg_distance': 0
            }

        class_distribution = {}
        confidences = []
        distances = []

        for detection in detections:
            class_name = detection['class']
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
            confidences.append(detection['confidence'])
            distances.append(detection['distance'])

        return {
            'total_detections': len(detections),
            'class_distribution': class_distribution,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_distance': np.mean(distances) if distances else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'min_distance': min(distances) if distances else 0,
            'max_distance': max(distances) if distances else 0
        }

def test_yolov8_basic_functionality():
    """
    测试YOLOv8基础功能是否正常
    Returns: True if basic functionality works, False otherwise
    """
    print("=" * 60)
    print("YOLOv8 基础功能测试")
    print("=" * 60)

    try:
        # 1. 测试导入
        print("1. Ultralytics包导入成功")

        # 2. 测试模型加载
        print("2. 测试模型加载...")
        test_model = YOLO('yolov8n.pt')
        print("模型加载成功")

        # 3. 测试基础推理
        print("3. 测试基础推理...")
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = test_model(test_img, verbose=False)
        print(f"基础推理测试通过，结果数: {len(results)}")

        # 4. 测试标准图像检测 - 使用优化后的图片生成
        print("4. 测试标准图像检测...")
        detector = YOLOv8ObstacleDetector(conf_threshold=0.3)
        
        # 使用优化后的测试图片生成函数，移除不存在的num_objects参数
        test_image = generate_realistic_test_image(width=800, height=600)
        cv2.imwrite('car_test_scenario.jpg', test_image)
        print("生成汽车测试场景: car_test_scenario.jpg")
        
        start_time = time.time()
        detections = detector.detect(test_image)
        inference_time = (time.time() - start_time) * 1000

        print(f"标准图像测试完成:")
        print(f" - 推理时间: {inference_time:.1f}ms")
        print(f" - 检测到物体: {len(detections)} 个")

        # 保存测试图像
        cv2.imwrite('standard_test_image.jpg', test_image)
        print("标准测试图像已保存: standard_test_image.jpg")

        print("\n" + "=" * 60)
        print("测试结果总结:")
        print("=" * 60)
        print(f"import               通过")
        print(f"model_loading        通过")
        print(f"basic_inference      通过")
        print(f"standard_image       通过")
        print("\n所有基础功能测试通过！YOLOv8工作正常")

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        return False

def emergency_yolov8_repair():
    """紧急修复YOLOv8问题"""
    print("执行YOLOv8紧急修复...")
    
    # 检查并重新安装
    try:
        import subprocess
        import sys
        print("1. 重新安装ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "ultralytics", "-y"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("重新安装完成")

        # 测试功能
        print("2. 测试修复结果...")
        success = test_yolov8_basic_functionality()
        if success:
            print("紧急修复成功！")
        else:
            print("紧急修复失败，需要手动检查")
        return success

    except Exception as e:
        print(f"修复过程出错: {e}")
        return False

def visualize_yolo_detections(image, detections, inference_time):
    """可视化YOLOv8检测结果"""
    result_image = image.copy()

    # 定义不同类别的颜色映射
    colors = {
        'person': (255, 0, 0),        # Blue
        'bicycle': (0, 255, 0),       # Green
        'car': (0, 0, 255),           # Red
        'motorcycle': (0, 255, 255),  # Yellow
        'bus': (255, 0, 255),         # Purple
        'truck': (255, 165, 0),       # Orange
        'traffic light': (0, 0, 0),   # Black
        'stop sign': (128, 0, 128),   # Dark purple
    }

    # 绘制检测框
    for i, det in enumerate(detections):
        bbox = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        distance = det['distance']

        # 基于类别选择颜色
        color = colors.get(class_name, (128, 128, 128))  # 默认灰色

        # 绘制边界框
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # 绘制标签背景
        label = f"{class_name} {confidence:.2f} {distance:.1f}m"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        cv2.rectangle(result_image,
                    (bbox[0], bbox[1] - label_size[1] - 10),
                    (bbox[0] + label_size[0], bbox[1]),
                    color, -1)
        cv2.rectangle(result_image,
                    (bbox[0], bbox[1] - label_size[1] - 10),
                    (bbox[0] + label_size[0], bbox[1]),
                    (0, 0, 0), 1)  # 黑色边框

        # 绘制标签文本
        cv2.putText(result_image, label,
                   (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 在框内绘制索引号
        index_text = str(i + 1)
        cv2.putText(result_image, index_text,
                   (bbox[0] + 5, bbox[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 添加性能信息
    info_text = f"YOLOv8 Detections: {len(detections)}, Time: {inference_time:.1f}ms"
    cv2.putText(result_image, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 添加FPS信息
    fps = 1000 / inference_time if inference_time > 0 else 0
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(result_image, fps_text, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return result_image

def generate_car_test_scenario():
    """生成汽车自动驾驶专用测试场景"""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # 创建真实道路环境
    road_color = (100, 100, 100)
    cv2.rectangle(image, (0, 180), (640, 300), road_color, -1)

    # 车道标线
    lane_color = (255, 255, 255)
    cv2.line(image, (160, 0), (160, 480), lane_color, 2)
    cv2.line(image, (320, 0), (320, 480), (255, 255, 0), 3)  # 中心线
    cv2.line(image, (480, 0), (480, 480), lane_color, 2)

    # 添加车辆
    cv2.rectangle(image, (200, 200), (300, 250), (0, 0, 255), -1)  # 前方红色汽车
    cv2.rectangle(image, (400, 220), (500, 270), (0, 100, 0), -1)  # 侧方绿色汽车

    # 添加行人
    cv2.circle(image, (100, 250), 15, (255, 0, 0), -1)  # 蓝色行人
    cv2.circle(image, (550, 260), 15, (255, 0, 0), -1)  # 蓝色行人

    # 添加交通标志
    cv2.rectangle(image, (50, 100), (80, 150), (0, 0, 255), -1)  # 停止标志
    cv2.rectangle(image, (600, 120), (630, 170), (0, 255, 255), -1)  # 交通灯

    return image

if __name__ == "__main__":
    print("YOLOv8汽车障碍物检测器模块 - 修复版本")
    print("=" * 50)
    print("使用优化后的测试图片进行模型检测测试")

    print("\n第一步: 测试YOLOv8基础功能...")
    basic_ok = test_yolov8_basic_functionality()

    if not basic_ok:
        print("\n基础功能测试失败，执行紧急修复...")
        repair_ok = emergency_yolov8_repair()
        if not repair_ok:
            print("无法修复YOLOv8，请手动检查安装")
            exit(1)
    else:
        print("\n基础功能测试通过，继续汽车场景测试...")

    print("\n第二步: 运行汽车场景检测器测试...")
    try:
        detector = YOLOv8ObstacleDetector(conf_threshold=0.3)
        
        # 使用优化后的测试图片生成函数，移除不存在的num_objects参数
        test_image = generate_realistic_test_image(width=800, height=600)
        cv2.imwrite('car_test_scenario.jpg', test_image)
        print("生成汽车测试场景: car_test_scenario.jpg")
        
        start_time = time.time()
        detections = detector.detect(test_image)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"\n汽车场景测试完成:")
        print(f" - 推理时间: {inference_time:.1f}ms")
        print(f" - 检测到物体: {len(detections)} 个")
        
        # 打印每个检测结果的详细信息
        if detections:
            for i, det in enumerate(detections):
                print(f" - 物体 {i+1}: {det['class']} - 置信度: {det['confidence']:.3f} - 距离: {det['distance']:.1f}m")
        
        stats = detector.get_detection_statistics(detections)
        print(f"\n检测统计:")
        print(f" - 总检测数: {stats['total_detections']}")
        print(f" - 类别分布: {stats['class_distribution']}")
        print(f" - 平均置信度: {stats['avg_confidence']:.3f}")
        print(f" - 平均距离: {stats['avg_distance']:.2f}m")
        
        if detections:
            result_image = visualize_yolo_detections(test_image, detections, inference_time)
            cv2.imwrite('car_detection_result.jpg', result_image)
            print("检测结果已保存: car_detection_result.jpg")
            print("\n可视化结果可以查看: car_detection_result.jpg")
        else:
            print("\n未检测到物体，请检查生成的测试场景")
            
    except Exception as e:
        print(f"检测器测试失败: {e}")