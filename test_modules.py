import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from car_system import ObstacleDetector, PathPlanner, CarController, CarSimulator

# Additional imports
try:
    from yolov8_detector import YOLOv8ObstacleDetector, visualize_yolo_detections
    YOLOv8_AVAILABLE = True
except ImportError:
    YOLOv8_AVAILABLE = False

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import json
from pathlib import Path
import math

def test_perception():
    """Complete perception module test (auto-selects detector)"""
    print("=" * 50)
    print("Car Perception Module Test")
    print("=" * 50)

    # Auto-select detector
    if YOLOv8_AVAILABLE:
        print("Using YOLOv8 Deep Learning Detector")
        detector = YOLOv8ObstacleDetector(conf_threshold=0.01)
        use_yolov8 = True
    else:
        print("YOLOv8 not available, using traditional image detector")
        detector = ObstacleDetector(conf_threshold=0.3)
        use_yolov8 = False

    # Test options
    test_modes = ['image', 'camera', 'synthetic']
    mode = test_modes[0]  # Can choose different test modes

    if mode == 'image':
        # Test from file
        image_path = 'test_image.jpg'
        if Path(image_path).exists():
            image = cv2.imread(image_path)
            print(f"Loading image from file: {image_path}")
        else:
            # Generate test image
            print("Generating synthetic test image...")
            image = generate_realistic_test_image()
            cv2.imwrite('test_image.jpg', image)
    elif mode == 'camera':
        # Test from camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera, using synthetic image instead")
            image = generate_realistic_test_image()
        else:
            ret, image = cap.read()
            if not ret:
                print("Camera read failed, using synthetic image instead")
                image = generate_realistic_test_image()
            cap.release()
    else:  # synthetic
        image = generate_realistic_test_image()

    # Run detection
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = (time.time() - start_time) * 1000

    # Display results
    print(f"Detection Results (Time: {inference_time:.1f}ms):")
    print("-" * 40)
    for i, det in enumerate(detections):
        print(f"Target {i+1}:")
        print(f"  Class: {det['class']}")
        print(f"  Confidence: {det['confidence']:.3f}")
        print(f"  Bounding Box: {det['bbox']}")
        print(f"  Distance: {det['distance']:.2f}m")
        print()

    # Visualize results
    if use_yolov8:
        result_image = visualize_yolo_detections(image, detections, inference_time)
        result_file = 'yolov8_perception_result.jpg'
    else:
        result_image = visualize_detections(image, detections, inference_time)
        result_file = 'perception_result.jpg'

    cv2.imwrite(result_file, result_image)
    print(f"Results saved to: {result_file}")

    # Performance statistics
    if len(detections) > 0:
        avg_confidence = np.mean([det['confidence'] for det in detections])
        avg_distance = np.mean([det['distance'] for det in detections])
        print(f"Performance Statistics:")
        print(f"  Average Confidence: {avg_confidence:.3f}")
        print(f"  Average Distance: {avg_distance:.2f}m")
        print(f"  Objects Detected: {len(detections)}")

        # Class distribution
        class_dist = {}
        for det in detections:
            class_name = det['class']
            class_dist[class_name] = class_dist.get(class_name, 0) + 1
        print(f"  Class Distribution: {class_dist}")
    else:
        print("Warning: No objects detected!")
        print("Suggestions: Try lowering confidence threshold or improving test image quality")

    return detections

def test_yolov8_perception():
    """Specialized YOLOv8 perception module test"""
    if not YOLOv8_AVAILABLE:
        print("YOLOv8 not available, please install ultralytics package first")
        return []

    print("=" * 50)
    print("YOLOv8 Car Perception Module Test")
    print("=" * 50)
    
    detector = YOLOv8ObstacleDetector(model_path='yolov8n.pt', conf_threshold=0.3)

    # Generate more realistic test image
    image = generate_realistic_test_image()

    # Run detection
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = (time.time() - start_time) * 1000

    # Display results
    print(f"YOLOv8 Detection Results (Time: {inference_time:.1f}ms):")
    print("-" * 40)
    for i, det in enumerate(detections):
        print(f"Target {i+1}:")
        print(f"  Class: {det['class']}")
        print(f"  Confidence: {det['confidence']:.3f}")
        print(f"  Bounding Box: {det['bbox']}")
        print(f"  Distance: {det['distance']:.2f}m")
        print()

    # Visualize results
    result_image = visualize_yolo_detections(image, detections, inference_time)
    cv2.imwrite('yolov8_detection_result.jpg', result_image)
    print("Results saved to: yolov8_detection_result.jpg")

    # Enhanced performance analysis
    if len(detections) > 0:
        class_distribution = {}
        for det in detections:
            class_name = det['class']
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1

        print(f"Detailed Analysis:")
        print(f"  Total Detections: {len(detections)}")
        print(f"  Class Distribution: {class_distribution}")
        print(f"  Processing Speed: {1000/inference_time:.1f} FPS")

        # Confidence analysis
        confidences = [det['confidence'] for det in detections]
        print(f"  Confidence Range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"  Average Confidence: {np.mean(confidences):.3f}")
    else:
        print("Warning: YOLOv8 detected no objects")
        print("Possible issues:")
        print("  - Test image may not contain recognizable objects")
        print("  - Confidence threshold may be too high")
        print("  - Model may need retraining for your specific use case")

    return detections

def generate_realistic_test_image(width=640, height=480):
    """Generate more realistic test image with objects YOLOv8 can recognize"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Add realistic sky background
    sky_gradient = np.zeros((height//2, width, 3), dtype=np.uint8)
    for y in range(height//2):
        ratio = y / (height//2)
        sky_color = (int(255*ratio), int(230*ratio + 25), int(200*ratio))
        sky_gradient[y, :] = sky_color
    image[:height//2, :] = sky_gradient

    # Add realistic road background with better texture
    road_color = (60, 60, 60)  # Dark gray road
    cv2.rectangle(image, (0, height//2-100), (width, height), road_color, -1)
    
    # Add road texture (subtle noise) to make it more realistic
    noise = np.random.randint(0, 10, (height//2+100, width, 3), dtype=np.uint8)
    road_region = image[height//2-100:, :, :]
    road_region = cv2.add(road_region, noise)
    image[height//2-100:, :, :] = road_region

    # Add realistic lane markings with proper proportions
    lane_color = (255, 255, 255)  # White lanes
    cv2.line(image, (width//4, 0), (width//4, height), lane_color, 4)
    cv2.line(image, (width//2, 0), (width//2, height), (255, 255, 0), 6)  # Yellow center
    cv2.line(image, (3*width//4, 0), (3*width//4, height), lane_color, 4)
    
    # Add dashed lane line for better realism
    for i in range(0, height, 40):
        cv2.line(image, (width//3, i), (width//3, i+20), (255, 255, 255), 4)
    
    # Add more realistic car with better proportions
    car_points = np.array([
        [100, height//2-50],   # Top left
        [220, height//2-50],   # Top right
        [240, height//2+30],   # Bottom right
        [80, height//2+30]     # Bottom left
    ], np.int32)
    # Use realistic car color
    car_color = (0, 0, 200)  # Blue car
    cv2.fillPoly(image, [car_points], car_color)
    cv2.polylines(image, [car_points], True, (0, 0, 0), 2)  # Black outline
    
    # Add more detailed windows
    window_points = np.array([
        [120, height//2-50],
        [200, height//2-50],
        [210, height//2],
        [110, height//2]
    ], np.int32)
    cv2.fillPoly(image, [window_points], (230, 230, 255))  # Light blue window
    
    # Add wheels (more realistic circles)
    cv2.circle(image, (110, height//2+30), 15, (0, 0, 0), -1)
    cv2.circle(image, (110, height//2+30), 8, (100, 100, 100), -1)
    cv2.circle(image, (210, height//2+30), 15, (0, 0, 0), -1)
    cv2.circle(image, (210, height//2+30), 8, (100, 100, 100), -1)
    
    # Add a person with better proportions
    person_center = (width//2, height//2+20)
    # Head
    cv2.circle(image, (person_center[0], person_center[1]-40), 15, (255, 200, 150), -1)
    # Body
    cv2.rectangle(image, 
                 (person_center[0]-15, person_center[1]-25),
                 (person_center[0]+15, person_center[1]+15),
                 (0, 0, 255), -1)  # Red shirt
    # Arms
    cv2.line(image, (person_center[0]-15, person_center[1]-15), 
             (person_center[0]-40, person_center[1]), (0, 0, 255), 5)
    cv2.line(image, (person_center[0]+15, person_center[1]-15), 
             (person_center[0]+40, person_center[1]), (0, 0, 255), 5)
    # Legs
    cv2.line(image, (person_center[0]-8, person_center[1]+15), 
             (person_center[0]-15, person_center[1]+50), (0, 0, 100), 5)
    cv2.line(image, (person_center[0]+8, person_center[1]+15), 
             (person_center[0]+15, person_center[1]+50), (0, 0, 100), 5)
    
    # Add a bicycle with better proportions
    # Frame
    cv2.line(image, (width-150, height//2+30), (width-100, height//2+20), (0, 100, 0), 4)
    cv2.line(image, (width-125, height//2+30), (width-100, height//2+20), (0, 100, 0), 4)
    # Wheels (larger and more realistic)
    cv2.circle(image, (width-150, height//2+30), 20, (50, 50, 50), 3)
    cv2.circle(image, (width-100, height//2+20), 20, (50, 50, 50), 3)
    # Handlebar
    cv2.line(image, (width-125, height//2-5), (width-100, height//2-10), (0, 100, 0), 3)
    # Seat
    cv2.rectangle(image, (width-135, height//2-5), (width-115, height//2+5), (100, 0, 100), -1)
    
    # Add a stop sign with proper proportions
    # Octagon shape for stop sign (more realistic than rectangle)
    stop_sign_points = np.array([
        [width-150, height//2-80],
        [width-125, height//2-100],
        [width-100, height//2-80],
        [width-100, height//2-50],
        [width-125, height//2-30],
        [width-150, height//2-50],
    ], np.int32)
    cv2.fillPoly(image, [stop_sign_points], (0, 0, 255))  # Red stop sign
    cv2.polylines(image, [stop_sign_points], True, (0, 0, 0), 2)  # Black outline
    # Add white border
    cv2.polylines(image, [stop_sign_points], True, (255, 255, 255), 1)
    
    # Add another stop sign for better detection
    cv2.rectangle(image, (50, 100), (80, 150), (0, 0, 255), -1)
    cv2.rectangle(image, (48, 98), (82, 152), (255, 255, 255), 2)
    
    # Add traffic lights with better proportions
    # Traffic light 1
    cv2.rectangle(image, (width//5, 50), (width//5+30, 120), (50, 50, 50), -1)  # Pole
    cv2.rectangle(image, (width//5+5, 50), (width//5+25, 90), (0, 0, 0), -1)    # Lights housing
    cv2.circle(image, (width//5+15, 60), 8, (0, 0, 255), -1)  # Red light
    cv2.circle(image, (width//5+15, 75), 8, (0, 150, 0), -1) # Green light
    
    # Traffic light 2
    cv2.rectangle(image, (width-100, 80), (width-70, 150), (50, 50, 50), -1)
    cv2.rectangle(image, (width-95, 80), (width-75, 120), (0, 0, 0), -1)
    cv2.circle(image, (width-85, 90), 8, (0, 0, 255), -1)
    
    # Add a truck (new object type)
    truck_points = np.array([
        [300, height//2-40],
        [400, height//2-40],
        [410, height//2+30],
        [290, height//2+30]
    ], np.int32)
    cv2.fillPoly(image, [truck_points], (100, 100, 100))  # Gray truck
    cv2.polylines(image, [truck_points], True, (0, 0, 0), 2)
    # Truck wheels
    cv2.circle(image, (320, height//2+30), 15, (0, 0, 0), -1)
    cv2.circle(image, (380, height//2+30), 15, (0, 0, 0), -1)
    
    # Add a bus (new object type)
    bus_points = np.array([
        [150, height-150],
        [250, height-150],
        [260, height-80],
        [140, height-80]
    ], np.int32)
    cv2.fillPoly(image, [bus_points], (0, 100, 100))  # Teal bus
    cv2.polylines(image, [bus_points], True, (0, 0, 0), 2)
    # Bus wheels
    cv2.circle(image, (170, height-80), 15, (0, 0, 0), -1)
    cv2.circle(image, (230, height-80), 15, (0, 0, 0), -1)
    # Bus windows
    cv2.rectangle(image, (160, height-140), (180, height-120), (230, 230, 255), -1)
    cv2.rectangle(image, (190, height-140), (210, height-120), (230, 230, 255), -1)
    cv2.rectangle(image, (220, height-140), (240, height-120), (230, 230, 255), -1)
    
    # Apply global contrast enhancement to improve detection
    image = cv2.convertScaleAbs(image, alpha=1.1, beta=0)  # Increase contrast
    
    return image

def visualize_detections(image, detections, inference_time):
    """Visualize detection results"""
    result_image = image.copy()

    # Draw detection boxes
    for det in detections:
        bbox = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        distance = det['distance']

        # Draw bounding box
        color = (0, 255, 0) if det['class'] == 'person' else (255, 0, 0)
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Draw label
        label = f"{class_name} {confidence:.2f} {distance:.1f}m"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_image,
                     (bbox[0], bbox[1] - label_size[1] - 10),
                     (bbox[0] + label_size[0], bbox[1]),
                     color, -1)
        cv2.putText(result_image, label,
                   (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add performance information
    info_text = f"Objects detected: {len(detections)}, Time: {inference_time:.1f}ms"
    cv2.putText(result_image, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return result_image

def test_planning():
    """Complete path planning module test"""
    print("\n" + "=" * 50)
    print("Car Path Planning Module Test")
    print("=" * 50)
    
    planner = PathPlanner(grid_size=50)

    # Enhanced test scenario configuration
    test_scenarios = [
        {
            'name': 'Simple Scenario - No Obstacles',
            'start': (5, 5),
            'goal': (45, 45),
            'obstacles': []
        },
        {
            'name': 'Complex Scenario - Multiple Obstacles',
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
            'name': 'Challenging Scenario - Narrow Corridor',
            'start': (5, 25),
            'goal': (45, 25),
            'obstacles': [
                {'distance': 2.0, 'bbox': [200, 100, 250, 200]},
                {'distance': 2.0, 'bbox': [200, 300, 250, 400]},
                {'distance': 3.0, 'bbox': [300, 100, 350, 200]},
                {'distance': 3.0, 'bbox': [300, 300, 350, 400]},
            ]
        },
        {
            'name': 'Realistic Scenario - Urban Environment',
            'start': (5, 5),
            'goal': (45, 45),
            'obstacles': [
                {'distance': 1.5, 'bbox': [150, 100, 200, 150]},
                {'distance': 2.0, 'bbox': [300, 150, 350, 200]},
                {'distance': 3.5, 'bbox': [200, 300, 250, 350]},
                {'distance': 4.0, 'bbox': [350, 250, 400, 300]},
                {'distance': 2.5, 'bbox': [100, 200, 150, 250]},
            ]
        }
    ]

    all_results = []

    for scenario in test_scenarios:
        print(f"Test Scenario: {scenario['name']}")
        print(f"Start: {scenario['start']}, Goal: {scenario['goal']}")

        # Update obstacle map
        planner.update_obstacle_grid(scenario['obstacles'], [0, 0])

        # Run path planning
        start_time = time.time()
        path = planner.astar_search(scenario['start'], scenario['goal'])
        planning_time = (time.time() - start_time) * 1000

        # Analyze results
        success = len(path) > 0
        path_length = len(path) if success else 0
        print(f"Planning Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"Planning Time: {planning_time:.1f}ms")
        print(f"Path Length: {path_length} points")

        if success:
            print(f"Path Sample: First 3 points {path[:3]}... Last 3 points {path[-3:]}")

            # Calculate path efficiency
            if len(path) > 1:
                start = path[0]
                goal = path[-1]
                straight_line_dist = math.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)
                path_efficiency = straight_line_dist / len(path) if len(path) > 0 else 0
                print(f"Path Efficiency: {path_efficiency:.3f}")
        else:
            print("Planning failed: No valid path found")

        # Save results
        result = {
            'scenario': scenario['name'],
            'success': success,
            'planning_time': planning_time,
            'path_length': path_length,
            'path': path,
            'obstacles': scenario['obstacles']
        }
        all_results.append(result)

        # Visualize current scenario
        visualize_planning_scenario(planner, path, scenario, planning_time)

    # Generate comprehensive report
    generate_planning_report(all_results)

    return all_results

def visualize_planning_scenario(planner, path, scenario, planning_time):
    """Visualize planning scenario and results"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid map
    grid = planner.obstacle_grid.T  # Transpose for correct display
    ax.imshow(grid, cmap='RdYlGn_r', origin='lower',
              extent=[0, planner.grid_size, 0, planner.grid_size])

    # Draw start and goal
    start = scenario['start']
    goal = scenario['goal']
    ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal')

    # Draw path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Planned Path')
        ax.plot(path_array[:, 0], path_array[:, 1], 'bo', markersize=3)

    # Set graph properties
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f"{scenario['name']}\nPlanning Time: {planning_time:.1f}ms")
    ax.legend()
    ax.grid(True)

    # Save image
    filename = f"planning_{scenario['name'].replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Scenario image saved: {filename}")
    plt.close()

def generate_planning_report(results):
    """Generate path planning test report"""
    print("\n" + "=" * 50)
    print("Car Path Planning Test Report")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]

    print(f"Total Scenarios: {len(results)}")
    print(f"Successful Scenarios: {len(successful_tests)}")
    print(f"Failed Scenarios: {len(failed_tests)}")
    print(f"Success Rate: {len(successful_tests)/len(results)*100:.1f}%")

    if successful_tests:
        avg_planning_time = np.mean([r['planning_time'] for r in successful_tests])
        avg_path_length = np.mean([r['path_length'] for r in successful_tests])
        print(f"Average Planning Time: {avg_planning_time:.1f}ms")
        print(f"Average Path Length: {avg_path_length:.1f} points")

        # Path quality analysis
        path_efficiencies = []
        for result in successful_tests:
            if len(result['path']) > 1:
                start = result['path'][0]
                goal = result['path'][-1]
                straight_line_dist = math.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)
                actual_path_length = len(result['path'])
                efficiency = straight_line_dist / actual_path_length if actual_path_length > 0 else 0
                path_efficiencies.append(efficiency)

        if path_efficiencies:
            avg_efficiency = np.mean(path_efficiencies)
            efficiency_std = np.std(path_efficiencies)
            print(f"Average Path Efficiency: {avg_efficiency:.3f}")
            print(f"Path Efficiency Std: {efficiency_std:.3f}")

            # Path quality assessment
            if avg_efficiency > 0.8:
                quality = "EXCELLENT"
            elif avg_efficiency > 0.6:
                quality = "GOOD"
            elif avg_efficiency > 0.4:
                quality = "FAIR"
            else:
                quality = "POOR"
            print(f"Path Quality: {quality}")

        # Performance assessment
        success_rate = len(successful_tests)/len(results)*100
        if success_rate >= 90:
            performance = "EXCELLENT"
        elif success_rate >= 75:
            performance = "GOOD"
        elif success_rate >= 60:
            performance = "FAIR"
        else:
            performance = "POOR"
        print(f"Overall Performance: {performance}")

    # Save detailed results to JSON
    report_data = {
        'summary': {
            'total_tests': len(results),
            'successful_tests': len(successful_tests),
            'success_rate': float(success_rate),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'detailed_results': results
    }

    with open('planning_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"Detailed report saved: planning_test_report.json")

def test_control():
    """Enhanced car control module test with improved analysis"""
    print("\n" + "=" * 50)
    print("Car Control Module Test")
    print("=" * 50)
    
    controller = CarController()
    simulator = CarSimulator(init_position=[0, 0], init_heading=0)

    # Enhanced test paths with varied challenges for car
    test_paths = {
        'Straight Line': [(i, 0) for i in range(1, 11)],
        'Gentle Curve': [(i, 0.1 * i) for i in range(1, 11)],
        'Sharp Turn': [(i, 2 * math.sin(0.3 * i)) for i in range(1, 11)],
        'S-shaped Path': [(i, 3 * math.sin(0.4 * i)) for i in range(1, 11)],
        'Circular Path': [(5 + 3 * math.cos(0.5 * i), 3 * math.sin(0.5 * i)) for i in range(12)]
    }

    all_control_results = []

    for path_name, path in test_paths.items():
        print(f"Test Path: {path_name}")
        print(f"Path Points: {len(path)}")

        # Reset controller and simulator for each test
        controller.reset()
        simulator.reset(init_position=[0, 0], init_heading=0)

        trajectory = []
        control_signals = []
        errors = []
        headings = []

        # Enhanced control loop with more steps
        max_steps = 80
        target_reached = False

        for step in range(max_steps):
            throttle, steering = controller.update_control(
                simulator.position, simulator.heading, path
            )
            position, heading = simulator.update(throttle, steering)

            # Record comprehensive data
            trajectory.append(position.copy())
            control_signals.append((throttle, steering))
            headings.append(heading)

            # Calculate tracking error to nearest path point
            if path:
                distances = [np.linalg.norm(position - path_point) for path_point in path]
                min_distance = min(distances)
                closest_point_idx = np.argmin(distances)
                errors.append(min_distance)

            # Check if reached path end
            if np.linalg.norm(position - path[-1]) < 0.3:
                print(f"  Reached target point! Steps: {step + 1}")
                target_reached = True
                break

        # Enhanced performance analysis
        if errors:
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            final_error = errors[-1] if errors else 0
            error_std = np.std(errors)
            print(f"  Average Tracking Error: {avg_error:.3f}m")
            print(f"  Maximum Tracking Error: {max_error:.3f}m")
            print(f"  Final Tracking Error: {final_error:.3f}m")
            print(f"  Error Standard Deviation: {error_std:.3f}m")
            print(f"  Target Reached: {'YES' if target_reached else 'NO'}")

        # Control stability analysis
        if len(control_signals) > 10:
            throttle_signals = [sig[0] for sig in control_signals]
            steering_signals = [sig[1] for sig in control_signals]
            throttle_variance = np.var(throttle_signals)
            steering_variance = np.var(steering_signals)
            throttle_range = max(throttle_signals) - min(throttle_signals)
            steering_range = max(steering_signals) - min(steering_signals)
            print(f"  Control Stability:")
            print(f"    Throttle Variance: {throttle_variance:.4f}")
            print(f"    Steering Variance: {steering_variance:.4f}")
            print(f"    Throttle Range: {throttle_range:.3f}")
            print(f"    Steering Range: {steering_range:.3f}")

        # Save comprehensive results
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
                'throttle_variance': float(throttle_variance) if len(control_signals) > 10 else 0.0,
                'steering_variance': float(steering_variance) if len(control_signals) > 10 else 0.0
            }
        }
        all_control_results.append(result)

        # Visualize control results
        visualize_control_result(result, path, path_name, simulator)

    # Generate enhanced control performance report
    generate_control_report(all_control_results)

    return all_control_results

def visualize_control_result(result, planned_path, path_name, simulator):
    """Enhanced visualization of control results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    trajectory = np.array(result['trajectory'])
    control_signals = np.array(result['control_signals'])
    errors = result['errors']
    planned_path = np.array(planned_path)

    # 1. Trajectory tracking plot
    ax1.plot(planned_path[:, 0], planned_path[:, 1], 'g--', linewidth=2, label='Planned Path')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Car Trajectory')
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')

    # Draw heading arrows at intervals
    for i in range(0, len(trajectory), 8):
        if i < len(trajectory):
            dx = 0.5 * math.cos(simulator.heading)
            dy = 0.5 * math.sin(simulator.heading)
            ax1.arrow(trajectory[i, 0], trajectory[i, 1], dx, dy,
                     head_width=0.2, head_length=0.3, fc='r', ec='r')

    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.set_title(f'{path_name} - Car Trajectory Tracking')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # 2. Tracking error plot
    ax2.plot(errors, 'r-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Tracking Error (m)')
    ax2.set_title('Car Path Tracking Error Over Time')
    ax2.grid(True)

    # Add error statistics to plot
    avg_error = np.mean(errors) if errors else 0
    ax2.axhline(y=avg_error, color='b', linestyle='--', label=f'Average Error: {avg_error:.3f}m')
    ax2.legend()

    # 3. Control signals plot
    time_steps = range(len(control_signals))
    ax3.plot(time_steps, control_signals[:, 0], 'b-', linewidth=2, label='Throttle')
    ax3.plot(time_steps, control_signals[:, 1], 'r-', linewidth=2, label='Steering')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Control Signal')
    ax3.set_title('Car Control Signals Variation')
    ax3.legend()
    ax3.grid(True)

    # 4. Performance metrics
    metrics = ['Average Error', 'Max Error', 'Final Error']
    values = [
        result['performance']['average_error'],
        result['performance']['max_error'],
        result['performance']['final_error']
    ]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax4.bar(metrics, values, color=colors)
    ax4.set_ylabel('Error (m)')
    ax4.set_title('Car Control Performance Metrics')
    ax4.grid(True, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    # Save image
    filename = f"control_{path_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Control result image saved: {filename}")
    plt.close()

def generate_control_report(results):
    """Generate enhanced car control test report"""
    print("\n" + "=" * 50)
    print("Car Control Module Test Report")
    print("=" * 50)
    
    print(f"Test Paths Count: {len(results)}")

    # Individual path performance
    for result in results:
        perf = result['performance']
        print(f"{result['path_name']}:")
        print(f"  Steps: {perf['steps']}")
        print(f"  Average Error: {perf['average_error']:.3f}m")
        print(f"  Maximum Error: {perf['max_error']:.3f}m")
        print(f"  Final Error: {perf['final_error']:.3f}m")
        print(f"  Error Std: {perf['error_std']:.3f}m")
        print(f"  Target Reached: {perf['target_reached']}")
        if perf['throttle_variance'] > 0:
            print(f"  Throttle Variance: {perf['throttle_variance']:.4f}")
            print(f"  Steering Variance: {perf['steering_variance']:.4f}")

    # Overall statistics
    if results:
        avg_errors = [r['performance']['average_error'] for r in results]
        max_errors = [r['performance']['max_error'] for r in results]
        target_reached_count = sum(1 for r in results if r['performance']['target_reached'])
        best_path = results[np.argmin(avg_errors)]['path_name']
        worst_path = results[np.argmax(avg_errors)]['path_name']

        print(f"Overall Statistics:")
        print(f"  Best Tracking Path: {best_path} (Average Error: {np.min(avg_errors):.3f}m)")
        print(f"  Most Challenging Path: {worst_path} (Average Error: {np.max(avg_errors):.3f}m)")
        print(f"  Overall Average Error: {np.mean(avg_errors):.3f}m")
        print(f"  Overall Maximum Error: {np.mean(max_errors):.3f}m")
        print(f"  Success Rate: {target_reached_count}/{len(results)} paths completed")

        # Performance assessment
        overall_avg_error = np.mean(avg_errors) if results else 0
        if overall_avg_error < 0.5:
            performance_level = "EXCELLENT"
        elif overall_avg_error < 1.0:
            performance_level = "GOOD"
        elif overall_avg_error < 2.0:
            performance_level = "FAIR"
        elif overall_avg_error < 3.0:
            performance_level = "POOR"
        else:
            performance_level = "UNACCEPTABLE"
        print(f"  Car Control Performance: {performance_level}")

    # Save detailed report
    report_data = {
        'summary': {
            'total_tests': len(results),
            'overall_average_error': float(overall_avg_error),
            'success_rate': float(target_reached_count/len(results)*100) if results else 0,
            'performance_level': performance_level,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'detailed_results': []
    }

    # Add simplified results without large arrays
    for result in results:
        # Convert performance metrics to native Python types
        performance_data = {}
        for key, value in result['performance'].items():
            if hasattr(value, 'item'):  # numpy scalar
                performance_data[key] = value.item()
            else:
                performance_data[key] = value

        simplified_result = {
            'path_name': result['path_name'],
            'performance': performance_data
        }
        report_data['detailed_results'].append(simplified_result)

    with open('control_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"Detailed report saved: control_test_report.json")

def generate_system_overview_visualization(perception_score, planning_score, control_score, filename):
    """生成系统性能概览可视化"""
    # 导入必要的库
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    modules = ['感知模块', '规划模块', '控制模块', '综合评分']
    scores = [perception_score, planning_score, control_score, 
              (perception_score * 0.2 + planning_score * 0.3 + control_score * 0.5)]
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    bars = ax.bar(modules, scores, color=colors)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    ax.set_ylim(0, 11)
    ax.set_ylabel('评分 (0-10)')
    ax.set_title('系统各模块性能评估')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"系统性能概览图已保存: {filename}")
    plt.close()

def generate_radar_chart(perception_score, planning_score, control_score, filename):
    """生成系统组件评分雷达图"""
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 数据准备
    categories = ['感知能力', '规划效率', '控制精度']
    values = [perception_score, planning_score, control_score]
    
    # 闭合雷达图
    values = np.concatenate((values, [values[0]]))
    categories = np.concatenate((categories, [categories[0]]))
    
    # 计算角度
    N = len(categories) - 1  # 减1因为重复了第一个元素
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles = np.concatenate((angles, [angles[0]]))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=2, color='#66B2FF', label='当前系统')
    ax.fill(angles, values, alpha=0.25, color='#66B2FF')
    
    # 设置坐标轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # 设置y轴范围
    ax.set_ylim(0, 10)
    
    # 添加网格和标题
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('系统组件能力雷达图', size=15, y=1.1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"系统雷达图已保存: {filename}")
    plt.close()

def run_comprehensive_test():
    """Run comprehensive test of the entire system"""
    print("Starting Comprehensive System Test...")
    print("This test will evaluate all major components of the system")
    
    # 确保可视化目录存在
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Perception Test
    print("\n" + "=" * 50)
    print("PERCEPTION MODULE TEST")
    print("=" * 50)
    detections = test_perception()
    perception_score = min(10, len(detections) * 2 + 2)
    
    # 生成感知模块可视化结果
    test_image = generate_realistic_test_image()
    # 注意：visualize_detections函数需要inference_time参数
    visualization_result = visualize_detections(test_image, detections, 0.0)  # 使用0作为inference_time
    cv2.imwrite('visualizations/perception_module_test.png', visualization_result)
    print("感知模块可视化结果已保存: visualizations/perception_module_test.png")
    
    # 2. Planning Test
    print("\n" + "=" * 50)
    print("PLANNING MODULE TEST")
    print("=" * 50)
    planning_results = test_planning()
    
    successful_plans = 0
    if planning_results:
        successful_plans = len([r for r in planning_results if isinstance(r, dict) and r.get('success', False)])
    planning_score = min(10, (successful_plans / len(planning_results)) * 10) if planning_results else 0
    
    # 为每个规划场景生成可视化结果
    if planning_results:
        for i, result in enumerate(planning_results):
            if isinstance(result, dict) and result.get('success', False):
                scenario = result.get('scenario', {})
                path = result.get('path', [])
                planning_time = result.get('planning_time', 0)
                
                if scenario and isinstance(scenario, dict) and 'name' in scenario:
                    # 创建新的PathPlanner实例
                    planner = PathPlanner()
                    
                    # 安全地添加障碍物
                    if 'obstacles' in scenario and isinstance(scenario['obstacles'], list):
                        for obstacle in scenario['obstacles']:
                            if isinstance(obstacle, dict):
                                planner.add_obstacle(obstacle)
                    
                    # 生成并保存可视化结果
                    try:
                        # 正确调用visualize_planning_scenario函数
                        visualize_planning_scenario(planner, path, scenario, planning_time)
                        # 移动生成的图片到visualizations目录
                        img_name = f"planning_{scenario['name'].replace(' ', '_').replace('-', '_')}.png"
                        if os.path.exists(img_name):
                            new_path = os.path.join('visualizations', img_name)
                            os.rename(img_name, new_path)
                            print(f"规划场景可视化结果已保存: {new_path}")
                    except Exception as e:
                        print(f"生成规划场景可视化时出错: {e}")
    
    # 3. Control Test
    print("\n" + "=" * 50)
    print("CONTROL MODULE TEST")
    print("=" * 50)
    control_results = test_control()
    
    # 计算控制模块得分
    control_score = 0
    if control_results:
        avg_errors = []
        for r in control_results:
            if isinstance(r, dict) and 'performance' in r and isinstance(r['performance'], dict) and 'average_error' in r['performance']:
                avg_errors.append(r['performance']['average_error'])
        
        if avg_errors:
            overall_avg_error = np.mean(avg_errors)
            if overall_avg_error < 0.5:
                control_score = 10
            elif overall_avg_error < 1.0:
                control_score = 8
            elif overall_avg_error < 2.0:
                control_score = 6
            elif overall_avg_error < 3.0:
                control_score = 4
            else:
                control_score = 2
    
    # 为控制测试结果生成可视化
    if control_results:
        for i, result in enumerate(control_results):
            if isinstance(result, dict):
                path_name = result.get('path_name', f'path_{i+1}')
                planned_path = result.get('path', [])
                
                # 创建CarSimulator实例
                simulator = CarSimulator()
                
                try:
                    # 调用visualize_control_result函数
                    visualize_control_result(result, planned_path, path_name, simulator)
                    # 移动生成的图片到visualizations目录
                    img_name = f"control_{path_name.replace(' ', '_')}.png"
                    if os.path.exists(img_name):
                        new_path = os.path.join('visualizations', img_name)
                        os.rename(img_name, new_path)
                        print(f"控制路径可视化结果已保存: {new_path}")
                except Exception as e:
                    print(f"生成控制路径可视化时出错: {e}")
    
    # 生成系统性能概览图
    try:
        filename = 'visualizations/system_performance_overview.png'
        generate_system_overview_visualization(perception_score, planning_score, control_score, filename)
    except Exception as e:
        print(f"生成系统性能概览图时出错: {e}")
    
    # 生成组件评分雷达图
    try:
        filename = 'visualizations/system_radar_chart.png'
        generate_radar_chart(perception_score, planning_score, control_score, filename)
    except Exception as e:
        print(f"生成雷达图时出错: {e}")
    
    # 计算综合得分
    overall_score = (perception_score * 0.2 + planning_score * 0.3 + control_score * 0.5)
    
    # 显示总体评估
    print("\n" + "=" * 50)
    print("COMPREHENSIVE SYSTEM TEST REPORT")
    print("=" * 50)
    print(f"Perception Module Score: {perception_score:.1f}/10")
    print(f"Planning Module Score: {planning_score:.1f}/10")
    print(f"Control Module Score: {control_score:.1f}/10")
    print(f"OVERALL SYSTEM SCORE: {overall_score:.1f}/10")
    
    # 评估等级
    if overall_score >= 8.5:
        print("\nEXCELLENT - System ready for deployment!")
    elif overall_score >= 7.0:
        print("\nGOOD - System functional with minor optimizations needed")
    elif overall_score >= 5.0:
        print("\nFAIR - System operational with improvements needed")
    else:
        print("\nPOOR - System requires major revisions")
        
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



def generate_radar_chart(perception_score, planning_score, control_score, filename):
    """生成系统组件评分雷达图"""
    # 数据准备
    categories = ['感知能力', '规划效率', '控制精度']
    values = [perception_score, planning_score, control_score]
    
    # 闭合雷达图
    values = np.concatenate((values, [values[0]]))
    categories = np.concatenate((categories, [categories[0]]))
    
    # 计算角度
    N = len(categories) - 1  # 减1因为重复了第一个元素
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles = np.concatenate((angles, [angles[0]]))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=2, color='#66B2FF', label='当前系统')
    ax.fill(angles, values, alpha=0.25, color='#66B2FF')
    
    # 设置坐标轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    
    # 设置y轴范围
    ax.set_ylim(0, 10)
    
    # 添加网格和标题
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('系统组件能力雷达图', size=15, y=1.1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"系统雷达图已保存: {filename}")
    plt.close()


if __name__ == "__main__":
    # Run individual tests
    # test_perception()
    # test_planning()
    # test_control()
    # Or run complete test
    comprehensive_results = run_comprehensive_test()