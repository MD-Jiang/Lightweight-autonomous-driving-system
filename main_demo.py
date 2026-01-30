from car_system import *
from test_modules import *
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def generate_visualization(image, detections, path, position, goal_position, filename):
    """生成系统可视化图像"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 感知结果可视化
        ax1.imshow(image)
        ax1.set_title('Obstacle Detection Results')
        for det in detections:
            x, y, w, h = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x, y - 10, f'{class_name}: {confidence:.2f}', color='red', fontsize=10)
        
        # 路径规划可视化
        ax2.set_title('Path Planning Visualization')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.grid(True)
        
        # 绘制起点
        ax2.scatter(position[0], position[1], color='blue', s=100, label='Current Position')
        ax2.text(position[0] + 0.5, position[1], 'Current', color='blue')
        
        # 绘制目标点
        ax2.scatter(goal_position[0], goal_position[1], color='green', s=100, label='Goal Position')
        ax2.text(goal_position[0] + 0.5, goal_position[1], 'Goal', color='green')
        
        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax2.plot(path_x, path_y, 'r-', linewidth=2, label='Planned Path')
        
        ax2.legend()
        ax2.set_aspect('equal')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"生成可视化图像: {filename}")
    except Exception as e:
        print(f"生成可视化图像时出错: {e}")

def run_simple_demo():
    """Run improved car obstacle avoidance demonstration"""
    print("Starting Improved Car Autonomous Driving Demo...")

    # Initialize system with improved components
    if YOLOv8_AVAILABLE:
        detector = YOLOv8ObstacleDetector(conf_threshold=0.3)
        print("Using YOLOv8 Deep Learning Detector")
    else:
        detector = ObstacleDetector(conf_threshold=0.3)
        print("Using traditional image detector")

    planner = PathPlanner()
    controller = CarController()
    simulator = CarSimulator(init_position=[2, 2])
    monitor = SystemMonitor()

    # Set more realistic target for car
    goal_position = [20, 20]  # Further target for car testing
    print("Improved car autonomous system initialization completed")
    print(f"Start Position: {simulator.position}")
    print(f"Goal Position: {goal_position}")
    print("Using enhanced adaptive control strategy for car")
    print("Starting autonomous driving operation...")

    # 确保可视化目录存在
    os.makedirs('visualizations', exist_ok=True)

    # Run demonstration
    max_steps = 200
    success_threshold = 0.5  # Adjusted for car

    for step in range(max_steps):
        if step % 15 == 0:  # Display status every 15 steps
            print(f"\n--- Step {step + 1} ---")

        # 1. Perception
        image = generate_realistic_test_image()
        start_time = time.time()
        detections = detector.detect(image)
        perception_time = (time.time() - start_time) * 1000

        if step % 15 == 0:
            print(f"Perception: {len(detections)} obstacles detected in {perception_time:.1f}ms")

        # Record perception performance
        monitor.record_perception(detections, perception_time)

        # 2. Planning
        planner.update_obstacle_grid(detections, simulator.position)
        start_time = time.time()
        path = planner.astar_search(
            (simulator.position[0]*3, simulator.position[1]*3),
            (goal_position[0]*3, goal_position[1]*3)
        )
        planning_time = (time.time() - start_time) * 1000
        world_path = [(p[0]/3, p[1]/3) for p in path] if path else []

        if step % 15 == 0 and path:
            print(f"Planning: {len(world_path)} path points generated in {planning_time:.1f}ms")
            if len(world_path) > 0:
                print(f"Next target: ({world_path[0][0]:.1f}, {world_path[0][1]:.1f})")

        # Record planning performance
        monitor.record_planning(planning_time)

        # 3. Control
        throttle, steering = controller.update_control(
            simulator.position, simulator.heading, world_path
        )

        # 4. Execution
        position, heading = simulator.update(throttle, steering)
        
        # 每30步生成一次可视化图像
        if step % 30 == 0:
            filename = f'visualizations/simple_demo_step_{step}.png'
            generate_visualization(image, detections, world_path, position, goal_position, filename)

        # Calculate tracking error
        if world_path:
            current_target = world_path[0]  # Next immediate target
            tracking_error = np.linalg.norm(position - current_target)
            monitor.record_control(tracking_error)
        else:
            tracking_error = np.linalg.norm(position - goal_position)

        if step % 15 == 0:
            print(f"Control: Throttle={throttle:.2f}, Steering={steering:.3f}")
            print(f"Status: Position=({position[0]:.2f}, {position[1]:.2f}), Heading={heading:.2f}")
            print(f"Tracking Error: {tracking_error:.3f}m")

        # Check if reached goal
        dist_to_goal = np.linalg.norm(position - goal_position)
        if dist_to_goal < success_threshold:
            print(f"\nSUCCESS: Car reached target point in {step + 1} steps!")
            print(f"Final position: ({position[0]:.2f}, {position[1]:.2f})")
            print(f"Target position: ({goal_position[0]:.2f}, {goal_position[1]:.2f})")
            print(f"Final error: {dist_to_goal:.3f}m")
            # 生成最终状态的可视化图像
            filename = 'visualizations/simple_demo_final.png'
            generate_visualization(image, detections, world_path, position, goal_position, filename)
            break

        # Display progress every 15 steps
        if step % 15 == 0:
            print(f"Progress: {dist_to_goal:.2f} meters to target")

        # Emergency stop if stuck or going too far
        if step > 80 and np.linalg.norm(position - [2, 2]) > 25:
            print("\nEmergency stop: Car moving too far from start")
            break

        # Check if stuck (minimal movement for too long)
        if step > 50 and step % 20 == 0:
            recent_movement = np.linalg.norm(position - simulator.position)
            if recent_movement < 0.1:
                print("\nCar appears stuck, attempting recovery...")
                # Reset controller to recover
                controller.reset()

    if step == max_steps - 1:
        final_dist = np.linalg.norm(position - goal_position)
        print(f"\nDemo ended: Maximum steps reached.")
        print(f"Final distance to target: {final_dist:.2f} meters")
        print(f"Final position: ({position[0]:.2f}, {position[1]:.2f})")
    else:
        final_dist = np.linalg.norm(position - goal_position)
        print(f"\nDemo completed successfully!")
        print(f"Final distance to target: {final_dist:.2f} meters")

    # Display performance summary
    print("\n" + "=" * 50)
    print("CAR AUTONOMOUS DRIVING DEMO PERFORMANCE SUMMARY")
    print("=" * 50)

    return {
        'final_position': position,
        'goal_position': goal_position,
        'final_error': final_dist,
        'steps': step + 1,
        'performance': monitor.generate_report()
    }

def run_advanced_demo():
    """Run advanced demonstration with multiple waypoints and improved car control"""
    print("Starting Advanced Car Autonomous Navigation Demo...")

    # Initialize system with improved components
    if YOLOv8_AVAILABLE:
        detector = YOLOv8ObstacleDetector(conf_threshold=0.3)
        print("Using YOLOv8 Deep Learning Detector")
    else:
        detector = ObstacleDetector()
        print("Using traditional image detector")

    planner = PathPlanner()
    controller = CarController()
    simulator = CarSimulator(init_position=[2, 2])
    monitor = SystemMonitor()

    # Multiple waypoints with varied challenges for car
    waypoints = [
        [5, 5],    # Simple straight line
        [10, 3],   # Gentle turn
        [8, 8],    # Diagonal movement
        [12, 5],   # Complex maneuver
        [15, 10]   # Final target
    ]

    current_waypoint = 0
    total_steps = 0
    max_total_steps = 400
    waypoint_success_threshold = 0.4

    print("Advanced car autonomous system initialization completed")
    print(f"Start Position: {simulator.position}")
    print(f"Waypoints: {waypoints}")
    print("Using enhanced adaptive control system for car")
    print("Starting advanced car navigation...")

    waypoint_performance = []

    while current_waypoint < len(waypoints) and total_steps < max_total_steps:
        goal_position = waypoints[current_waypoint]

        if total_steps % 25 == 0:
            print(f"\n--- Navigating to Waypoint {current_waypoint + 1}: {goal_position} ---")
            print(f"Total Steps: {total_steps}")
            print(f"Current Position: ({simulator.position[0]:.2f}, {simulator.position[1]:.2f})")

        # 1. Perception
        image = generate_realistic_test_image()
        start_time = time.time()
        detections = detector.detect(image)
        perception_time = (time.time() - start_time) * 1000

        if total_steps % 25 == 0 and len(detections) > 0:
            print(f"Perception: {len(detections)} obstacles detected")

        monitor.record_perception(detections, perception_time)

        # 2. Planning
        planner.update_obstacle_grid(detections, simulator.position)
        start_time = time.time()
        path = planner.astar_search(
            (simulator.position[0]*3, simulator.position[1]*3),
            (goal_position[0]*3, goal_position[1]*3)
        )
        planning_time = (time.time() - start_time) * 1000
        world_path = [(p[0]/3, p[1]/3) for p in path] if path else []

        if total_steps % 25 == 0 and path:
            print(f"Planning: {len(world_path)} path points to waypoint")

        monitor.record_planning(planning_time)

        # 3. Control
        throttle, steering = controller.update_control(
            simulator.position, simulator.heading, world_path
        )

        # 4. Execution
        position, heading = simulator.update(throttle, steering)
        
        # 每25步生成一次可视化图像
        if total_steps % 25 == 0:
            filename = f'visualizations/advanced_demo_waypoint_{current_waypoint}_step_{total_steps}.png'
            generate_visualization(image, detections, world_path, position, goal_position, filename)
        
        # Calculate tracking error
        if world_path:
            current_target = world_path[0]
            tracking_error = np.linalg.norm(position - current_target)
            monitor.record_control(tracking_error)

        if total_steps % 25 == 0:
            print(f"Control: Throttle={throttle:.2f}, Steering={steering:.3f}")
            print(f"Position: ({position[0]:.2f}, {position[1]:.2f})")

        # Check if reached current waypoint
        dist_to_waypoint = np.linalg.norm(position - goal_position)
        if dist_to_waypoint < waypoint_success_threshold:
            waypoint_time = total_steps
            print(f"\nReached Waypoint {current_waypoint + 1} in {waypoint_time} steps!")
            print(f"Position error: {dist_to_waypoint:.3f}m")
            waypoint_performance.append({
                'waypoint': current_waypoint + 1,
                'position': goal_position,
                'steps': waypoint_time,
                'error': dist_to_waypoint
            })
            current_waypoint += 1
            if current_waypoint < len(waypoints):
                print(f"Proceeding to next waypoint: {waypoints[current_waypoint]}")
                # Brief pause between waypoints
                time.sleep(0.5)
                continue

        total_steps += 1

        # Emergency stop conditions
        if total_steps > 150 and np.linalg.norm(position - [2, 2]) > 30:
            print("\nEmergency stop: Car moving too far from start")
            break

        # Stuck detection
        if total_steps % 30 == 0 and total_steps > 100:
            recent_movement = np.linalg.norm(position - simulator.position)
            if recent_movement < 0.2:
                print("\nCar appears stuck, resetting controller...")
                controller.reset()

    # Demo completion summary
    print("\n" + "=" * 50)
    print("ADVANCED CAR AUTONOMOUS DRIVING DEMO COMPLETION REPORT")
    print("=" * 50)

    if current_waypoint == len(waypoints):
        print(f"SUCCESS: Successfully visited all {len(waypoints)} waypoints!")
    else:
        print(f"Completed {current_waypoint} out of {len(waypoints)} waypoints.")

    # Display waypoint performance
    if waypoint_performance:
        print("\nWaypoint Performance:")
        for wp in waypoint_performance:
            print(f"  Waypoint {wp['waypoint']}: {wp['steps']} steps, error: {wp['error']:.3f}m")

    # Overall performance
    print(f"\nTotal Steps: {total_steps}")
    print(f"Final Position: ({position[0]:.2f}, {position[1]:.2f})")

    # System performance summary
    monitor.print_performance_summary()

    return {
        'waypoints_reached': current_waypoint,
        'total_waypoints': len(waypoints),
        'total_steps': total_steps,
        'final_position': position,
        'waypoint_performance': waypoint_performance,
        'system_performance': monitor.generate_report()
    }

def run_obstacle_avoidance_demo():
    """Specialized demo focusing on car obstacle avoidance capabilities"""
    print("Starting Car Obstacle Avoidance Specialized Demo...")

    # Initialize system
    detector = YOLOv8ObstacleDetector(conf_threshold=0.3) if YOLOv8_AVAILABLE else ObstacleDetector()
    planner = PathPlanner()
    controller = CarController()
    simulator = CarSimulator(init_position=[2, 2])
    monitor = SystemMonitor()

    # Challenging scenario with obstacles for car
    goal_position = [12, 12]
    print("Car Obstacle Avoidance Demo Initialized")
    print(f"Start: {simulator.position}, Goal: {goal_position}")
    print("This demo tests the car's ability to navigate around obstacles")
    print("Starting car obstacle avoidance test...")
    
    # 确保可视化目录存在
    os.makedirs('visualizations', exist_ok=True)

    max_steps = 250
    obstacle_encounters = 0

    for step in range(max_steps):
        if step % 20 == 0:
            print(f"\n--- Step {step + 1} ---")

        # Generate test image with obstacles
        image = generate_realistic_test_image()

        # Perception
        detections = detector.detect(image)

        # Count obstacles
        if len(detections) > 0:
            if step % 20 == 0:
                print(f"Obstacles detected: {len(detections)}")
            obstacle_encounters += len(detections)

        # Planning with obstacles
        planner.update_obstacle_grid(detections, simulator.position)
        path = planner.astar_search(
            (simulator.position[0]*3, simulator.position[1]*3),
            (goal_position[0]*3, goal_position[1]*3)
        )
        world_path = [(p[0]/3, p[1]/3) for p in path] if path else []

        if step % 20 == 0 and path:
            path_length = len(world_path)
            print(f"Path planned: {path_length} points")
            if len(detections) > 0:
                print("  (Path adjusted for obstacle avoidance)")

        # Control and execution
        throttle, steering = controller.update_control(
            simulator.position, simulator.heading, world_path
        )
        position, heading = simulator.update(throttle, steering)

        if step % 20 == 0:
            print(f"Control: T={throttle:.2f}, S={steering:.3f}")
            print(f"Position: ({position[0]:.2f}, {position[1]:.2f})")

        # 每20步生成一次可视化图像
        if step % 20 == 0:
            filename = f'visualizations/obstacle_avoidance_step_{step}.png'
            generate_visualization(image, detections, world_path, position, goal_position, filename)

        # Check completion
        if np.linalg.norm(position - goal_position) < 0.4:
            print(f"\nSUCCESS: Car reached goal while avoiding {obstacle_encounters} obstacles!")
            # 生成到达目标的可视化图像
            filename = 'visualizations/obstacle_avoidance_reached.png'
            generate_visualization(image, detections, world_path, position, goal_position, filename)
            break

    # 生成最终状态的可视化图像
    if image is not None and detections is not None:
        filename = 'visualizations/obstacle_avoidance_final.png'
        generate_visualization(image, detections, world_path, position, goal_position, filename)

    # Demo summary
    final_error = np.linalg.norm(position - goal_position)
    print("\n" + "=" * 50)
    print("CAR OBSTACLE AVOIDANCE DEMO SUMMARY")
    print("=" * 50)
    print(f"Final Position: ({position[0]:.2f}, {position[1]:.2f})")
    print(f"Target Position: ({goal_position[0]:.2f}, {goal_position[1]:.2f})")
    print(f"Final Error: {final_error:.3f}m")
    print(f"Obstacle Encounters: {obstacle_encounters}")
    print(f"Steps Taken: {step + 1}")

    # Performance assessment
    if final_error < 0.5:
        print("Performance: EXCELLENT - Successful car obstacle avoidance")
    elif final_error < 1.0:
        print("Performance: GOOD - Acceptable car obstacle avoidance")
    else:
        print("Performance: NEEDS IMPROVEMENT - Car obstacle avoidance challenged")

    return {
        'final_error': final_error,
        'obstacle_encounters': obstacle_encounters,
        'steps': step + 1,
        'success': final_error < 0.5
    }

def test_system_performance():
    """Enhanced car system performance test with detailed metrics"""
    print("Starting Enhanced Car Autonomous System Performance Test...")

    # Test perception
    print("\n" + "=" * 50)
    print("CAR PERCEPTION MODULE PERFORMANCE TEST")
    print("=" * 50)
    detections = test_perception()

    # Test planning
    print("\n" + "=" * 50)
    print("CAR PATH PLANNING MODULE PERFORMANCE TEST")
    print("=" * 50)
    planning_results = test_planning()

    # Test control
    print("\n" + "=" * 50)
    print("CAR CONTROL MODULE PERFORMANCE TEST")
    print("=" * 50)
    control_results = test_control()

    # Generate comprehensive performance report
    print("\n" + "=" * 50)
    print("ENHANCED CAR AUTONOMOUS SYSTEM PERFORMANCE REPORT")
    print("=" * 50)

    # Perception performance
    perception_score = min(10, len(detections) * 2 + 2)  # Bonus for detections
    print(f"Car Perception Module Score: {perception_score:.1f}/10")
    print(f"  - Objects Detected: {len(detections)}")
    if len(detections) > 0:
        avg_confidence = np.mean([det['confidence'] for det in detections])
        print(f"  - Average Confidence: {avg_confidence:.3f}")

    # Planning performance
    successful_plans = len([r for r in planning_results if r['success']])
    planning_score = min(10, (successful_plans / len(planning_results)) * 10)
    if successful_plans > 0:
        avg_planning_time = np.mean([r['planning_time'] for r in planning_results if r['success']])
        avg_path_length = np.mean([r['path_length'] for r in planning_results if r['success']])
        print(f"Car Planning Module Score: {planning_score:.1f}/10")
        print(f"  - Success Rate: {successful_plans}/{len(planning_results)}")
        print(f"  - Average Planning Time: {avg_planning_time:.1f}ms")
        print(f"  - Average Path Length: {avg_path_length:.1f} points")
    else:
        planning_score = 0
        print("Car Planning Module Score: 0/10 (No successful plans)")

    # Control performance
    if control_results and len(control_results) > 0:
        avg_errors = [r['performance']['average_error'] for r in control_results]
        overall_avg_error = np.mean(avg_errors) if avg_errors else 0

        # Enhanced scoring based on error magnitude for car
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

        best_path = control_results[np.argmin(avg_errors)]['path_name']
        worst_path = control_results[np.argmax(avg_errors)]['path_name']

        print(f"Car Control Module Score: {control_score:.1f}/10")
        print(f"  - Average Tracking Error: {overall_avg_error:.3f}m")
        print(f"  - Best Performance: {best_path} ({np.min(avg_errors):.3f}m)")
        print(f"  - Most Challenging: {worst_path} ({np.max(avg_errors):.3f}m)")
    else:
        control_score = 0
        print("Car Control Module Score: 0/10 (No control data)")

    # Overall car system score with weighted components
    overall_score = (perception_score * 0.2 + planning_score * 0.3 + control_score * 0.5)
    print(f"\nOVERALL CAR SYSTEM SCORE: {overall_score:.1f}/10")

    # Detailed assessment
    if overall_score >= 8.5:
        print("EXCELLENT - Car autonomous system ready for real-world deployment!")
        print("  All modules performing at high level")
    elif overall_score >= 7.0:
        print("VERY GOOD - Car system functional with minor optimizations needed")
        print("  Suitable for controlled environment deployment")
    elif overall_score >= 5.0:
        print("GOOD - Car system operational with some improvements needed")
        print("  Ready for further testing and development")
    elif overall_score >= 3.0:
        print("FAIR - Car system needs significant improvements")
        print("  Core functionality present but requires optimization")
    else:
        print("POOR - Car system requires major revisions")
        print("  Fundamental issues need to be addressed")

    # Recommendations
    print("\nCAR SYSTEM RECOMMENDATIONS:")
    if control_score < 6:
        print("  - Focus on car control algorithm optimization")
        print("  - Tune PID parameters and adaptive control strategies")
    if planning_score < 7:
        print("  - Improve path planning for complex car scenarios")
        print("  - Enhance obstacle mapping accuracy")
    if perception_score < 6:
        print("  - Enhance car object detection sensitivity")
        print("  - Consider model retraining or parameter adjustment")

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

def main():    
    """Enhanced main menu for car autonomous system with improved demo options"""
    # Display system information
    if YOLOv8_AVAILABLE:
        print("Current: YOLOv8 Deep Learning Detector")
    else:
        print("Current: Traditional Image Detector")
    print("Car Autonomous Driving System v2.0")
    print("With Improved Control Algorithms and Adaptive Strategies")
    
    # Auto-test mode - run comprehensive test directly
    print("\nStarting Auto-Test Mode...")
    if YOLOv8_AVAILABLE:
        print("Running Comprehensive Test automatically...")
        run_comprehensive_test()
        print("\nTest completed successfully!")
        return
    else:
        print("Running Simple Demo automatically...")
        run_simple_demo()
        print("\nDemo completed successfully!")
        return
    
    # Interactive mode (disabled in auto-test mode)
    while True:
        print("\n" + "=" * 50)
        print("CAR AUTONOMOUS DRIVING SYSTEM")
        print("=" * 50)
        print("1. Run Simple Demo (Improved Car Control)")
        print("2. Run Advanced Demo (Multiple Waypoints)")
        print("3. Run Obstacle Avoidance Demo")
        print("4. Test Perception Module")

        # Dynamic menu generation
        if YOLOv8_AVAILABLE:
            print("5. Test YOLOv8 Perception Module")
            print("6. Test Path Planning Module")
            print("7. Test Control Module")
            print("8. Run Comprehensive Test")
            print("9. System Performance Test")
            print("0. Exit System")
        else:
            print("5. Test Path Planning Module")
            print("6. Test Control Module")
            print("7. Run Comprehensive Test")
            print("8. System Performance Test")
            print("9. Exit System")

        if YOLOv8_AVAILABLE:
            choice = input("\nSelect operation (0-9): ").strip()
        else:
            choice = input("\nSelect operation (1-9): ").strip()

        if choice == '1':
            run_simple_demo()
        elif choice == '2':
            run_advanced_demo()
        elif choice == '3':
            run_obstacle_avoidance_demo()
        elif choice == '4':
            test_perception()
        elif YOLOv8_AVAILABLE:
            if choice == '5':
                test_yolov8_perception()
            elif choice == '6':
                test_planning()
            elif choice == '7':
                test_control()
            elif choice == '8':
                run_comprehensive_test()
            elif choice == '9':
                test_system_performance()
            elif choice == '0':
                print("\nThank you for using Car Autonomous Driving System!")
                print("Goodbye!")
                break
            else:
                print("Invalid selection, please try again!")
        else:
            if choice == '5':
                test_planning()
            elif choice == '6':
                test_control()
            elif choice == '7':
                run_comprehensive_test()
            elif choice == '8':
                test_system_performance()
            elif choice == '9':
                print("\nThank you for using Car Autonomous Driving System!")
                print("Goodbye!")
                break
            else:
                print("Invalid selection, please try again!")

        # Brief pause between operations
        time.sleep(1)

if __name__ == "__main__":
    main()