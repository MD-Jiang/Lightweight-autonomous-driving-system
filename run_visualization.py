# run_visualization.py
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from car_system import create_system_components
from visualization import RealTimeVisualizer

def setup_system():
    """Initialize system components"""
    print("Initializing car autonomous system...")
    
    # Create system components
    detector, planner, controller, simulator, monitor = create_system_components()
    
    system_components = {
        'detector': detector,
        'planner': planner,
        'controller': controller, 
        'simulator': simulator,
        'monitor': monitor
    }
    
    return system_components

def run_basic_visualization():
    """Run basic visualization demo"""
    print("=" * 60)
    print("Car Autonomous System Visualization Demo")
    print("=" * 60)
    
    # Initialize system
    system_components = setup_system()
    
    # Create visualizer
    visualizer = RealTimeVisualizer(system_components)
    
    # Set demo parameters
    goal_position = [12, 12]  # Target position
    max_steps = 80           # Maximum steps
    
    print(f"Goal position: {goal_position}")
    print(f"Max steps: {max_steps}")
    print("Starting real-time visualization demo...")
    
    # Start demo
    visualizer.start_real_time_demo(goal_position, max_steps)
    
    print("Demo completed!")

def run_interactive_demo():
    """Run interactive demo"""
    system_components = setup_system()
    visualizer = RealTimeVisualizer(system_components)
    
    # User selects target
    print("\nSelect target position:")
    print("1. Easy target (8, 8)")
    print("2. Medium target (12, 12)") 
    print("3. Challenging target (15, 15)")
    print("4. Custom target")
    
    choice = input("Please choose (1-4): ").strip()
    
    if choice == '1':
        goal = [8, 8]
    elif choice == '2':
        goal = [12, 12]
    elif choice == '3':
        goal = [15, 15]
    elif choice == '4':
        x = float(input("Enter target X coordinate: "))
        y = float(input("Enter target Y coordinate: "))
        goal = [x, y]
    else:
        goal = [10, 10]
    
    # Run demo
    visualizer.start_real_time_demo(goal, max_steps=60)

if __name__ == "__main__":
    # Use default fonts to avoid Chinese character issues
    print("Car Autonomous Visualization System")
    print("1. Run basic demo")
    print("2. Run interactive demo")
    
    choice = input("Please select mode (1-2): ").strip()
    
    if choice == '1':
        run_basic_visualization()
    elif choice == '2':
        run_interactive_demo()
    else:
        print("Invalid choice, running basic demo...")
        run_basic_visualization()
