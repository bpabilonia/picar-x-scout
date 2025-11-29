#!/usr/bin/env python3
"""
SLAM Navigation System for PiCar-X Scout

This module provides comprehensive indoor navigation capabilities:
1. Room Mapping - Build occupancy grid maps of indoor environments
2. Obstacle Avoidance - Real-time dynamic obstacle detection and avoidance
3. Path Planning - A* based path planning with autonomous patrol modes

Supported SLAM Algorithms:
- GMapping (ROS-based) - Requires ROS installation
- Hector SLAM (for high-frequency LIDAR) - Requires ROS installation
- Custom EKF SLAM (for ultrasonic/camera setups) - No ROS required

Hardware Support:
- RPLIDAR A1/A2/A3 via rplidar SDK
- Ultrasonic sensors (built into PiCar-X)
- Camera-based visual odometry via OpenCV

Visualization:
- matplotlib for static map visualization
- pygame for real-time interactive display

Usage:
    # Non-ROS mode with ultrasonic/camera (default):
    sudo python3 slam_navigation.py
    
    # ROS mode with LIDAR:
    rosrun picar_scout slam_navigation.py --ros --lidar
    
    # Keyboard control mode:
    sudo python3 slam_navigation.py --keyboard

Dependencies:
    - OpenCV (cv2)
    - numpy
    - matplotlib or pygame
    - rplidar (optional, for LIDAR)
    - rospy (optional, for ROS integration)
"""

import os
import sys
import time
import math
import threading
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import heapq

# Core dependencies
import numpy as np

# Optional imports with graceful fallbacks
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Vision features disabled.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless operation
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Static visualization disabled.")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Real-time visualization disabled.")

# ROS imports (optional)
ROS_AVAILABLE = False
try:
    import rospy
    from nav_msgs.msg import OccupancyGrid as ROSOccupancyGrid, Odometry, Path
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import PoseStamped, Twist
    from tf.transformations import euler_from_quaternion
    ROS_AVAILABLE = True
except ImportError:
    # Define placeholder types for type hints when ROS not available
    ROSOccupancyGrid = None
    Odometry = None
    LaserScan = None
    Twist = None

# RPLIDAR SDK (optional)
RPLIDAR_AVAILABLE = False
try:
    from rplidar import RPLidar
    RPLIDAR_AVAILABLE = True
except ImportError:
    pass

# PiCar-X imports
try:
    from picarx import Picarx
    from robot_hat import Pin
    PICARX_AVAILABLE = True
except ImportError:
    PICARX_AVAILABLE = False
    print("Warning: PiCar-X library not available. Running in simulation mode.")

# Local imports
try:
    from utils import gray_print, warn, error
except ImportError:
    def gray_print(msg): print(f"\033[1;30m{msg}\033[0m")
    def warn(msg): print(f"\033[0;33m{msg}\033[0m")
    def error(msg): print(f"\033[0;31m{msg}\033[0m")

# ============================================================================
# Configuration
# ============================================================================
ROBOT_NAME = "Scout"

# Map parameters
MAP_RESOLUTION = 0.05  # meters per cell (5cm resolution)
MAP_WIDTH = 200  # cells (10 meters)
MAP_HEIGHT = 200  # cells (10 meters)
MAP_ORIGIN_X = MAP_WIDTH // 2  # Robot starts at center
MAP_ORIGIN_Y = MAP_HEIGHT // 2

# Occupancy grid values
UNKNOWN = -1
FREE = 0
OCCUPIED = 100

# Robot physical parameters
ROBOT_RADIUS = 0.15  # meters (15cm radius for collision checking)
WHEEL_BASE = 0.12  # meters
MAX_SPEED = 0.5  # m/s
MAX_TURN_RATE = 1.5  # rad/s

# Sensor parameters
ULTRASONIC_MAX_RANGE = 4.0  # meters
ULTRASONIC_MIN_RANGE = 0.02  # meters
LIDAR_MAX_RANGE = 12.0  # meters
LIDAR_MIN_RANGE = 0.15  # meters

# Path planning
PATH_PLANNING_FREQUENCY = 2.0  # Hz
GOAL_TOLERANCE = 0.1  # meters
HEADING_TOLERANCE = 0.2  # radians

# Patrol parameters
PATROL_SPEED = 25  # PiCar-X speed (0-100)
PATROL_WAYPOINT_RADIUS = 0.3  # meters - how close to get to waypoint

# Visualization
VIZ_UPDATE_RATE = 10  # Hz
VIZ_WINDOW_SIZE = (800, 800)


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class Pose2D:
    """2D robot pose (x, y, theta)."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Pose2D':
        return cls(x=arr[0], y=arr[1], theta=arr[2])


@dataclass
class LaserReading:
    """Single laser/sonar reading."""
    angle: float  # radians
    distance: float  # meters
    intensity: float = 1.0


@dataclass
class ScanData:
    """Collection of sensor readings."""
    timestamp: float
    readings: List[LaserReading]
    pose: Pose2D


@dataclass
class MapCell:
    """Occupancy grid cell with log-odds representation."""
    log_odds: float = 0.0
    last_update: float = 0.0
    
    @property
    def probability(self) -> float:
        """Convert log-odds to probability."""
        return 1.0 - 1.0 / (1.0 + math.exp(self.log_odds))
    
    @property
    def occupancy_value(self) -> int:
        """Get occupancy grid value (-1: unknown, 0-100: probability)."""
        if abs(self.log_odds) < 0.5:
            return UNKNOWN
        return int(self.probability * 100)


class NavigationMode(Enum):
    """Navigation modes."""
    IDLE = "idle"
    MAPPING = "mapping"
    PATROL = "patrol"
    GOTO = "goto"
    EXPLORE = "explore"


# ============================================================================
# Extended Kalman Filter SLAM
# ============================================================================
class EKFSLAM:
    """
    Extended Kalman Filter SLAM implementation.
    
    Works with ultrasonic sensors and wheel odometry.
    Maintains robot pose and landmark positions.
    """
    
    def __init__(self):
        # State vector: [x, y, theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]
        self.state = np.zeros(3)  # Initial pose
        self.covariance = np.eye(3) * 0.1  # Initial uncertainty
        
        # Landmarks (detected features)
        self.landmarks = {}  # id -> (x, y)
        self.landmark_count = 0
        
        # Motion noise parameters
        self.alpha = np.array([0.1, 0.01, 0.01, 0.1])  # Motion noise
        
        # Observation noise
        self.Q = np.diag([0.1, 0.1])  # Range, bearing noise
        
        # Data association threshold
        self.association_threshold = 0.5  # meters
        
        self.lock = threading.Lock()
    
    def predict(self, v: float, omega: float, dt: float):
        """
        Predict step using motion model.
        
        Args:
            v: Linear velocity (m/s)
            omega: Angular velocity (rad/s)
            dt: Time step (seconds)
        """
        with self.lock:
            theta = self.state[2]
            
            # Motion model
            if abs(omega) > 1e-5:
                # Arc motion
                dx = -v/omega * math.sin(theta) + v/omega * math.sin(theta + omega*dt)
                dy = v/omega * math.cos(theta) - v/omega * math.cos(theta + omega*dt)
                dtheta = omega * dt
            else:
                # Straight line motion
                dx = v * dt * math.cos(theta)
                dy = v * dt * math.sin(theta)
                dtheta = 0
            
            # Update state
            self.state[0] += dx
            self.state[1] += dy
            self.state[2] += dtheta
            self.state[2] = self._normalize_angle(self.state[2])
            
            # Jacobian of motion model
            G = np.eye(len(self.state))
            if abs(omega) > 1e-5:
                G[0, 2] = -v/omega * math.cos(theta) + v/omega * math.cos(theta + omega*dt)
                G[1, 2] = -v/omega * math.sin(theta) + v/omega * math.sin(theta + omega*dt)
            else:
                G[0, 2] = -v * dt * math.sin(theta)
                G[1, 2] = v * dt * math.cos(theta)
            
            # Motion noise
            R = np.zeros((len(self.state), len(self.state)))
            R[0, 0] = self.alpha[0] * v**2 + self.alpha[1] * omega**2
            R[1, 1] = self.alpha[0] * v**2 + self.alpha[1] * omega**2
            R[2, 2] = self.alpha[2] * v**2 + self.alpha[3] * omega**2
            
            # Update covariance
            self.covariance = G @ self.covariance @ G.T + R
    
    def update(self, observations: List[Tuple[float, float, int]]):
        """
        Update step with observations.
        
        Args:
            observations: List of (range, bearing, landmark_id) tuples
                         Use landmark_id=-1 for new landmarks
        """
        with self.lock:
            for r, phi, lm_id in observations:
                if lm_id < 0:
                    # New landmark
                    lm_id = self._add_landmark(r, phi)
                
                if lm_id in self.landmarks:
                    self._update_with_landmark(r, phi, lm_id)
    
    def _add_landmark(self, r: float, phi: float) -> int:
        """Add a new landmark to the state."""
        # Calculate landmark position in world frame
        x = self.state[0] + r * math.cos(self.state[2] + phi)
        y = self.state[1] + r * math.sin(self.state[2] + phi)
        
        lm_id = self.landmark_count
        self.landmarks[lm_id] = (x, y)
        self.landmark_count += 1
        
        # Extend state and covariance
        new_state = np.zeros(len(self.state) + 2)
        new_state[:len(self.state)] = self.state
        new_state[-2] = x
        new_state[-1] = y
        self.state = new_state
        
        new_cov = np.eye(len(self.state)) * 1000  # High initial uncertainty for new landmark
        new_cov[:len(self.covariance), :len(self.covariance)] = self.covariance
        self.covariance = new_cov
        
        return lm_id
    
    def _update_with_landmark(self, r: float, phi: float, lm_id: int):
        """Update state using landmark observation."""
        lm_x, lm_y = self.landmarks[lm_id]
        
        # Expected observation
        dx = lm_x - self.state[0]
        dy = lm_y - self.state[1]
        q = dx**2 + dy**2
        
        expected_r = math.sqrt(q)
        expected_phi = math.atan2(dy, dx) - self.state[2]
        expected_phi = self._normalize_angle(expected_phi)
        
        # Innovation
        z = np.array([r, phi])
        z_expected = np.array([expected_r, expected_phi])
        innovation = z - z_expected
        innovation[1] = self._normalize_angle(innovation[1])
        
        # Jacobian
        H = np.zeros((2, len(self.state)))
        H[0, 0] = -dx / expected_r
        H[0, 1] = -dy / expected_r
        H[0, 2] = 0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1
        
        # Kalman gain
        S = H @ self.covariance @ H.T + self.Q
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update
        self.state = self.state + K @ innovation
        self.state[2] = self._normalize_angle(self.state[2])
        self.covariance = (np.eye(len(self.state)) - K @ H) @ self.covariance
        
        # Update landmark position in dictionary
        lm_idx = 3 + lm_id * 2
        if lm_idx + 1 < len(self.state):
            self.landmarks[lm_id] = (self.state[lm_idx], self.state[lm_idx + 1])
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_pose(self) -> Pose2D:
        """Get current robot pose."""
        with self.lock:
            return Pose2D(x=self.state[0], y=self.state[1], theta=self.state[2])
    
    def get_landmarks(self) -> Dict[int, Tuple[float, float]]:
        """Get all landmarks."""
        with self.lock:
            return self.landmarks.copy()


# ============================================================================
# Occupancy Grid Map
# ============================================================================
class OccupancyGridMap:
    """
    Probabilistic occupancy grid map using log-odds representation.
    
    Supports:
    - Ray casting for sensor updates
    - Map saving/loading
    - Visualization
    """
    
    def __init__(self, width: int = MAP_WIDTH, height: int = MAP_HEIGHT, 
                 resolution: float = MAP_RESOLUTION):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Origin in world coordinates (meters)
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2
        
        # Log-odds grid
        self.grid = np.zeros((height, width), dtype=np.float32)
        
        # Log-odds parameters
        self.l_occ = 0.85  # Log-odds for occupied
        self.l_free = -0.4  # Log-odds for free
        self.l_max = 10.0  # Clamp value
        self.l_min = -10.0
        
        self.lock = threading.Lock()
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x = gx * self.resolution + self.origin_x + self.resolution / 2
        y = gy * self.resolution + self.origin_y + self.resolution / 2
        return x, y
    
    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are valid."""
        return 0 <= gx < self.width and 0 <= gy < self.height
    
    def update_with_scan(self, pose: Pose2D, scan: ScanData):
        """
        Update map with a laser/sonar scan using ray casting.
        
        Args:
            pose: Robot pose when scan was taken
            scan: Sensor readings
        """
        with self.lock:
            for reading in scan.readings:
                if reading.distance < 0.01 or reading.distance > ULTRASONIC_MAX_RANGE:
                    continue
                
                # Calculate end point in world frame
                angle = pose.theta + reading.angle
                end_x = pose.x + reading.distance * math.cos(angle)
                end_y = pose.y + reading.distance * math.sin(angle)
                
                # Ray cast from robot to end point
                self._ray_cast(pose.x, pose.y, end_x, end_y, hit=True)
    
    def _ray_cast(self, x0: float, y0: float, x1: float, y1: float, hit: bool):
        """
        Cast a ray and update cells along the path.
        
        Uses Bresenham's line algorithm for efficiency.
        """
        gx0, gy0 = self.world_to_grid(x0, y0)
        gx1, gy1 = self.world_to_grid(x1, y1)
        
        # Get cells along the ray
        cells = self._bresenham(gx0, gy0, gx1, gy1)
        
        # Update all cells except the last as free
        for gx, gy in cells[:-1]:
            if self.is_valid(gx, gy):
                self.grid[gy, gx] += self.l_free
                self.grid[gy, gx] = max(self.l_min, self.grid[gy, gx])
        
        # Update last cell (hit point) as occupied
        if cells and hit:
            gx, gy = cells[-1]
            if self.is_valid(gx, gy):
                self.grid[gy, gx] += self.l_occ
                self.grid[gy, gx] = min(self.l_max, self.grid[gy, gx])
    
    def _bresenham(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm."""
        cells = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            cells.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return cells
    
    def get_occupancy_grid(self) -> np.ndarray:
        """Get occupancy grid as values -1 (unknown), 0-100 (probability %).
        
        Cells with log-odds close to 0 (never observed) are marked as -1 (unknown).
        """
        with self.lock:
            # Convert log-odds to probability
            prob = 1.0 - 1.0 / (1.0 + np.exp(self.grid))
            result = (prob * 100).astype(np.int8)
            
            # Mark cells with log-odds near 0 as unknown (-1)
            # These are cells that haven't been observed enough
            unknown_mask = np.abs(self.grid) < 0.1
            result[unknown_mask] = -1
            
            return result
    
    def get_binary_map(self, threshold: float = 0.6) -> np.ndarray:
        """Get binary map (True = occupied)."""
        with self.lock:
            prob = 1.0 - 1.0 / (1.0 + np.exp(self.grid))
            return prob > threshold
    
    def is_occupied(self, x: float, y: float) -> bool:
        """Check if a world position is occupied."""
        gx, gy = self.world_to_grid(x, y)
        if not self.is_valid(gx, gy):
            return True  # Out of bounds = occupied
        
        with self.lock:
            prob = 1.0 - 1.0 / (1.0 + math.exp(self.grid[gy, gx]))
            return prob > 0.6
    
    def is_free(self, x: float, y: float) -> bool:
        """Check if a world position is free."""
        gx, gy = self.world_to_grid(x, y)
        if not self.is_valid(gx, gy):
            return False
        
        with self.lock:
            prob = 1.0 - 1.0 / (1.0 + math.exp(self.grid[gy, gx]))
            return prob < 0.4
    
    def save(self, filename: str):
        """Save map to file."""
        with self.lock:
            data = {
                'width': self.width,
                'height': self.height,
                'resolution': self.resolution,
                'origin_x': self.origin_x,
                'origin_y': self.origin_y,
                'grid': self.grid.tolist()
            }
            with open(filename, 'w') as f:
                json.dump(data, f)
        print(f"Map saved to {filename}")
    
    def load(self, filename: str):
        """Load map from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        with self.lock:
            self.width = data['width']
            self.height = data['height']
            self.resolution = data['resolution']
            self.origin_x = data['origin_x']
            self.origin_y = data['origin_y']
            self.grid = np.array(data['grid'], dtype=np.float32)
        print(f"Map loaded from {filename}")


# ============================================================================
# Path Planner (A* Algorithm)
# ============================================================================
class AStarPlanner:
    """
    A* path planning on occupancy grid.
    
    Features:
    - Efficient priority queue based search
    - Inflation of obstacles for safe paths
    - Path smoothing
    """
    
    def __init__(self, occupancy_map: OccupancyGridMap, 
                 robot_radius: float = ROBOT_RADIUS):
        self.map = occupancy_map
        self.robot_radius = robot_radius
        
        # Inflation radius in cells
        self.inflation_radius = int(math.ceil(robot_radius / occupancy_map.resolution)) + 1
        
        # 8-connected neighbors (including diagonals)
        self.neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
    
    def plan(self, start: Pose2D, goal: Pose2D) -> Optional[List[Pose2D]]:
        """
        Plan path from start to goal.
        
        Args:
            start: Start pose
            goal: Goal pose
            
        Returns:
            List of waypoints or None if no path found
        """
        # Convert to grid coordinates
        start_cell = self.map.world_to_grid(start.x, start.y)
        goal_cell = self.map.world_to_grid(goal.x, goal.y)
        
        if not self.map.is_valid(*start_cell) or not self.map.is_valid(*goal_cell):
            warn("Start or goal outside map bounds")
            return None
        
        # Get inflated cost map
        cost_map = self._get_cost_map()
        
        if cost_map[goal_cell[1], goal_cell[0]] > 250:
            warn("Goal is in obstacle or too close to obstacle")
            return None
        
        # A* search
        path_cells = self._astar(start_cell, goal_cell, cost_map)
        
        if path_cells is None:
            return None
        
        # Convert to world coordinates
        path = []
        for gx, gy in path_cells:
            x, y = self.map.grid_to_world(gx, gy)
            path.append(Pose2D(x=x, y=y, theta=0))
        
        # Smooth path
        path = self._smooth_path(path)
        
        # Add heading to waypoints
        for i in range(len(path) - 1):
            dx = path[i + 1].x - path[i].x
            dy = path[i + 1].y - path[i].y
            path[i].theta = math.atan2(dy, dx)
        
        if len(path) > 0:
            path[-1].theta = goal.theta
        
        return path
    
    def _get_cost_map(self) -> np.ndarray:
        """Create cost map with inflated obstacles."""
        binary_map = self.map.get_binary_map(threshold=0.5)
        cost_map = np.zeros_like(binary_map, dtype=np.float32)
        
        # Mark occupied cells
        cost_map[binary_map] = 255
        
        # Inflate obstacles
        if OPENCV_AVAILABLE:
            kernel_size = 2 * self.inflation_radius + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            cost_map = cv2.dilate(cost_map, kernel)
        else:
            # Simple dilation without OpenCV
            inflated = np.zeros_like(cost_map)
            for y in range(cost_map.shape[0]):
                for x in range(cost_map.shape[1]):
                    if cost_map[y, x] > 0:
                        for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                            for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < cost_map.shape[0] and 0 <= nx < cost_map.shape[1]:
                                    inflated[ny, nx] = max(inflated[ny, nx], 255)
            cost_map = inflated
        
        return cost_map
    
    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int], 
               cost_map: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """A* search algorithm."""
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        visited = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            
            if current in visited:
                continue
            visited.add(current)
            
            for dx, dy, cost in self.neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.map.is_valid(*neighbor):
                    continue
                
                if cost_map[neighbor[1], neighbor[0]] > 250:
                    continue
                
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        
        return None  # No path found
    
    def _smooth_path(self, path: List[Pose2D], iterations: int = 50) -> List[Pose2D]:
        """Smooth path using gradient descent."""
        if len(path) < 3:
            return path
        
        # Convert to numpy array
        coords = np.array([[p.x, p.y] for p in path])
        
        # Smoothing parameters
        weight_data = 0.5
        weight_smooth = 0.3
        tolerance = 0.00001
        
        new_coords = coords.copy()
        
        for _ in range(iterations):
            change = 0
            for i in range(1, len(coords) - 1):
                for j in range(2):
                    aux = new_coords[i, j]
                    new_coords[i, j] += weight_data * (coords[i, j] - new_coords[i, j])
                    new_coords[i, j] += weight_smooth * (
                        new_coords[i-1, j] + new_coords[i+1, j] - 2 * new_coords[i, j])
                    change += abs(aux - new_coords[i, j])
            
            if change < tolerance:
                break
        
        # Convert back to Pose2D
        return [Pose2D(x=c[0], y=c[1]) for c in new_coords]


# ============================================================================
# Obstacle Avoidance (Dynamic Window Approach)
# ============================================================================
class DynamicWindowApproach:
    """
    Local obstacle avoidance using Dynamic Window Approach (DWA).
    
    Samples velocity space and selects best velocity for trajectory.
    """
    
    def __init__(self, max_speed: float = MAX_SPEED, 
                 max_turn: float = MAX_TURN_RATE):
        self.max_speed = max_speed
        self.max_turn = max_turn
        
        # DWA parameters
        self.speed_resolution = 0.05  # m/s
        self.turn_resolution = 0.1  # rad/s
        self.predict_time = 2.0  # seconds
        self.dt = 0.1  # seconds
        
        # Cost weights
        self.heading_weight = 1.0
        self.clearance_weight = 2.0
        self.velocity_weight = 0.5
    
    def compute_velocity(self, pose: Pose2D, goal: Pose2D, 
                        obstacles: List[Tuple[float, float]], 
                        current_v: float, current_omega: float) -> Tuple[float, float]:
        """
        Compute best velocity command to reach goal while avoiding obstacles.
        
        Args:
            pose: Current robot pose
            goal: Goal position
            obstacles: List of obstacle positions (x, y)
            current_v: Current linear velocity
            current_omega: Current angular velocity
            
        Returns:
            (linear_velocity, angular_velocity)
        """
        # Generate velocity samples
        min_v = max(0, current_v - 0.5)
        max_v = min(self.max_speed, current_v + 0.5)
        min_omega = max(-self.max_turn, current_omega - 1.0)
        max_omega = min(self.max_turn, current_omega + 1.0)
        
        best_v, best_omega = 0, 0
        best_cost = float('-inf')
        
        v = min_v
        while v <= max_v:
            omega = min_omega
            while omega <= max_omega:
                # Simulate trajectory
                trajectory = self._simulate_trajectory(pose, v, omega)
                
                # Check for collision
                min_dist = self._check_collision(trajectory, obstacles)
                
                if min_dist > ROBOT_RADIUS:
                    # Calculate cost
                    heading_cost = self._heading_cost(trajectory[-1], goal)
                    clearance_cost = min_dist
                    velocity_cost = v
                    
                    total_cost = (self.heading_weight * heading_cost +
                                 self.clearance_weight * clearance_cost +
                                 self.velocity_weight * velocity_cost)
                    
                    if total_cost > best_cost:
                        best_cost = total_cost
                        best_v, best_omega = v, omega
                
                omega += self.turn_resolution
            v += self.speed_resolution
        
        return best_v, best_omega
    
    def _simulate_trajectory(self, pose: Pose2D, v: float, 
                            omega: float) -> List[Pose2D]:
        """Simulate robot trajectory with given velocities."""
        trajectory = [pose]
        current = Pose2D(x=pose.x, y=pose.y, theta=pose.theta)
        
        t = 0
        while t < self.predict_time:
            if abs(omega) > 1e-5:
                current.x += -v/omega * math.sin(current.theta) + \
                            v/omega * math.sin(current.theta + omega * self.dt)
                current.y += v/omega * math.cos(current.theta) - \
                            v/omega * math.cos(current.theta + omega * self.dt)
            else:
                current.x += v * self.dt * math.cos(current.theta)
                current.y += v * self.dt * math.sin(current.theta)
            
            current.theta += omega * self.dt
            trajectory.append(Pose2D(x=current.x, y=current.y, theta=current.theta))
            t += self.dt
        
        return trajectory
    
    def _check_collision(self, trajectory: List[Pose2D], 
                        obstacles: List[Tuple[float, float]]) -> float:
        """Check minimum distance to obstacles along trajectory."""
        min_dist = float('inf')
        
        for pose in trajectory:
            for ox, oy in obstacles:
                dist = math.sqrt((pose.x - ox)**2 + (pose.y - oy)**2)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _heading_cost(self, pose: Pose2D, goal: Pose2D) -> float:
        """Calculate heading alignment with goal."""
        angle_to_goal = math.atan2(goal.y - pose.y, goal.x - pose.x)
        angle_diff = pose.theta - angle_to_goal
        
        # Normalize to [-pi, pi] first, then take absolute value
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        return math.pi - abs(angle_diff)


# ============================================================================
# ROS SLAM Interface (GMapping / Hector SLAM)
# ============================================================================
if ROS_AVAILABLE:
    class ROSSlamInterface:
        """
        Interface to ROS-based SLAM systems.
        
        Supports:
        - GMapping (grid-based SLAM)
        - Hector SLAM (high-frequency LIDAR)
        """
        
        def __init__(self, slam_type: str = 'gmapping'):
            self.slam_type = slam_type
            
            # Initialize ROS node
            rospy.init_node('picar_slam', anonymous=True)
            
            # Subscribers
            self.map_sub = rospy.Subscriber('/map', ROSOccupancyGrid, self._map_callback)
            self.odom_sub = rospy.Subscriber('/odom', Odometry, self._odom_callback)
            
            # Publishers
            self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
            self.scan_pub = rospy.Publisher('/scan', LaserScan, queue_size=10)
            
            # Data
            self.current_map = None
            self.current_pose = Pose2D()
            self.lock = threading.Lock()
            
            print(f"ROS SLAM interface initialized ({slam_type})")
        
        def _map_callback(self, msg):
            """Handle incoming map data."""
            with self.lock:
                self.current_map = msg
        
        def _odom_callback(self, msg):
            """Handle incoming odometry data."""
            with self.lock:
                self.current_pose.x = msg.pose.pose.position.x
                self.current_pose.y = msg.pose.pose.position.y
                
                # Extract yaw from quaternion
                orientation = msg.pose.pose.orientation
                _, _, yaw = euler_from_quaternion([
                    orientation.x, orientation.y, orientation.z, orientation.w
                ])
                self.current_pose.theta = yaw
        
        def publish_scan(self, scan: ScanData):
            """Publish scan data to ROS."""
            msg = LaserScan()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'laser_frame'
            
            msg.angle_min = -math.pi
            msg.angle_max = math.pi
            msg.angle_increment = 2 * math.pi / len(scan.readings)
            msg.range_min = LIDAR_MIN_RANGE
            msg.range_max = LIDAR_MAX_RANGE
            
            msg.ranges = [r.distance for r in scan.readings]
            msg.intensities = [r.intensity for r in scan.readings]
            
            self.scan_pub.publish(msg)
        
        def send_velocity(self, linear: float, angular: float):
            """Send velocity command via ROS."""
            msg = Twist()
            msg.linear.x = linear
            msg.angular.z = angular
            self.cmd_pub.publish(msg)
        
        def get_pose(self) -> Pose2D:
            """Get current pose from ROS."""
            with self.lock:
                return Pose2D(x=self.current_pose.x, y=self.current_pose.y, 
                             theta=self.current_pose.theta)
        
        def get_map(self) -> Optional[np.ndarray]:
            """Get current occupancy grid from ROS."""
            with self.lock:
                if self.current_map is None:
                    return None
                
                # Convert to numpy array
                width = self.current_map.info.width
                height = self.current_map.info.height
                data = np.array(self.current_map.data).reshape((height, width))
                
                return data
else:
    # Placeholder class when ROS is not available
    class ROSSlamInterface:
        """Placeholder for ROS SLAM interface when ROS is not installed."""
        def __init__(self, slam_type: str = 'gmapping'):
            raise RuntimeError("ROS is not available. Install ROS and rospy to use ROS-based SLAM.")


# ============================================================================
# RPLIDAR Interface
# ============================================================================
class RPLIDARInterface:
    """
    Interface to RPLIDAR sensors.
    
    Supports RPLIDAR A1, A2, A3.
    """
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        if not RPLIDAR_AVAILABLE:
            raise RuntimeError("rplidar library not available. Install with: pip install rplidar")
        
        self.port = port
        self.baudrate = baudrate
        self.lidar = None
        self.running = False
        self.latest_scan = None
        self.lock = threading.Lock()
        
        self._connect()
    
    def _connect(self):
        """Connect to LIDAR."""
        try:
            self.lidar = RPLidar(self.port, self.baudrate)
            info = self.lidar.get_info()
            print(f"RPLIDAR connected: {info}")
            health = self.lidar.get_health()
            print(f"Health: {health}")
        except Exception as e:
            error(f"Failed to connect to RPLIDAR: {e}")
            self.lidar = None
    
    def start_scanning(self):
        """Start continuous scanning in background thread."""
        if self.lidar is None:
            return
        
        self.running = True
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()
    
    def _scan_loop(self):
        """Continuous scanning loop."""
        try:
            for scan in self.lidar.iter_scans():
                if not self.running:
                    break
                
                readings = []
                for quality, angle, distance in scan:
                    if quality > 0 and LIDAR_MIN_RANGE * 1000 < distance < LIDAR_MAX_RANGE * 1000:
                        readings.append(LaserReading(
                            angle=math.radians(angle),
                            distance=distance / 1000.0,  # Convert to meters
                            intensity=quality
                        ))
                
                with self.lock:
                    self.latest_scan = ScanData(
                        timestamp=time.time(),
                        readings=readings,
                        pose=Pose2D()
                    )
                    
        except Exception as e:
            error(f"LIDAR scan error: {e}")
    
    def get_scan(self) -> Optional[ScanData]:
        """Get latest scan data."""
        with self.lock:
            return self.latest_scan
    
    def stop(self):
        """Stop scanning and disconnect."""
        self.running = False
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.disconnect()
            except:
                pass


# ============================================================================
# Visualization
# ============================================================================
class MapVisualizer:
    """
    Map and path visualization.
    
    Supports:
    - matplotlib for static images
    - pygame for real-time display
    """
    
    def __init__(self, occupancy_map: OccupancyGridMap, 
                 use_pygame: bool = True):
        self.map = occupancy_map
        self.use_pygame = use_pygame and PYGAME_AVAILABLE
        
        self.robot_pose = Pose2D()
        self.path = []
        self.landmarks = {}
        self.scan_points = []
        
        self.running = False
        self.window = None
        
        if self.use_pygame:
            self._init_pygame()
    
    def _init_pygame(self):
        """Initialize pygame window."""
        pygame.init()
        self.window = pygame.display.set_mode(VIZ_WINDOW_SIZE)
        pygame.display.set_caption(f"{ROBOT_NAME} - SLAM Navigation")
        
        self.font = pygame.font.Font(None, 24)
        
        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_UNKNOWN = (60, 60, 70)
        self.COLOR_FREE = (40, 50, 60)
        self.COLOR_OCCUPIED = (220, 80, 80)
        self.COLOR_ROBOT = (100, 200, 255)
        self.COLOR_PATH = (80, 200, 120)
        self.COLOR_LANDMARK = (255, 200, 80)
        self.COLOR_SCAN = (150, 150, 200)
    
    def update(self, pose: Pose2D, path: List[Pose2D] = None, 
               landmarks: Dict = None, scan_points: List = None):
        """Update visualization data."""
        self.robot_pose = pose
        if path is not None:
            self.path = path
        if landmarks is not None:
            self.landmarks = landmarks
        if scan_points is not None:
            self.scan_points = scan_points
    
    def render(self):
        """Render current state."""
        if self.use_pygame:
            self._render_pygame()
        elif MATPLOTLIB_AVAILABLE:
            self._render_matplotlib()
    
    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        gx, gy = self.map.world_to_grid(x, y)
        
        # Scale to window size
        scale_x = VIZ_WINDOW_SIZE[0] / self.map.width
        scale_y = VIZ_WINDOW_SIZE[1] / self.map.height
        
        screen_x = int(gx * scale_x)
        screen_y = int((self.map.height - gy) * scale_y)  # Flip Y
        
        return screen_x, screen_y
    
    def _render_pygame(self):
        """Render using pygame."""
        if self.window is None:
            return
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
        
        # Clear screen
        self.window.fill(self.COLOR_BG)
        
        # Draw map
        occupancy = self.map.get_occupancy_grid()
        scale_x = VIZ_WINDOW_SIZE[0] / self.map.width
        scale_y = VIZ_WINDOW_SIZE[1] / self.map.height
        
        for gy in range(0, self.map.height, 2):  # Skip every other cell for performance
            for gx in range(0, self.map.width, 2):
                value = occupancy[gy, gx]
                
                if value < 0:
                    color = self.COLOR_UNKNOWN
                elif value < 30:
                    color = self.COLOR_FREE
                else:
                    intensity = min(255, value * 2.5)
                    color = (intensity, 50, 50)
                
                rect = pygame.Rect(
                    int(gx * scale_x), 
                    int((self.map.height - gy) * scale_y),
                    int(scale_x * 2) + 1, 
                    int(scale_y * 2) + 1
                )
                pygame.draw.rect(self.window, color, rect)
        
        # Draw scan points
        for x, y in self.scan_points:
            sx, sy = self._world_to_screen(x, y)
            pygame.draw.circle(self.window, self.COLOR_SCAN, (sx, sy), 2)
        
        # Draw path
        if len(self.path) > 1:
            points = [self._world_to_screen(p.x, p.y) for p in self.path]
            pygame.draw.lines(self.window, self.COLOR_PATH, False, points, 2)
            
            for point in points:
                pygame.draw.circle(self.window, self.COLOR_PATH, point, 4)
        
        # Draw landmarks
        for lm_id, (lx, ly) in self.landmarks.items():
            sx, sy = self._world_to_screen(lx, ly)
            pygame.draw.circle(self.window, self.COLOR_LANDMARK, (sx, sy), 5)
        
        # Draw robot
        rx, ry = self._world_to_screen(self.robot_pose.x, self.robot_pose.y)
        pygame.draw.circle(self.window, self.COLOR_ROBOT, (rx, ry), 10)
        
        # Draw robot heading
        heading_len = 20
        hx = rx + int(heading_len * math.cos(-self.robot_pose.theta + math.pi/2))
        hy = ry + int(heading_len * math.sin(-self.robot_pose.theta + math.pi/2))
        pygame.draw.line(self.window, (255, 255, 255), (rx, ry), (hx, hy), 3)
        
        # Draw info text
        info_text = f"Pose: ({self.robot_pose.x:.2f}, {self.robot_pose.y:.2f}, {math.degrees(self.robot_pose.theta):.1f}°)"
        text_surface = self.font.render(info_text, True, (200, 200, 200))
        self.window.blit(text_surface, (10, 10))
        
        pygame.display.flip()
    
    def _render_matplotlib(self):
        """Render using matplotlib (for saving images)."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw map
        occupancy = self.map.get_occupancy_grid()
        
        # Custom colormap
        cmap = LinearSegmentedColormap.from_list('slam', 
            [(0.2, 0.2, 0.25), (0.3, 0.4, 0.5), (0.8, 0.3, 0.3)])
        
        ax.imshow(occupancy, cmap=cmap, origin='lower', 
                  extent=[self.map.origin_x, 
                         self.map.origin_x + self.map.width * self.map.resolution,
                         self.map.origin_y, 
                         self.map.origin_y + self.map.height * self.map.resolution])
        
        # Draw path
        if self.path:
            path_x = [p.x for p in self.path]
            path_y = [p.y for p in self.path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path')
        
        # Draw landmarks
        for lm_id, (lx, ly) in self.landmarks.items():
            ax.plot(lx, ly, 'yo', markersize=8)
        
        # Draw robot
        ax.plot(self.robot_pose.x, self.robot_pose.y, 'co', markersize=12)
        ax.arrow(self.robot_pose.x, self.robot_pose.y,
                0.3 * math.cos(self.robot_pose.theta),
                0.3 * math.sin(self.robot_pose.theta),
                head_width=0.1, color='cyan')
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'{ROBOT_NAME} SLAM Map')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('slam_map.png', dpi=150)
        plt.close()
    
    def close(self):
        """Close visualization."""
        if self.use_pygame and pygame.get_init():
            pygame.quit()


# ============================================================================
# Main Navigation Controller
# ============================================================================
class SLAMNavigationController:
    """
    Main SLAM navigation controller integrating all components.
    
    Coordinates:
    - SLAM (EKF or ROS-based)
    - Mapping
    - Path planning
    - Obstacle avoidance
    - Visualization
    """
    
    def __init__(self, use_ros: bool = False, use_lidar: bool = False,
                 lidar_port: str = '/dev/ttyUSB0'):
        self.use_ros = use_ros and ROS_AVAILABLE
        self.use_lidar = use_lidar and RPLIDAR_AVAILABLE
        
        # Mode
        self.mode = NavigationMode.IDLE
        
        # Initialize components based on configuration
        if self.use_ros:
            self.slam = ROSSlamInterface()
            self.occupancy_map = None  # Will get from ROS
        else:
            self.slam = EKFSLAM()
            self.occupancy_map = OccupancyGridMap()
        
        # LIDAR
        self.lidar = None
        if self.use_lidar:
            try:
                self.lidar = RPLIDARInterface(port=lidar_port)
                self.lidar.start_scanning()
            except Exception as e:
                warn(f"LIDAR initialization failed: {e}")
        
        # Path planner
        if self.occupancy_map:
            self.planner = AStarPlanner(self.occupancy_map)
        else:
            self.planner = None
        
        # Obstacle avoidance
        self.dwa = DynamicWindowApproach()
        
        # Visualization
        self.visualizer = None
        if self.occupancy_map:
            try:
                self.visualizer = MapVisualizer(self.occupancy_map, 
                                               use_pygame=PYGAME_AVAILABLE)
            except Exception as e:
                warn(f"Visualization initialization failed: {e}")
        
        # PiCar-X
        self.car = None
        if PICARX_AVAILABLE:
            try:
                self.car = Picarx()
                time.sleep(0.5)
                self.car.reset()
                print("PiCar-X initialized")
            except Exception as e:
                warn(f"PiCar-X initialization failed: {e}")
        
        # Camera for visual odometry
        self.camera = None
        if OPENCV_AVAILABLE and not use_lidar:
            try:
                from vilib import Vilib
                Vilib.camera_start(vflip=False, hflip=False)
                self.camera = Vilib
                print("Camera initialized for visual odometry")
            except Exception as e:
                warn(f"Camera initialization failed: {e}")
        
        # State
        self.current_pose = Pose2D()
        self.goal_pose = None
        self.current_path = []
        self.patrol_waypoints = []
        self.patrol_index = 0
        
        # Velocity state
        self.current_v = 0.0
        self.current_omega = 0.0
        
        # Control
        self.running = False
        self.lock = threading.Lock()
        
        # Odometry
        self.last_time = time.time()
        self.left_encoder = 0
        self.right_encoder = 0
        
        print(f"\n{'='*60}")
        print(f"  {ROBOT_NAME} SLAM Navigation System")
        print(f"  ROS Mode: {self.use_ros}")
        print(f"  LIDAR: {self.use_lidar}")
        print(f"  Visualization: {self.visualizer is not None}")
        print(f"{'='*60}\n")
    
    def start(self):
        """Start navigation system."""
        self.running = True
        
        # Start main control loop
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        # Start visualization loop
        if self.visualizer:
            self.viz_thread = threading.Thread(target=self._visualization_loop, daemon=True)
            self.viz_thread.start()
    
    def stop(self):
        """Stop navigation system."""
        self.running = False
        self.mode = NavigationMode.IDLE
        
        if self.car:
            self.car.stop()
            self.car.reset()
        
        if self.lidar:
            self.lidar.stop()
        
        if self.visualizer:
            self.visualizer.close()
    
    def set_mode(self, mode: NavigationMode):
        """Set navigation mode."""
        with self.lock:
            self.mode = mode
            print(f"Navigation mode: {mode.value}")
    
    def go_to(self, x: float, y: float, theta: float = 0):
        """Navigate to goal position."""
        self.goal_pose = Pose2D(x=x, y=y, theta=theta)
        
        # Plan path
        if self.planner:
            path = self.planner.plan(self.current_pose, self.goal_pose)
            if path:
                self.current_path = path
                self.set_mode(NavigationMode.GOTO)
                print(f"Path planned: {len(path)} waypoints")
            else:
                warn("Failed to plan path to goal")
        else:
            # Direct navigation without path planning
            self.current_path = [self.current_pose, self.goal_pose]
            self.set_mode(NavigationMode.GOTO)
    
    def start_patrol(self, waypoints: List[Tuple[float, float]]):
        """Start autonomous patrol through waypoints."""
        self.patrol_waypoints = [Pose2D(x=x, y=y) for x, y in waypoints]
        self.patrol_index = 0
        self.set_mode(NavigationMode.PATROL)
        print(f"Starting patrol with {len(waypoints)} waypoints")
    
    def start_mapping(self):
        """Start mapping mode."""
        self.set_mode(NavigationMode.MAPPING)
        print("Mapping mode started - drive the robot to explore")
    
    def start_exploration(self):
        """Start autonomous exploration."""
        self.set_mode(NavigationMode.EXPLORE)
        print("Exploration mode started")
    
    def save_map(self, filename: str = "map.json"):
        """Save current map."""
        if self.occupancy_map:
            self.occupancy_map.save(filename)
    
    def load_map(self, filename: str = "map.json"):
        """Load map from file."""
        if self.occupancy_map:
            self.occupancy_map.load(filename)
            # Reinitialize planner with loaded map
            self.planner = AStarPlanner(self.occupancy_map)
    
    def _control_loop(self):
        """Main control loop."""
        rate = 10  # Hz
        dt = 1.0 / rate
        
        while self.running:
            try:
                # Get sensor data
                scan = self._get_scan()
                
                # Update odometry
                self._update_odometry(dt)
                
                # Update SLAM
                if scan and not self.use_ros:
                    # Update EKF with wheel odometry
                    self.slam.predict(self.current_v, self.current_omega, dt)
                    
                    # Update map
                    if self.occupancy_map:
                        scan.pose = self.current_pose
                        self.occupancy_map.update_with_scan(self.current_pose, scan)
                
                # Get current pose from SLAM (works for both ROS and EKF modes)
                self.current_pose = self.slam.get_pose()
                
                # Execute mode-specific behavior
                if self.mode == NavigationMode.MAPPING:
                    self._mapping_step(scan)
                elif self.mode == NavigationMode.GOTO:
                    self._goto_step(scan)
                elif self.mode == NavigationMode.PATROL:
                    self._patrol_step(scan)
                elif self.mode == NavigationMode.EXPLORE:
                    self._explore_step(scan)
                elif self.mode == NavigationMode.IDLE:
                    self._send_velocity(0, 0)
                
            except Exception as e:
                error(f"Control loop error: {e}")
            
            time.sleep(dt)
    
    def _get_scan(self) -> Optional[ScanData]:
        """Get current scan data from sensors."""
        readings = []
        
        if self.use_lidar and self.lidar:
            scan = self.lidar.get_scan()
            if scan:
                return scan
        
        # Fall back to ultrasonic
        if self.car:
            distance = self.car.get_distance()
            if distance > 0:
                # Single forward reading
                readings.append(LaserReading(
                    angle=0,
                    distance=distance / 100.0  # Convert cm to meters
                ))
        
        if readings:
            return ScanData(timestamp=time.time(), readings=readings, pose=self.current_pose)
        
        return None
    
    def _update_odometry(self, dt: float):
        """Update odometry from wheel encoders or visual odometry."""
        if not self.use_ros:
            # Simple velocity integration
            if self.car:
                # Estimate velocity from motor commands
                # In a real implementation, use wheel encoders
                pass
    
    def _mapping_step(self, scan: Optional[ScanData]):
        """Mapping mode - just update map, no autonomous motion."""
        pass  # Map update happens in control loop
    
    def _goto_step(self, scan: Optional[ScanData]):
        """Navigate to goal."""
        if not self.current_path:
            self.set_mode(NavigationMode.IDLE)
            return
        
        # Get current waypoint
        waypoint = self.current_path[0]
        
        # Check if reached waypoint
        dist = math.sqrt((waypoint.x - self.current_pose.x)**2 + 
                        (waypoint.y - self.current_pose.y)**2)
        
        if dist < GOAL_TOLERANCE:
            self.current_path.pop(0)
            if not self.current_path:
                print("Reached goal!")
                self.set_mode(NavigationMode.IDLE)
                return
            waypoint = self.current_path[0]
        
        # Compute velocity command
        v, omega = self._compute_velocity_to_waypoint(waypoint, scan)
        self._send_velocity(v, omega)
    
    def _patrol_step(self, scan: Optional[ScanData]):
        """Patrol through waypoints."""
        if not self.patrol_waypoints:
            self.set_mode(NavigationMode.IDLE)
            return
        
        # Get current patrol waypoint
        waypoint = self.patrol_waypoints[self.patrol_index]
        
        # Check if reached waypoint
        dist = math.sqrt((waypoint.x - self.current_pose.x)**2 + 
                        (waypoint.y - self.current_pose.y)**2)
        
        if dist < PATROL_WAYPOINT_RADIUS:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_waypoints)
            print(f"Reached waypoint {self.patrol_index}, continuing patrol...")
            waypoint = self.patrol_waypoints[self.patrol_index]
        
        # Navigate to waypoint
        v, omega = self._compute_velocity_to_waypoint(waypoint, scan)
        self._send_velocity(v, omega)
    
    def _explore_step(self, scan: Optional[ScanData]):
        """Autonomous exploration to map unknown areas."""
        if not self.occupancy_map:
            return
        
        # Check if map is mostly unknown (needs initial exploration)
        occupancy = self.occupancy_map.get_occupancy_grid()
        unknown_count = np.sum(occupancy < 0)
        total_cells = occupancy.size
        unknown_ratio = unknown_count / total_cells
        
        # Find nearest frontier (boundary between known and unknown)
        frontier = self._find_frontier()
        
        if frontier:
            # Plan path to frontier
            path = self.planner.plan(self.current_pose, frontier)
            if path:
                self.current_path = path
                self._goto_step(scan)
            else:
                # Can't plan to frontier, do random exploration
                self._random_exploration(scan)
        elif unknown_ratio > 0.95:
            # Map is mostly unknown - need to do initial random exploration
            # to build up some known areas before frontier-based exploration works
            gray_print(f"Map {unknown_ratio*100:.0f}% unknown - random exploration")
            self._random_exploration(scan)
        else:
            print("Exploration complete - no more frontiers")
            self.set_mode(NavigationMode.IDLE)
    
    def _find_frontier(self) -> Optional[Pose2D]:
        """Find nearest frontier cell for exploration."""
        occupancy = self.occupancy_map.get_occupancy_grid()
        
        # Find unknown cells adjacent to free cells
        frontiers = []
        
        for y in range(1, self.occupancy_map.height - 1):
            for x in range(1, self.occupancy_map.width - 1):
                if occupancy[y, x] < 0:  # Unknown
                    # Check if adjacent to free space
                    is_frontier = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if occupancy[y + dy, x + dx] < 30 and occupancy[y + dy, x + dx] >= 0:
                                is_frontier = True
                                break
                        if is_frontier:
                            break
                    if is_frontier:
                        wx, wy = self.occupancy_map.grid_to_world(x, y)
                        frontiers.append((wx, wy))
        
        if not frontiers:
            return None
        
        # Find nearest frontier
        min_dist = float('inf')
        nearest = None
        
        for fx, fy in frontiers:
            dist = math.sqrt((fx - self.current_pose.x)**2 + (fy - self.current_pose.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = Pose2D(x=fx, y=fy)
        
        return nearest
    
    def _random_exploration(self, scan: Optional[ScanData]):
        """Random walk for exploration."""
        # Get obstacles from scan
        obstacles = []
        if scan:
            for reading in scan.readings:
                if reading.distance < 1.0:
                    angle = self.current_pose.theta + reading.angle
                    ox = self.current_pose.x + reading.distance * math.cos(angle)
                    oy = self.current_pose.y + reading.distance * math.sin(angle)
                    obstacles.append((ox, oy))
        
        # Use DWA for random goal
        import random
        random_angle = random.uniform(-math.pi, math.pi)
        random_goal = Pose2D(
            x=self.current_pose.x + 2.0 * math.cos(random_angle),
            y=self.current_pose.y + 2.0 * math.sin(random_angle)
        )
        
        v, omega = self.dwa.compute_velocity(
            self.current_pose, random_goal, obstacles,
            self.current_v, self.current_omega
        )
        
        self._send_velocity(v, omega)
    
    def _compute_velocity_to_waypoint(self, waypoint: Pose2D, 
                                      scan: Optional[ScanData]) -> Tuple[float, float]:
        """Compute velocity command to reach waypoint while avoiding obstacles."""
        # Get obstacles from scan
        obstacles = []
        if scan:
            for reading in scan.readings:
                if reading.distance < 2.0:  # Only consider nearby obstacles
                    angle = self.current_pose.theta + reading.angle
                    ox = self.current_pose.x + reading.distance * math.cos(angle)
                    oy = self.current_pose.y + reading.distance * math.sin(angle)
                    obstacles.append((ox, oy))
        
        # Use DWA for local planning
        v, omega = self.dwa.compute_velocity(
            self.current_pose, waypoint, obstacles,
            self.current_v, self.current_omega
        )
        
        self.current_v = v
        self.current_omega = omega
        
        return v, omega
    
    def _send_velocity(self, v: float, omega: float):
        """Send velocity command to robot."""
        if self.use_ros:
            self.slam.send_velocity(v, omega)
        elif self.car:
            # Convert to PiCar-X commands
            # Simple differential drive conversion
            speed = int(v * 100 / MAX_SPEED)  # 0-100
            speed = max(-100, min(100, speed))
            
            # Convert angular velocity to steering angle
            # omega = v * tan(steering) / wheelbase
            if abs(v) > 0.01:
                steering_rad = math.atan(omega * WHEEL_BASE / max(abs(v), 0.1))
                steering_deg = math.degrees(steering_rad)
                steering_deg = max(-35, min(35, steering_deg))
            else:
                steering_deg = 0
            
            self.car.set_dir_servo_angle(steering_deg)
            
            if speed >= 0:
                self.car.forward(speed)
            else:
                self.car.backward(-speed)
    
    def _visualization_loop(self):
        """Visualization update loop."""
        while self.running:
            try:
                # Prepare visualization data
                scan_points = []
                if self.lidar:
                    scan = self.lidar.get_scan()
                    if scan:
                        for reading in scan.readings:
                            angle = self.current_pose.theta + reading.angle
                            x = self.current_pose.x + reading.distance * math.cos(angle)
                            y = self.current_pose.y + reading.distance * math.sin(angle)
                            scan_points.append((x, y))
                
                landmarks = {}
                if not self.use_ros:
                    landmarks = self.slam.get_landmarks()
                
                # Update visualizer
                self.visualizer.update(
                    pose=self.current_pose,
                    path=self.current_path,
                    landmarks=landmarks,
                    scan_points=scan_points
                )
                
                # Render
                self.visualizer.render()
                
            except Exception as e:
                pass  # Ignore visualization errors
            
            time.sleep(1.0 / VIZ_UPDATE_RATE)


# ============================================================================
# Keyboard Control
# ============================================================================
def keyboard_control(controller: SLAMNavigationController):
    """Keyboard control for testing."""
    import readline
    
    print("\n" + "="*60)
    print("SLAM Navigation - Keyboard Control")
    print("="*60)
    print("\nCommands:")
    print("  map       - Start mapping mode")
    print("  explore   - Start autonomous exploration")
    print("  go X Y    - Navigate to position (meters)")
    print("  patrol    - Start patrol with predefined waypoints")
    print("  stop      - Stop and idle")
    print("  save      - Save map to file")
    print("  load      - Load map from file")
    print("  status    - Show current status")
    print("  quit      - Exit")
    print()
    
    # Predefined patrol waypoints (in meters)
    patrol_points = [
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
    ]
    
    while controller.running:
        try:
            cmd = input(f"\033[1;36m[{controller.mode.value}] >\033[0m ").strip().lower()
            
            if not cmd:
                continue
            
            parts = cmd.split()
            command = parts[0]
            
            if command == 'quit' or command == 'exit':
                break
            
            elif command == 'map':
                controller.start_mapping()
            
            elif command == 'explore':
                controller.start_exploration()
            
            elif command == 'go' and len(parts) >= 3:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    theta = float(parts[3]) if len(parts) > 3 else 0
                    controller.go_to(x, y, theta)
                except ValueError:
                    print("Usage: go X Y [theta]")
            
            elif command == 'patrol':
                controller.start_patrol(patrol_points)
            
            elif command == 'stop':
                controller.set_mode(NavigationMode.IDLE)
            
            elif command == 'save':
                filename = parts[1] if len(parts) > 1 else "slam_map.json"
                controller.save_map(filename)
            
            elif command == 'load':
                filename = parts[1] if len(parts) > 1 else "slam_map.json"
                controller.load_map(filename)
            
            elif command == 'status':
                pose = controller.current_pose
                print(f"\nStatus:")
                print(f"  Mode: {controller.mode.value}")
                print(f"  Position: ({pose.x:.2f}, {pose.y:.2f})")
                print(f"  Heading: {math.degrees(pose.theta):.1f}°")
                print(f"  Path waypoints: {len(controller.current_path)}")
                if controller.occupancy_map:
                    print(f"  Map size: {controller.occupancy_map.width}x{controller.occupancy_map.height}")
                print()
            
            else:
                print(f"Unknown command: {command}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PiCar-X SLAM Navigation')
    parser.add_argument('--ros', action='store_true', help='Use ROS for SLAM')
    parser.add_argument('--lidar', action='store_true', help='Use RPLIDAR')
    parser.add_argument('--lidar-port', default='/dev/ttyUSB0', help='LIDAR serial port')
    parser.add_argument('--keyboard', action='store_true', help='Enable keyboard control')
    parser.add_argument('--map', type=str, help='Load map from file')
    args = parser.parse_args()
    
    # Check requirements
    if args.ros and not ROS_AVAILABLE:
        error("ROS requested but not available. Install ROS and rospy.")
        sys.exit(1)
    
    if args.lidar and not RPLIDAR_AVAILABLE:
        error("LIDAR requested but rplidar library not available.")
        error("Install with: pip install rplidar")
        sys.exit(1)
    
    # Create controller
    controller = SLAMNavigationController(
        use_ros=args.ros,
        use_lidar=args.lidar,
        lidar_port=args.lidar_port
    )
    
    # Load map if specified
    if args.map:
        controller.load_map(args.map)
    
    try:
        # Start controller
        controller.start()
        
        # Run keyboard control or wait
        if args.keyboard:
            keyboard_control(controller)
        else:
            # Default: start exploration
            print("Starting autonomous exploration...")
            controller.start_exploration()
            
            # Wait for interrupt
            while controller.running:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        controller.stop()
        print("SLAM Navigation shutdown complete.")


if __name__ == "__main__":
    import warnings
    import atexit
    
    # Suppress GPIO cleanup warnings that occur during Python shutdown
    def suppress_gpio_warnings():
        warnings.filterwarnings('ignore', message='.*GPIO.*')
    
    atexit.register(suppress_gpio_warnings)
    
    # Suppress lgpio errors during exit
    import sys
    _original_excepthook = sys.excepthook
    
    def _quiet_excepthook(exc_type, exc_value, exc_tb):
        # Suppress lgpio.error during cleanup
        if 'lgpio' in str(exc_type) or 'GPIO' in str(exc_value):
            return
        _original_excepthook(exc_type, exc_value, exc_tb)
    
    sys.excepthook = _quiet_excepthook
    
    main()

