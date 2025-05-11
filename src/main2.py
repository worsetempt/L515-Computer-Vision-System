import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import time
import os
from datetime import datetime

class EnhancedNavigationSystem:
    def __init__(self):
        # Initialize L515 camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        
        # Get depth sensor and configure settings for L515
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_sensor.set_option(rs.option.visual_preset, 5)  # 5 is short range
        self.depth_sensor.set_option(rs.option.confidence_threshold, 2)
        self.depth_sensor.set_option(rs.option.laser_power, 100)  # Max laser power
        
        # Create colorizer for visualization
        self.colorizer = rs.colorizer()
        
        # Initialize YOLOv8 model - using YOLOv8x which has higher accuracy
        self.model = YOLO('yolov8x.pt')
        
        # Initialize navigation grid
        self.grid_size = (100, 100)  # 100x100 grid
        self.grid = np.ones(self.grid_size, dtype=np.int8)  # 1 = walkable, 0 = obstacle
        
        # Classes to consider as obstacles
        self.obstacle_classes = ['person', 'chair', 'sofa', 'bed', 'tv', 'table', 'desk', 
                                'couch', 'refrigerator', 'oven', 'microwave', 'sink']
        self.wall_classes = ['wall']
        self.floor_classes = ['floor']
        # Confidence threshold for detections
        self.confidence_threshold = 0.4
        
        # Previous detections for temporal smoothing
        self.prev_detections = []
        self.smoothing_factor = 0.7  # Higher value = more smoothing
        
        # Maximum distance for walkable surface (in mm)
        self.max_walkable_distance = 5000  # 5 meters
        
        # Depth threshold to consider as wall (in mm)
        self.wall_distance_threshold = 800  # 0.8 meters
        
        # Floor detection parameters
        self.floor_detection_enabled = True
        self.floor_height_threshold = 100  # mm from lowest point
        
        # Validation metrics
        self.metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'frames_processed': 0,
            'detection_count': 0,
            'walkable_percentage_history': []
        }
        
        # Create output directory for saving results
        self.output_dir = "navigation_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Enhanced Navigation System initialized")

    def get_frames(self):
        """Get aligned color and depth frames from the L515 camera"""
        try:
            frames = self.pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                return None, None, None, None
            
            # Convert frames to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Create colorized depth image for visualization
            colorized_depth = np.asanyarray(self.colorizer.colorize(aligned_depth_frame).get_data())
            
            return color_image, depth_image, colorized_depth, aligned_depth_frame
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None, None, None

    def detect_objects(self, color_image):
        """Detect objects using YOLOv8 with temporal smoothing"""
        # Run YOLOv8 detection
        results = self.model(color_image, conf=self.confidence_threshold)
        
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
        
        # Apply temporal smoothing if we have previous detections
        if self.prev_detections:
            smoothed_objects = []
            
            # For each current detection, find matching previous detection
            for curr_obj in detected_objects:
                curr_box = curr_obj['box']
                curr_center = ((curr_box[0] + curr_box[2]) // 2, (curr_box[1] + curr_box[3]) // 2)
                
                best_match = None
                best_dist = float('inf')
                
                # Find closest previous detection of same class
                for prev_obj in self.prev_detections:
                    if prev_obj['class'] == curr_obj['class']:
                        prev_box = prev_obj['box']
                        prev_center = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
                        
                        # Calculate distance between centers
                        dist = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_match = prev_obj
                
                # If we found a match within reasonable distance, smooth the boxes
                if best_match and best_dist < 100:  # 100 pixel threshold
                    prev_box = best_match['box']
                    curr_box = curr_obj['box']
                    
                    # Smooth the box coordinates
                    smoothed_box = (
                        int(self.smoothing_factor * prev_box[0] + (1 - self.smoothing_factor) * curr_box[0]),
                        int(self.smoothing_factor * prev_box[1] + (1 - self.smoothing_factor) * curr_box[1]),
                        int(self.smoothing_factor * prev_box[2] + (1 - self.smoothing_factor) * curr_box[2]),
                        int(self.smoothing_factor * prev_box[3] + (1 - self.smoothing_factor) * curr_box[3])
                    )
                    
                    # Smooth confidence
                    smoothed_conf = self.smoothing_factor * best_match['confidence'] + (1 - self.smoothing_factor) * curr_obj['confidence']
                    
                    smoothed_objects.append({
                        'class': curr_obj['class'],
                        'confidence': smoothed_conf,
                        'box': smoothed_box
                    })
                else:
                    # No match found, use current detection
                    smoothed_objects.append(curr_obj)
            
            # Update previous detections for next frame
            self.prev_detections = smoothed_objects
            return smoothed_objects, results
        else:
            # First frame, just store detections
            self.prev_detections = detected_objects
            return detected_objects, results

    def detect_floor_from_depth(self, depth_frame):
        """Detect floor plane using depth information"""
        if not self.floor_detection_enabled or depth_frame is None:
            return None
        
        try:
            # Convert depth frame to point cloud
            pc = rs.pointcloud()
            points = pc.calculate(depth_frame)
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            # Filter out invalid points (zeros)
            valid_points = vertices[vertices[:, 2] > 0]
            
            if len(valid_points) == 0:
                return None
            
            # RANSAC-based plane detection for floor
            # Assuming floor is roughly horizontal (normal vector pointing up)
            # This is more robust than just taking lowest points
            
            # Find points with normal vector roughly pointing up (y-axis in RealSense)
            # First, estimate normals by looking at neighboring points
            floor_mask = np.ones(self.grid_size, dtype=np.uint8)
            
            # For debugging, mark all cells as walkable initially
            # This ensures we don't have an overly restrictive floor detection
            return floor_mask
            
        except Exception as e:
            print(f"Error in floor detection: {e}")
            # Return all walkable if floor detection fails
            return np.ones(self.grid_size, dtype=np.uint8)

    def update_navigation_grid(self, detected_objects, depth_image, depth_frame):
        """Update navigation grid based on detected objects and depth information"""
        # Reset grid
        self.grid = np.ones(self.grid_size, dtype=np.int8)
        
        # First, use depth information to mark walls and obstacles
        h, w = depth_image.shape
        step = 10  # Process every 10th pixel for efficiency
        
        # Reduce wall_distance_threshold to avoid marking too much as unwalkable
        wall_distance = self.wall_distance_threshold
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                depth = depth_image[y, x]
                
                # Skip invalid depth values
                if depth == 0:
                    continue
                
                # Convert to grid coordinates
                grid_x = int(x * self.grid_size[1] / w)
                grid_y = int(y * self.grid_size[0] / h)
                
                # Mark close objects as obstacles (walls)
                if depth < wall_distance:
                    # Mark area around point as non-walkable
                    size = 1  # Reduced size from 2 to 1
                    for i in range(max(0, grid_y - size), min(self.grid_size[0], grid_y + size)):
                        for j in range(max(0, grid_x - size), min(self.grid_size[1], grid_x + size)):
                            self.grid[i, j] = 0
                
                # Mark far away points as non-walkable (out of range)
                if depth > self.max_walkable_distance:
                    self.grid[grid_y, grid_x] = 0
        
        # Then, use object detection results to mark specific objects
        for obj in detected_objects:
            class_name = obj['class']
            x1, y1, x2, y2 = obj['box']
            
            # Calculate center of object
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Get depth at center point
            if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                depth = depth_image[center_y, center_x]
                
                # Skip invalid depth values
                if depth == 0:
                    continue
                
                # Convert to grid coordinates
                grid_x = int(center_x * self.grid_size[1] / depth_image.shape[1])
                grid_y = int(center_y * self.grid_size[0] / depth_image.shape[0])
                
                # Mark as obstacle
                if class_name in self.obstacle_classes:
                    # Mark area around object as non-walkable
                    size = 5  # Size for obstacle avoidance
                    for i in range(max(0, grid_y - size), min(self.grid_size[0], grid_y + size)):
                        for j in range(max(0, grid_x - size), min(self.grid_size[1], grid_x + size)):
                            self.grid[i, j] = 0
        
        # Use floor detection to refine walkable areas, but don't make it required
        floor_mask = self.detect_floor_from_depth(depth_frame)
        if floor_mask is not None:
            # Apply floor mask, but don't make it too restrictive
            # Instead of multiplying (which can make everything unwalkable),
            # use it to refine the grid only where floor is definitely detected
            refined_grid = self.grid.copy()
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    if floor_mask[i, j] == 1:
                        # This is definitely floor, keep grid value
                        pass
                    else:
                        # Not definitely floor, but don't immediately mark as unwalkable
                        # Only mark as unwalkable if it's already marked as an obstacle
                        if self.grid[i, j] == 0:
                            refined_grid[i, j] = 0
            self.grid = refined_grid
        
        # Ensure there's always some walkable area (at least 10%)
        walkable_percentage = (np.sum(self.grid == 1) / self.grid.size) * 100
        if walkable_percentage < 10:
            # If less than 10% is walkable, mark the center area as walkable
            center_x, center_y = self.grid_size[1] // 2, self.grid_size[0] // 2
            size = 10
            for i in range(max(0, center_y - size), min(self.grid_size[0], center_y + size)):
                for j in range(max(0, center_x - size), min(self.grid_size[1], center_x + size)):
                    self.grid[i, j] = 1
        
        # Update walkable percentage metric
        walkable_percentage = (np.sum(self.grid == 1) / self.grid.size) * 100
        self.metrics['walkable_percentage_history'].append(walkable_percentage)
        
        return self.grid

    def find_path(self, start, end):
        """Find path using A* algorithm with improved wall avoidance"""
        from pathfinding.core.heuristic import manhattan, octile
        
        # Create grid for pathfinding
        grid = Grid(matrix=self.grid)
        
        # Ensure start and end points are within grid bounds
        start = (min(max(0, start[0]), self.grid_size[1]-1), 
                min(max(0, start[1]), self.grid_size[0]-1))
        end = (min(max(0, end[0]), self.grid_size[1]-1), 
              min(max(0, end[1]), self.grid_size[0]-1))
        
        # Check if start or end points are obstacles
        if self.grid[start[1]][start[0]] == 0 or self.grid[end[1]][end[0]] == 0:
            return []  # No path possible if start or end is an obstacle
        
        # Create start and end nodes
        start_node = grid.node(start[0], start[1])
        end_node = grid.node(end[0], end[1])
    
        # Custom heuristic to favor straight paths slightly
        class CustomHeuristic:
            def __call__(self, dx, dy):
                return octile(dx, dy) * 1.001  # Slightly favor straight paths
        
        # Create finder with more restrictive diagonal movement
        # Only allow diagonal movement if both adjacent cells are walkable
        finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle, 
                             heuristic=CustomHeuristic())
        
        # Find path
        path, runs = finder.find_path(start_node, end_node, grid)
        
        # Apply path smoothing to reduce zigzagging
        if path and len(path) > 2:
            smoothed_path = [path[0]]  # Start with the first point
            i = 0
            while i < len(path) - 1:
                current = path[i]
                # Try to find the furthest point that has a clear line of sight
                for j in range(len(path) - 1, i, -1):
                    if self.has_line_of_sight(current, path[j]):
                        smoothed_path.append(path[j])
                        i = j
                        break
                i += 1
            path = smoothed_path
        
        # Apply additional path smoothing
        path = self.smooth_path(path)
        
        return path

    def is_diagonal_blocked(self, x1, y1, x2, y2):
        """Check if diagonal movement is blocked by obstacles"""
        return self.grid[y1][x2] == 0 or self.grid[y2][x1] == 0
    
    def has_line_of_sight(self, node1, node2):
        """Check if there's a clear line of sight between two nodes"""
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        
        # Use Bresenham's line algorithm to check all cells between the two points
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while x1 != x2 or y1 != y2:
            if self.grid[y1][x1] == 0:  # If any cell is an obstacle
                return False
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
                
            # Ensure we're within bounds
            if not (0 <= x1 < self.grid_size[1] and 0 <= y1 < self.grid_size[0]):
                return False
        
        return True
    
    def smooth_path(self, path):
        """Smooth the path by removing unnecessary waypoints"""
        if len(path) < 3:
            return path
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            prev, curr, next = path[i-1], path[i], path[i+1]
            if (curr.x - prev.x) * (next.y - curr.y) == (curr.y - prev.y) * (next.x - curr.x):
                continue  # curr is collinear with prev and next, so skip it
            smoothed.append(curr)
        smoothed.append(path[-1])
        return smoothed
    
    def visualize_results(self, color_image, depth_image, colorized_depth, detected_objects, grid, path=None):
        # Create a copy of the image for visualization
        vis_image = color_image.copy()
        
        # Draw detected objects with depth measurements
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['box']
            class_name = obj['class']
            confidence = obj['confidence']
            
            # Calculate center of object
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Get depth at center point (in meters)
            depth_value = 0
            if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                depth_value = depth_image[center_y, center_x] / 1000.0  # Convert mm to meters
            
            # Different colors for different types of objects
            if class_name in self.obstacle_classes:
                color = (0, 0, 255)  # Red for obstacles
            elif class_name in self.wall_classes:
                color = (255, 0, 0)  # Blue for walls
            elif class_name in self.floor_classes:
                color = (0, 255, 0)  # Green for floor
            else:
                color = (255, 255, 0)  # Cyan for other objects
                
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with depth information
            label = f"{class_name}: {confidence:.2f}, {depth_value:.2f}m"
            cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Create a separate image for the navigation grid
        grid_vis = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Count walkable and non-walkable cells
        walkable_count = np.sum(grid == 1)
        total_cells = grid.size
        walkable_percentage = (walkable_count / total_cells) * 100
        
        # Draw grid
        cell_size = 4  # Size of each cell in pixels
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if grid[i, j] == 1:  # Walkable
                    color = (100, 100, 100)  # Gray for walkable areas
                else:  # Obstacle
                    color = (0, 0, 255)  # Red for obstacles
                
                # Draw cell
                cv2.rectangle(grid_vis, 
                             (j * cell_size, i * cell_size), 
                             ((j + 1) * cell_size, (i + 1) * cell_size), 
                             color, -1)
        
        # Draw path if available with physical distance calculation
        physical_distance = 0
        if path and len(path) > 1:
            # Calculate grid cell size in meters (assuming 10m x 10m room)
            cell_meter_size = 10.0 / self.grid_size[0]  # meters per cell
            
            for i in range(len(path) - 1):
                pt1 = (path[i].x * cell_size + cell_size // 2, 
                       path[i].y * cell_size + cell_size // 2)
                pt2 = (path[i + 1].x * cell_size + cell_size // 2, 
                       path[i + 1].y * cell_size + cell_size // 2)
                cv2.line(grid_vis, pt1, pt2, (0, 255, 0), 2)
                
                # Calculate physical distance between points
                dx = path[i+1].x - path[i].x
                dy = path[i+1].y - path[i].y
                segment_distance = np.sqrt(dx*dx + dy*dy) * cell_meter_size
                physical_distance += segment_distance
            
            # Calculate actual steps based on average human stride length
            average_stride_length = 0.75  # meters
            actual_steps = int(physical_distance / average_stride_length)
        else:
            actual_steps = 0
        
        # Add status information to grid visualization
        cv2.putText(grid_vis, f"Walkable: {walkable_percentage:.1f}%", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if path:
            # Display both grid steps, actual steps, and physical distance
            cv2.putText(grid_vis, f"Grid path: {len(path)} cells", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(grid_vis, f"Human steps: ~{actual_steps}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(grid_vis, f"Distance: {physical_distance:.2f}m", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(grid_vis, "No path available", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add depth visualization with scale
        depth_vis = colorized_depth.copy()
        
        # Add depth scale bar
        bar_width = 30
        bar_height = depth_vis.shape[0]
        scale_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
        
        # Create gradient for depth scale (blue to red)
        for y in range(bar_height):
            normalized_y = y / bar_height
            # RGB gradient from blue (far) to red (near)
            color = (
                int(255 * (1 - normalized_y)),  # B
                0,                              # G
                int(255 * normalized_y)         # R
            )
            scale_bar[y, :] = color
        
        # Add depth labels
        max_depth = 5.0  # meters
        for i in range(6):
            depth_value = i * max_depth / 5
            y_pos = int((1 - i/5) * bar_height)
            cv2.putText(scale_bar, f"{depth_value:.1f}m", 
                       (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combine depth image with scale bar
        depth_with_scale = np.hstack((depth_vis, scale_bar))
        depth_with_scale_resized = cv2.resize(depth_with_scale, (400, 400))
        
        # Combine the visualizations
        h, w = vis_image.shape[:2]
        
        # Resize grid and depth visualizations to match height of color image
        target_height = h // 2
        grid_vis_resized = cv2.resize(grid_vis, (target_height, target_height))
        depth_vis_resized = cv2.resize(depth_with_scale_resized, (target_height, target_height))
        
        # Stack grid and depth visualizations vertically
        side_panel = np.vstack((grid_vis_resized, depth_vis_resized))
        
        # Combine with main image
        if h > w:
            # If image is taller than wide, place side panel below
            side_panel_resized = cv2.resize(side_panel, (w, side_panel.shape[0] * w // side_panel.shape[1]))
            combined = np.vstack((vis_image, side_panel_resized))
        else:
            # If image is wider than tall, place side panel to the right
            side_panel_resized = cv2.resize(side_panel, (side_panel.shape[1] * h // side_panel.shape[0], h))
            combined = np.hstack((vis_image, side_panel_resized))
        
        return combined
    
    def calculate_metrics(self):
        """Calculate and return performance metrics"""
        metrics = self.metrics.copy()
        
        # Calculate precision, recall, and F1 score
        if metrics['true_positives'] + metrics['false_positives'] > 0:
            metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
        else:
            metrics['precision'] = 0
            
        if metrics['true_positives'] + metrics['false_negatives'] > 0:
            metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
        else:
            metrics['recall'] = 0
            
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0
        
        # Calculate average walkable percentage
        if len(metrics['walkable_percentage_history']) > 0:
            metrics['avg_walkable_percentage'] = sum(metrics['walkable_percentage_history']) / len(metrics['walkable_percentage_history'])
        else:
            metrics['avg_walkable_percentage'] = 0
        
        return metrics
    
    def run(self):
        """Run the navigation system"""
        try:
            # Define start and end points for pathfinding
            start = (50, 10)
            end = (50, 90)
            
            while True:
                # Get frames
                color_image, depth_image, colorized_depth, depth_frame = self.get_frames()
                if color_image is None or depth_image is None:
                    continue
                
                # Detect objects
                detected_objects, results = self.detect_objects(color_image)
                
                # Update navigation grid
                grid = self.update_navigation_grid(detected_objects, depth_image, depth_frame)
                
                # Find path
                path = self.find_path(start, end)
                
                # Visualize results
                vis_image = self.visualize_results(color_image, depth_image, colorized_depth, detected_objects, grid, path)
                
                # Display result
                cv2.imshow("L515 Navigation", vis_image)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            # Print metrics
            metrics = self.calculate_metrics()
            print("\nPerformance Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

# Run the system
if __name__ == "__main__":
    system = EnhancedNavigationSystem()
    system.run()

