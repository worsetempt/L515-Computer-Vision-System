import pyrealsense2 as rs
import numpy as np
from scipy.spatial import cKDTree
import threading
import time

class RealObstacleDetectionSystem:
    def __init__(self, distance_threshold=0.5):
        self.distance_threshold = distance_threshold
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        self.obstacle_tracker = ObstacleTracker()

    def get_point_cloud(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        return np.column_stack((vtx['f0'], vtx['f1'], vtx['f2']))

    def detect_obstacles(self):
        point_cloud = self.get_point_cloud()
        obstacles = self.cluster_points(point_cloud)
        tracked_obstacles = self.obstacle_tracker.update(obstacles)
        return tracked_obstacles

    def cluster_points(self, points):
        tree = cKDTree(points)
        clusters = []
        processed = set()

        for i, point in enumerate(points):
            if i in processed:
                continue
            neighbors = tree.query_ball_point(point, self.distance_threshold)
            if len(neighbors) > 10:  # Minimum cluster size
                cluster = points[neighbors]
                center = np.mean(cluster, axis=0)
                size = np.max(cluster, axis=0) - np.min(cluster, axis=0)
                clusters.append((center, size))
                processed.update(neighbors)

        return clusters

class ObstacleTracker:
    def __init__(self):
        self.obstacles = []
        self.id_counter = 0

    def update(self, detected_obstacles):
        if not self.obstacles:
            for obs in detected_obstacles:
                self.obstacles.append(Obstacle(self.id_counter, obs[0], obs[1]))
                self.id_counter += 1
        else:
            matched_obstacles = self.match_obstacles(detected_obstacles)
            self.update_tracked_obstacles(matched_obstacles)
            self.add_new_obstacles(matched_obstacles, detected_obstacles)

        return self.obstacles

    def match_obstacles(self, detected_obstacles):
        matched = []
        for tracked in self.obstacles:
            best_match = None
            best_distance = float('inf')
            for i, detected in enumerate(detected_obstacles):
                distance = np.linalg.norm(tracked.position - detected[0])
                if distance < best_distance:
                    best_distance = distance
                    best_match = (i, detected)
            if best_match and best_distance < 0.5:  # Adjust threshold as needed
                matched.append((tracked, best_match[0], best_match[1]))
        return matched

    def update_tracked_obstacles(self, matched_obstacles):
        for tracked, _, detected in matched_obstacles:
            tracked.update(detected[0], detected[1])

    def add_new_obstacles(self, matched_obstacles, detected_obstacles):
        matched_indices = set(match[1] for match in matched_obstacles)
        for i, detected in enumerate(detected_obstacles):
            if i not in matched_indices:
                self.obstacles.append(Obstacle(self.id_counter, detected[0], detected[1]))
                self.id_counter += 1

class Obstacle:
    def __init__(self, id, position, size):
        self.id = id
        self.position = position
        self.size = size
        self.velocity = np.zeros(3)

    def update(self, new_position, new_size):
        self.velocity = new_position - self.position
        self.position = new_position
        self.size = new_size

class DynamicNavigationSystem:
    def __init__(self, resolution=0.1):
        self.pcd = None
        self.occupancy_grid = None
        self.min_bound = None
        self.resolution = resolution
        self.lock = threading.Lock()
        self.obstacle_system = RealObstacleDetectionSystem()

    def update_point_cloud(self, new_pcd):
        with self.lock:
            self.pcd = self.process_point_cloud(new_pcd)
            self.occupancy_grid, self.min_bound, _ = self.create_occupancy_grid(self.pcd)

    def process_point_cloud(self, pcd):
        pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
        cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd_down.select_by_index(ind)

    def create_occupancy_grid(self, pcd):
        points = np.asarray(pcd.points)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        
        grid_size = np.ceil((max_bound - min_bound) / self.resolution).astype(int)
        occupancy_grid = np.zeros(grid_size[:2], dtype=bool)
        
        for point in points:
            idx = np.floor((point[:2] - min_bound[:2]) / self.resolution).astype(int)
            occupancy_grid[idx[0], idx[1]] = True
        
        return occupancy_grid, min_bound, self.resolution

    def is_valid(self, point):
        with self.lock:
            idx = np.floor((point - self.min_bound[:2]) / self.resolution).astype(int)
            if not (0 <= idx[0] < self.occupancy_grid.shape[0] and 0 <= idx[1] < self.occupancy_grid.shape[1]):
                return False
            if self.occupancy_grid[idx[0], idx[1]]:
                return False
            obstacles = self.obstacle_system.detect_obstacles()
            for obstacle in obstacles:
                if np.linalg.norm(point - obstacle.position[:2]) < np.max(obstacle.size[:2]) / 2:
                    return False
            return True

    def plan_path(self, start, goal):
        path = [start]
        current = start
        while np.linalg.norm(current - goal) > self.resolution:
            direction = (goal - current) / np.linalg.norm(goal - current)
            next_point = current + direction * self.resolution
            if self.is_valid(next_point):
                path.append(next_point)
                current = next_point
            else:
                perpendicular = np.array([-direction[1], direction[0]])
                next_point = current + perpendicular * self.resolution
                if self.is_valid(next_point):
                    path.append(next_point)
                    current = next_point
                else:
                    next_point = current - perpendicular * self.resolution
                    if self.is_valid(next_point):
                        path.append(next_point)
                        current = next_point
                    else:
                        print("Cannot find a valid path")
                        break
        return np.array(path)

def visualize_path(nav_system, path):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(nav_system.occupancy_grid.T, origin='lower', cmap='binary')
    path_pixels = np.floor((path - nav_system.min_bound[:2]) / nav_system.resolution).astype(int)
    plt.plot(path_pixels[:, 0], path_pixels[:, 1], 'r-')
    obstacles = nav_system.obstacle_system.detect_obstacles()
    for obstacle in obstacles:
        pos = (obstacle.position[:2] - nav_system.min_bound[:2]) / nav_system.resolution
        size = obstacle.size[:2] / nav_system.resolution
        rect = plt.Rectangle(pos - size/2, size[0], size[1], fill=False, edgecolor='b')
        plt.gca().add_patch(rect)
    plt.title("Navigation Path")
    plt.draw()
    plt.pause(0.001)

# Main execution
if __name__ == "__main__":
    import open3d as o3d
    import matplotlib.pyplot as plt

    nav_system = DynamicNavigationSystem()

    # Load initial point cloud
    initial_pcd = o3d.io.read_point_cloud("pointcloud.ply")
    nav_system.update_point_cloud(initial_pcd)

    plt.ion()  # Turn on interactive mode for real-time plotting

    while True:
        start = np.array([0, 0])
        goal = np.array([4, 4])

        path = nav_system.plan_path(start, goal)
        visualize_path(nav_system, path)

        time.sleep(0.5)  # Replan every 0.5 seconds

        # In a real scenario, you would update the point cloud from the camera here
        # For this example, we're reusing the same point cloud
        nav_system.update_point_cloud(initial_pcd)