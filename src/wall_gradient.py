import pyrealsense2 as rs
import numpy as np
import cv2

def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    return pipeline

def process_frames(pipeline):
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Resize depth image to match color frame dimensions
            depth_resized = cv2.resize(depth_image, (1920, 1080), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Calculate vertical gradient on resized depth
            gradient_y = np.gradient(depth_resized, axis=0)
            
            # Create masks with proper dimensions
            walls_mask = (gradient_y < -5).astype(np.uint8) * 255
            floors_mask = (gradient_y > 5).astype(np.uint8) * 255
            
            # Convert to boolean masks
            walls_mask = walls_mask.astype(bool)
            floors_mask = floors_mask.astype(bool)
            
            # Initialize segmented image
            segmented_image = np.zeros_like(color_image)
            
            # Apply masks using correct dimensions
            segmented_image[walls_mask] = [255, 0, 0]  # Blue for walls
            segmented_image[floors_mask] = [0, 255, 0] # Green for floors
            
            # Combine with original image
            result = cv2.addWeighted(color_image, 0.7, segmented_image, 0.3, 0)
            
            cv2.imshow('Segmentation', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = initialize_camera()
    process_frames(pipeline)