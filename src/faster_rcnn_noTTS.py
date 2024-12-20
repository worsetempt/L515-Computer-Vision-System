import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

def main():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error starting the pipeline: {e}")
        return

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Create align object to align depth frames to color frames
    align = rs.align(rs.stream.color)

    # Load Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model.eval()

    # Get the list of categories
    categories = weights.meta["categories"]

    while True:
        try:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convert the image from BGR to RGB
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Prepare the input for the model
            input_image = torch.from_numpy(color_image_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

            # Perform detection
            with torch.no_grad():
                predictions = model(input_image)

            # Process predictions
            for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
                if score > 0.7:
                    box = box.cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    class_name = categories[label]

                    # Get depth of center point of bounding box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    depth = depth_image[center_y, center_x] * depth_scale

                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, f"{class_name} - {depth:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the result
            cv2.imshow("Faster R-CNN Object Detection with Depth", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
