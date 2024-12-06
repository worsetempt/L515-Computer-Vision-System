import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

def get_color(class_id):
    color = tuple(((class_id * 5) % 256, (class_id * 50) % 256, (class_id * 100) % 256))
    return color

def main():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error starting the pipeline: {e}")
        return

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        pipeline.stop()
        return

    while True:
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            results = model(color_image)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    if conf > 0.5:
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        depth = depth_image[center_y, center_x] * depth_scale

                        color = get_color(cls)

                        cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(color_image, f"{label} - {depth:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("YOLOv8 Object Detection with Depth", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()