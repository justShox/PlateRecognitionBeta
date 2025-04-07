import cv2
import numpy as np
from datetime import datetime
import os
from yolov5 import YOLOv5
import time
from paddleocr import PaddleOCR
from custom_ocr import custom_ocr as cocr
from utils import get_screen_size, resize_frame_to_screen
from sql_queries import insert_detection_data, setup_database, get_cam_info


def load_plate_model():
    """Load YOLOv5 model for plate recognition."""
    model_path = os.path.join(os.getcwd(), 'models', 'plate_detection.pt')
    return YOLOv5(model_path, device="cpu")


def initialize_video_stream(rtsp_url):
    """Initialize video capture from a given RTSP URL."""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Cannot open video stream {rtsp_url}")
        return False
    return cap


def detect_plates(img, model):
    """Detect license plates in the image using YOLOv5 model."""
    results = model.predict(img)
    return results


def preprocess_plate_image(plate_img):
    """Preprocess the plate image for better OCR results."""
    height, width = plate_img.shape[:2]
    plate_img = cv2.resize(plate_img, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_plate_img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    filtered = cv2.bilateralFilter(equalized, 5, 17, 17)
    _, thresh_plate_img = cv2.threshold(filtered, 70, 255, cv2.THRESH_BINARY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh_plate_img, -1, kernel)
    inverted = cv2.bitwise_not(sharpened)
    return gray_plate_img, thresh_plate_img, filtered, inverted


# def process_camera(camera_url, cam_index, plate_model, conn, area, screen_width, screen_height, ocr):
#     """Process function for each camera."""
#     cap = initialize_video_stream(camera_url)
#     is_open = False
#
#     try:
#         is_open = cap.isOpened()
#     except:
#         pass
#
#     if is_open:
#
#         ret, frame = cap.read()
#
#         cap.release()
#
#         if not ret:
#             print(f"Connection lost with camera {camera_url}. Attempting to reconnect...")
#             cap.release()
#
#             for attempt in range(3):
#                 print(f"Reconnection attempt {attempt + 1}...")
#                 cap = initialize_video_stream(camera_url)
#                 ret, frame = cap.read()
#
#                 if ret:
#                     print(f"Reconnected to camera {camera_url}.")
#                     break
#
#                 print(f"Reconnection attempt {attempt + 1} failed. Retrying in 1 second...")
#                 time.sleep(1)
#
#             if not ret:
#                 print(f"Failed to reconnect to camera {camera_url}. Exiting...")
#                 return False
#
#         frame = resize_frame_to_screen(frame, screen_width, screen_height)
#
#         x1, y1, x2, y2 = area
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
#         cv2.putText(frame, "Detection area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 250, 100), 2)
#
#         recognition_frame = frame[y1:y2, x1:x2]
#
#         plate_results = detect_plates(recognition_frame, plate_model)
#         print("xxx", plate_results)
#
#
#         for bbox in plate_results.xyxy[0]:
#             px1, py1, px2, py2, conf, cls = bbox
#             px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
#
#             if conf > 0.5:
#                 plate_img = recognition_frame[py1:py2, px1:px2]
#
#                 gray_plate_img, thresh_plate_img, filtered, inverted = preprocess_plate_image(plate_img)
#
#                 for img in [gray_plate_img, thresh_plate_img, filtered, inverted]:
#                     try:
#                         plate_text, score = rp(img, ocr)
#                         if plate_text:
#                             break
#                     except RuntimeError as e:
#                         print(f"Error processing OCR on cropped plate image: {e}")
#                         continue
#
#                 if plate_text:
#                     print(f'Camera {cam_index}: Plate Detected: {plate_text}')
#                     insert_detection_data(conn, plate_text, camera_url)
#                 else:
#                     print(f'Camera {cam_index}: Plate Detected: None')
#
#                 # Draw bounding box around the detected plate
#                 cv2.rectangle(frame, (px1+x1, py1+y1), (px2+x1, py2+y1), (0, 255, 0), 2)
#
#                 # Display detected text above the bounding box
#                 cv2.putText(frame, f"{plate_text}", (px1+x1, py1+y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
#
#         cv2.imshow(f'Camera {cam_index}', frame)
#         cv2.waitKey(1)
#
#         return True
#
#     return False

def process_video_optimized(video_path, cam_index, plate_model, conn, area, screen_width, screen_height, ocr):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False

    frame_count = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip every other frame
        if frame_count % 2 != 0:
            frame_count += 1
            continue

        start_time = time.time()

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Detection area
        x1, y1, x2, y2 = area
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        recognition_frame = frame[y1:y2, x1:x2]

        # Detect license plates
        plate_results = detect_plates(recognition_frame, plate_model)

        for bbox in plate_results.xyxy[0]:
            px1, py1, px2, py2, conf, cls = bbox
            px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])

            if conf > 0.3:  # Lower confidence threshold
                plate_img = recognition_frame[py1:py2, px1:px2]

                # Preprocess plate image
                plate_img = cv2.resize(plate_img, (200, 50))
                _, thresh_plate_img = cv2.threshold(cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY), 70, 255, cv2.THRESH_BINARY)

                # Perform OCR
                plate_text, score = cocr(thresh_plate_img, ocr)
                if plate_text:
                    print(f'Camera {cam_index}: Plate Detected: {plate_text}')
                    insert_detection_data(conn, plate_text, video_path)

        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time
        frame_count += 1

        print(f"Frame {frame_count}: Time taken = {processing_time:.2f} seconds")

    # Calculate and print average FPS
    if frame_count > 0:
        avg_time_per_frame = total_time / frame_count
        fps = 1 / avg_time_per_frame
        print(f"Average time per frame: {avg_time_per_frame:.2f} seconds")
        print(f"Average FPS: {fps:.2f}")

    cap.release()
    return True

def main():
    plate_model = load_plate_model()
    screen_width, screen_height = get_screen_size()
    conn = setup_database()
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

    # Replace camera URLs with local video file paths
    video_paths = [
        "test_video_1.mp4",
        # "path/to/your/video2.mp4",
    ]

    areas = [(0, 0, 1280, 720)] * len(video_paths)  # Define detection areas for each video

    while True:
        x = datetime.now()
        for i, (video_path, area) in enumerate(zip(video_paths, areas)):
            process_video_optimized(video_path, i + 1, plate_model, conn, area, screen_width, screen_height, ocr)
        y = datetime.now()
        print("PROCESS TIME: ", y - x)

# def main():
#     plate_model = load_plate_model()
#     screen_width, screen_height = get_screen_size()
#     conn = setup_database()
#     ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
#
#     camera_urls = [
#         0
#         # 2,
#         # "rtsp://admin:admin.123@192.168.1.20:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:admin.123@192.168.1.21:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:admin.123@192.168.1.22:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:admin.123@192.168.1.23:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:admin.123@192.168.1.24:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:admin.123@192.168.1.25:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:admin.123@192.168.1.26:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:admin.123@192.168.1.27:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:admin.123@192.168.1.28:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:admin.123@192.168.1.29:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.10:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.11:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.13:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.14:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.15:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.16:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.17:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.18:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.19:554/cam/realmonitor?channel=1&subtype=0",
#         # "rtsp://admin:E12345678r@192.168.1.64:554/Streaming/Channels/101",
#         # "rtsp://admin:E12345678r@192.168.1.2:554/Streaming/Channels/101",
#     ]
#
#     # Initialize video streams once for all cameras
#     # caps = [initialize_video_stream(url) for url in camera_urls]
#
#     areas = [get_cam_info(url) for url in camera_urls]
#
#     while True:
#         x = datetime.now()
#         for i, (url, area) in enumerate(zip(camera_urls, areas)):
#             process_camera(url, i + 1, plate_model, conn, area, screen_width, screen_height, ocr)
#         y = datetime.now()
#         print("PROCESS TIME: ", y-x)


if __name__ == "__main__":
    main()
