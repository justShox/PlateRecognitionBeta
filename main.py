import cv2
import os
from datetime import datetime
from yolov5 import YOLOv5
from paddleocr import PaddleOCR
from custom_ocr.custom_ocr import read_license_plate
from utils.resize_frame import resize_frame_to_screen
from sql_setup.database import setup_database, insert_detection_data


def load_plate_model():
    model_path = os.path.join(os.getcwd(), 'models', 'plate_detection.pt')
    return YOLOv5(model_path, device="cpu")


def process_video(video_path, plate_model, ocr, session):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break


        frame = resize_frame_to_screen(frame, 1280, 720)

        plate_results = plate_model.predict(frame)

        for bbox in plate_results.xyxy[0]:
            px1, py1, px2, py2, conf, cls = bbox
            px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])

            if conf > 0.5:
                plate_img = frame[py1:py2, px1:px2]
                plate_result = read_license_plate(plate_img, ocr)

                if isinstance(plate_result, tuple) and len(plate_result) >= 2:
                    text, confidence = plate_result
                    if text is not None and isinstance(text, str) and text.strip():
                        print(f"Plate Detected: {text}, Confidence: {confidence}")

                        insert_detection_data(session, text, float(confidence), video_path)

                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                        cv2.putText(frame, text, (px1, py1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                elif plate_result is not None and isinstance(plate_result, str) and plate_result.strip():
                    text = plate_result
                    print(f"Plate Detected: {text}, Confidence: {conf}")
                    insert_detection_data(session, text, float(conf), video_path)

                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (px1, py1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("License Plate Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    video_path = os.path.join(os.getcwd(), 'videos', 'test_video_1.mp4')

    plate_model = load_plate_model()
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    session = setup_database()

    process_video(video_path, plate_model, ocr, session)


if __name__ == "__main__":
    main()

# import cv2
# import os
# import threading
# import queue
# import time
# import numpy as np
# from datetime import datetime
# from yolov5 import YOLOv5
# from paddleocr import PaddleOCR
# from custom_ocr.custom_ocr import read_license_plate
# from utils.resize_frame import resize_frame_to_screen
# from sql_setup.database import setup_database, insert_detection_data
# from concurrent.futures import ThreadPoolExecutor
# import glob
# import collections
#
# cv_lock = threading.Lock()
#
# detected_plates = collections.defaultdict(lambda: {'last_seen': 0, 'count': 0})
#
#
# def load_plate_model():
#     """Load YOLOv5 model for plate recognition."""
#     model_path = os.path.join(os.getcwd(), 'models', 'plate_detection.pt')
#     return YOLOv5(model_path, device="cpu")
#
#
# def process_video_frames(video_path, plate_model, ocr, session, result_queue):
#     """Process video frames and put results in queue - no GUI operations here."""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Cannot open video file {video_path}")
#         return
#
#     video_name = os.path.basename(video_path)
#     frame_count = 0
#     plate_cache = {}  # Cache to track plates detected in this video
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"End of video or error reading frame in {video_name}.")
#             break
#
#         frame_count += 1
#         processed_frame = frame.copy()
#
#         processed_frame = resize_frame_to_screen(processed_frame, 1280, 720)
#
#         detection_start_time = time.time()
#
#         plate_results = plate_model.predict(processed_frame)
#
#         detections = []
#         for bbox in plate_results.xyxy[0]:
#             px1, py1, px2, py2, conf, cls = bbox
#             px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
#
#             if conf > 0.5:
#                 plate_img = processed_frame[py1:py2, px1:px2]
#
#                 ocr_start_time = time.time()
#                 plate_result = read_license_plate(plate_img, ocr)
#                 ocr_time = time.time() - ocr_start_time
#
#                 if isinstance(plate_result, tuple) and len(plate_result) >= 2:
#                     text, confidence = plate_result
#                     if (text is not None and isinstance(text, str) and text.strip() and
#                             float(confidence) > 0.8):  # 80% confidence threshold
#
#                         current_time = time.time()
#                         plate_key = f"{video_name}:{text}"
#
#                         if plate_key not in plate_cache or current_time - plate_cache[plate_key]['time'] > 5.0:
#                             detection_time = time.time() - detection_start_time
#
#                             print(
#                                 f"[{video_name}] Plate Detected: {text}, Confidence: {confidence:.2f}, Detection Time: {detection_time:.3f}s, OCR Time: {ocr_time:.3f}s")
#
#                             insert_detection_data(session, text, float(confidence), video_path)
#
#                             plate_cache[plate_key] = {'time': current_time, 'confidence': confidence}
#
#                         detections.append((px1, py1, px2, py2, text, confidence, detection_time))
#
#                 elif plate_result is not None and isinstance(plate_result, str) and plate_result.strip():
#                     text = plate_result
#                     if conf > 0.85:
#                         # Check if this plate was recently detected
#                         current_time = time.time()
#                         plate_key = f"{video_name}:{text}"
#
#                         if plate_key not in plate_cache or current_time - plate_cache[plate_key]['time'] > 5.0:
#                             detection_time = time.time() - detection_start_time
#
#                             print(
#                                 f"[{video_name}] Plate Detected: {text}, Confidence: {conf:.2f}, Detection Time: {detection_time:.3f}s, OCR Time: {ocr_time:.3f}s")
#
#                             insert_detection_data(session, text, float(conf), video_path)
#
#                             plate_cache[plate_key] = {'time': current_time, 'confidence': conf}
#
#                         detections.append((px1, py1, px2, py2, text, conf, detection_time))
#
#         result_queue.put((video_name, processed_frame, detections))
#
#         if frame_count % 2 == 0:  # Process every other frame
#             time.sleep(0.01)  # Small sleep to prevent hogging resources
#
#     cap.release()
#     result_queue.put((video_name, None, None))
#
#
# def display_results(result_queue, max_videos):
#     """Display results from the queue in a single window with grid layout."""
#     running_videos = set()
#     video_frames = {}
#
#     with cv_lock:
#         window_name = "License Plate Detection - Multiple Videos"
#         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(window_name, 1920, 1080)  # Change as resolution of ur screen to fit all the videos if needed
#
#     start_time = time.time()
#     fps_counter = 0
#
#     while True:
#         try:
#             video_name, frame, detections = result_queue.get(timeout=1.0)
#
#             if frame is None:
#                 running_videos.discard(video_name)
#                 if video_name in video_frames:
#                     del video_frames[video_name]
#                 # If no more videos, exit
#                 if len(running_videos) == 0:
#                     break
#                 continue
#
#             running_videos.add(video_name)
#
#             fps_counter += 1
#             elapsed_time = time.time() - start_time
#             if elapsed_time >= 1.0:
#                 fps = fps_counter / elapsed_time
#                 fps_counter = 0
#                 start_time = time.time()
#             else:
#                 fps = fps_counter / (elapsed_time if elapsed_time > 0 else 1)
#
#             if detections:
#                 for px1, py1, px2, py2, text, conf, detection_time in detections:
#                     cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
#
#                     label = f"{text} ({conf:.2f})"
#                     cv2.putText(frame, label, (px1, py1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#
#                     time_label = f"{detection_time * 1000:.1f}ms"
#                     cv2.putText(frame, time_label, (px1, py2 + 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#
#             cv2.putText(frame, video_name, (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#             cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
#
#             video_frames[video_name] = frame
#
#             num_videos = len(video_frames)
#             if num_videos > 0:
#                 # Calculate grid size based on number of videos
#                 if num_videos <= 4:
#                     grid_cols, grid_rows = 2, 2
#                 elif num_videos <= 6:
#                     grid_cols, grid_rows = 3, 2
#                 elif num_videos <= 9:
#                     grid_cols, grid_rows = 3, 3
#                 else:  # Up to 12 videos
#                     grid_cols, grid_rows = 4, 3
#
#                 h, w = next(iter(video_frames.values())).shape[:2]
#                 grid_h, grid_w = h // grid_rows, w // grid_cols
#
#                 canvas = np.zeros((grid_rows * grid_h, grid_cols * grid_w, 3), dtype=np.uint8)
#
#                 idx = 0
#                 for _, frame in video_frames.items():
#                     if idx >= grid_rows * grid_cols:
#                         break
#                     row, col = idx // grid_cols, idx % grid_cols
#                     resized = cv2.resize(frame, (grid_w, grid_h))
#                     y_start = row * grid_h
#                     y_end = (row + 1) * grid_h
#                     x_start = col * grid_w
#                     x_end = (col + 1) * grid_w
#                     canvas[y_start:y_end, x_start:x_end] = resized
#                     idx += 1
#
#                 with cv_lock:
#                     cv2.imshow(window_name, canvas)
#                     key = cv2.waitKey(1)
#                     if key == ord('q'):
#                         break
#         except queue.Empty:
#             if len(running_videos) == 0:
#                 break
#
#             if video_frames:
#                 with cv_lock:
#                     cv2.waitKey(1)
#         except Exception as e:
#             print(f"Error in display thread: {e}")
#
#     with cv_lock:
#         cv2.destroyAllWindows()
#
#
# def process_multiple_videos(video_paths, max_concurrent=10):
#     plate_model = load_plate_model()
#     ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
#     session = setup_database()
#
#     result_queue = queue.Queue(maxsize=100)
#
#     display_thread = threading.Thread(
#         target=display_results,
#         args=(result_queue, len(video_paths))
#     )
#     display_thread.daemon = True
#     display_thread.start()
#
#     with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
#         futures = []
#         for video_path in video_paths:
#             future = executor.submit(
#                 process_video_frames,
#                 video_path,
#                 plate_model,
#                 ocr,
#                 session,
#                 result_queue
#             )
#             futures.append(future)
#
#         for future in futures:
#             try:
#                 future.result()
#             except Exception as e:
#                 print(f"Error in video processing: {e}")
#
#     display_thread.join(timeout=5.0)
#
#     print("All videos processed. Detection results saved to database.")
#
#
# def find_videos(path):
#     if os.path.isfile(path):
#         return [path]
#     elif os.path.isdir(path):
#         video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
#         videos = []
#         for ext in video_extensions:
#             videos.extend(glob.glob(os.path.join(path, f'*{ext}')))
#         return videos
#     else:
#         return glob.glob(path)
#
#
# def main():
#     video_dir = os.path.join(os.getcwd(), 'videos')
#
#     specific_videos = [
#         os.path.join(os.getcwd(), 'videos', 'test_video_1.mp4'),
#         os.path.join(os.getcwd(), 'videos', 'test_video_1.mp4'),
#         os.path.join(os.getcwd(), 'videos', 'test_video_1.mp4'),
#         os.path.join(os.getcwd(), 'videos', 'test_video_1.mp4'),
#         os.path.join(os.getcwd(), 'videos', 'test_video_1.mp4'),
#         os.path.join(os.getcwd(), 'videos', 'test_video_1.mp4'),
#         os.path.join(os.getcwd(), 'videos', 'test_video_1.mp4'),
#         os.path.join(os.getcwd(), 'videos', 'test_video_1.mp4'),
#         # os.path.join(os.getcwd(), 'videos', 'test_video_2.mp4'),
#         # Add more video paths as needed
#     ]
#
#     videos_to_process = find_videos(video_dir)
#     # videos_to_process = specific_videos
#
#     print(f"Found {len(videos_to_process)} videos to process:")
#     for i, video in enumerate(videos_to_process):
#         print(f"{i + 1}. {video}")
#
#     max_concurrent = min(10, len(videos_to_process))
#
#     process_multiple_videos(videos_to_process, max_concurrent)
#
#
# if __name__ == "__main__":
#     main()
