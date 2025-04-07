import torch
import cv2


model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/plate_detection.pt')
model.eval()

def detect_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    results.print()
    results.show()
    results.save(save_dir='../ALPR_test/output/')

def detect_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        rendered_frame = results.render()[0]
        out.write(rendered_frame)
        cv2.imshow('YOLOv5 Detection', rendered_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# detect_image('images/license1.jpg')

detect_video('videos/test_video_1.mp4', '../ALPR_test/output/video_output.mp4')