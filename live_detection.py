import torch
import cv2
import easyocr

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/plate_detection.pt')
model.eval()  # Set the model to evaluation mode

reader = easyocr.Reader(['en'])


def recognize_plate_easyocr(img, coords):
    x1, y1, x2, y2 = coords
    plate_img = img[int(y1):int(y2), int(x1):int(x2)]

    ocr_result = reader.readtext(plate_img)

    if ocr_result:
        return ocr_result[0][1]
    return ""



cap = cv2.VideoCapture(0)  #0 - for laptop camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = model(frame_rgb)
    detections = results.xyxy[0].numpy()

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if cls == 0:
            plate_text = recognize_plate_easyocr(frame_rgb, [x1, y1, x2, y2])
            print(f"Detected License Plate: {plate_text}")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Live License Plate Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()