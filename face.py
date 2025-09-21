import traceback
from ultralytics import YOLO
from deepface import DeepFace
import cv2

# Load YOLOv11 model (trained to detect faces)
model = YOLO('yolo11n.pt')  # Replace with a face-trained model if available

# Load reference image for recognition
reference_img = cv2.imread('li.jpg')  # Image of the person to recognize

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using YOLOv11
    results = model.predict(source=frame, conf=0.5, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            if cls_name == 'person':  # Or 'face' if model supports it
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]

                # Compare cropped face to reference image
                try:
                    verification = DeepFace.verify(face_crop, reference_img, enforce_detection=False)
                    match = verification['verified']
                    label = 'MATCH' if match else 'NO MATCH'
                except Exception as e:
                    label = 'ERROR'
                    print(f"DeepFace error: {e}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Full traceback:")
                    traceback.print_exc()   
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('Facial Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
