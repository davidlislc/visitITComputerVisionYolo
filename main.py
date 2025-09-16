import cv2
from ultralytics import YOLO
model = YOLO('yolo11n.pt')

#results = model('catanddog.jpeg')
#results[0].show()
camera = cv2.VideoCapture(0)
while True: 
    ret, frame = camera.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv11 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()