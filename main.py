import cv2
from ultralytics import YOLO
import pyttsx3
import time 

model = YOLO('yolo11n.pt')
engine = pyttsx3.init()
prevText = ""
text=""
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

    text=""       
    for result in results:
        for obj in result.boxes.data.tolist():
            class_id = int(obj[5])
            class_name = model.names[class_id]
            text += class_name + ", "
   
    if len(results) != 0:
        print(text)    
        print(prevText) 
        print(text != prevText)
        if text != prevText:
            prevText = text
            engine.say("the following objects: " + text)
            engine.runAndWait()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)  # Sleep for 1 second each loop

camera.release()
cv2.destroyAllWindows()