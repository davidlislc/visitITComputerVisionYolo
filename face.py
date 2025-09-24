import traceback
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import pyttsx3
import threading
import time

# Load YOLOv11 model (trained to detect faces)
model = YOLO('yolo11n.pt')  # Replace with a face-trained model if available

# Add face detection using OpenCV as backup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load reference image for recognition
reference_img = cv2.imread('li.jpg')  # Image of the person to recognize

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.8)  # Volume level

# Speech control variables
last_match_time = 0
speech_cooldown = 3  # seconds between announcements
speaking = False

def speak_async(text):
    """Speak text asynchronously to avoid blocking video processing."""
    global speaking
    if not speaking:
        speaking = True
        def speak():
            global speaking
            try:
                engine.say(text)
                engine.runAndWait()
            finally:
                speaking = False
        
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using YOLOv11
    results = model.predict(source=frame, conf=0.5, verbose=False)
    
    current_time = time.time()

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls_name = result.names[int(box.cls)]
                if cls_name == 'person':  # Or 'face' if model supports it
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_crop = frame[y1:y2, x1:x2]
                    
                    # Use OpenCV to find face within the person detection
                    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    for (fx, fy, fw, fh) in faces:
                        x1_face, y1_face = x1 + fx, y1 + fy
                        x2_face, y2_face = x1 + fx + fw, y1 + fy + fh
                        face_crop = frame[y1_face:y2_face, x1_face:x2_face]

                        # Compare cropped face to reference image
                        try:
                            verification = DeepFace.verify(
                                face_crop, 
                                reference_img, 
                                enforce_detection=False
                            )
                            match = verification['verified']
                            distance = verification['distance']
                            
                            if match:
                                label = f'MATCH ({distance:.3f})'
                                color = (0, 255, 0)  # Green for match
                                
                                # Speak "Match" if enough time has passed
                                if current_time - last_match_time > speech_cooldown:
                                    speak_async("Face match detected")
                                    last_match_time = current_time
                                    print(f"Match announced! Distance: {distance:.3f}")
                            else:
                                label = f'NO MATCH ({distance:.3f})'
                                color = (0, 0, 255)  # Red for no match
                                
                        except Exception as e:
                            label = 'ERROR'
                            color = (0, 255, 255)  # Yellow for error
                            print(f"DeepFace error: {e}")
                            print(f"Error type: {type(e).__name__}")
                            print("Full traceback:")
                            traceback.print_exc()
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1_face, y1_face), (x2_face, y2_face), color, 2)
                        cv2.putText(
                            frame, 
                            label, 
                            (x1_face, y1_face - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            color, 
                            2
                        )

    # Add status indicator
    status_text = "Speaking..." if speaking else "Ready"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Facial Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
