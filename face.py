import traceback
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import pyttsx3
import threading
import time
import os
import glob

# Load YOLOv11 model (trained to detect faces)
model = YOLO('yolo11n.pt')  # Replace with a face-trained model if available

# Add face detection using OpenCV as backup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load reference images with names
def load_reference_faces(reference_folder='faces'):
    """
    Load reference images from a folder structure like:
    faces/
    ├── john.jpg
    ├── mary.png
    └── alex.jpeg
    """
    reference_faces = {}
    
    if not os.path.exists(reference_folder):
        print(f"Creating {reference_folder} folder...")
        os.makedirs(reference_folder)
        print(f"Please add reference images to the {reference_folder} folder")
        return reference_faces
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    for ext in extensions:
        image_files = glob.glob(os.path.join(reference_folder, ext))
        for image_path in image_files:
            # Get name from filename (without extension)
            name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Load image
            img = cv2.imread(image_path)
            if img is not None:
                reference_faces[name] = {
                    'image': img,
                    'path': image_path
                }
                print(f"✓ Loaded reference face: {name}")
            else:
                print(f"✗ Failed to load: {image_path}")
    
    return reference_faces

# Alternative: Manual face dictionary
def create_manual_face_database():
    """Manually define face database with names and image paths."""
    face_database = {
        'John': cv2.imread('john.jpg'),
        'Mary': cv2.imread('mary.jpg'),
        'Alex': cv2.imread('alex.jpg'),
        'Li': cv2.imread('li.jpg')  # Your existing reference
    }
    
    # Filter out None values (failed to load)
    face_database = {name: img for name, img in face_database.items() if img is not None}
    
    for name in face_database:
        print(f"✓ Loaded reference face: {name}")
    
    return face_database

# Load reference faces (choose one method)
reference_faces = load_reference_faces('faces')  # Method 1: From folder
# reference_faces = create_manual_face_database()  # Method 2: Manual

if not reference_faces:
    print("No reference faces loaded. Using default li.jpg")
    reference_faces = {'Li': cv2.imread('li.jpg')}

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.8)

# Speech control variables
last_announcement = {}  # Track last announcement time per person
speech_cooldown = 5  # seconds between announcements for same person
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

def recognize_face(face_crop, reference_faces, threshold=0.6):
    """
    Recognize face against multiple reference faces.
    Returns: (name, confidence, distance) or (None, 0, float('inf'))
    """
    best_match = None
    best_distance = float('inf')
    best_name = None
    
    for name, face_data in reference_faces.items():
        try:
            if isinstance(face_data, dict):
                reference_img = face_data['image']
            else:
                reference_img = face_data
                
            verification = DeepFace.verify(
                face_crop,
                reference_img,
                enforce_detection=False,
                model_name='VGG-Face'  # You can try 'Facenet', 'OpenFace', etc.
            )
            
            distance = verification['distance']
            verified = verification['verified']
            
            # Keep track of best match regardless of verification threshold
            if distance < best_distance:
                best_distance = distance
                best_name = name
                best_match = verified
                
        except Exception as e:
            print(f"Error comparing with {name}: {str(e)[:50]}")
            continue
    
    # Return best match if found and verified, or closest match with confidence
    if best_match and best_distance < threshold:
        confidence = max(0, (threshold - best_distance) / threshold)
        return best_name, confidence, best_distance
    elif best_name:
        # Return closest match even if not verified, with low confidence
        confidence = max(0, (threshold - best_distance) / threshold) * 0.5
        return f"Unknown ({best_name}?)", confidence, best_distance
    else:
        return None, 0, float('inf')

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
                if cls_name == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_crop = frame[y1:y2, x1:x2]
                    
                    # Use OpenCV to find face within the person detection
                    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    for (fx, fy, fw, fh) in faces:
                        x1_face, y1_face = x1 + fx, y1 + fy
                        x2_face, y2_face = x1 + fx + fw, y1 + fy + fh
                        face_crop = frame[y1_face:y2_face, x1_face:x2_face]

                        # Recognize face against all reference faces
                        try:
                            name, confidence, distance = recognize_face(face_crop, reference_faces)
                            
                            if name and confidence > 0.3:  # Minimum confidence threshold
                                if "Unknown" not in name:
                                    # Verified match
                                    label = f'{name} ({confidence:.2f})'
                                    color = (0, 255, 0)  # Green for known person
                                    
                                    # Announce name if enough time has passed
                                    if (name not in last_announcement or 
                                        current_time - last_announcement[name] > speech_cooldown):
                                        speak_async(f"Hello {name}")
                                        last_announcement[name] = current_time
                                        print(f"Announced: {name} (confidence: {confidence:.2f})")
                                else:
                                    # Possible match but low confidence
                                    label = name
                                    color = (0, 165, 255)  # Orange for uncertain
                            else:
                                label = f'Unknown Person'
                                color = (0, 0, 255)  # Red for unknown
                                
                        except Exception as e:
                            label = 'ERROR'
                            color = (0, 255, 255)  # Yellow for error
                            print(f"Recognition error: {e}")
                        
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

    # Add status indicators
    status_text = "Speaking..." if speaking else "Ready"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show number of loaded faces
    cv2.putText(frame, f"Known faces: {len(reference_faces)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Facial Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
