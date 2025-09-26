import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import time
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

class SimultaneousDetectionSystem:
    def __init__(self):
        # Check if CUDA is available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load both YOLO models
        self.object_model = YOLO('yolo11n.pt')      # General object detection
        self.face_model = YOLO('yolov8n-face.pt')   # Face detection
        
        # Initialize MTCNN for face detection and alignment
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709, 
            post_process=True,
            device=self.device,
            keep_all=False
        )
        
        # Initialize InceptionResnetV1 for face recognition
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Settings
        self.FRAME_SKIP = 3
        self.frame_count = 0
        self.reference_embeddings = {}
        self.load_reference_faces()
        
        # TTS
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)
        self.speaking = False
        
        # Announcement tracking
        self.last_announcement = {}
        self.prev_objects = set()
        self.last_object_announcement = 0
        
    def load_reference_faces(self):
        """Load reference faces and compute embeddings using FaceNet."""
        faces_folder = 'faces'
        if not os.path.exists(faces_folder):
            os.makedirs(faces_folder)
            print(f"Created {faces_folder} folder. Please add reference images.")
            return
            
        print("Loading reference faces with FaceNet...")
        
        for filename in os.listdir(faces_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(faces_folder, filename)
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_cropped = self.mtcnn(img)
                    
                    if img_cropped is not None:
                        img_cropped = img_cropped.unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            embedding = self.resnet(img_cropped)
                            
                        embedding = embedding.cpu().numpy().flatten()
                        embedding = embedding / np.linalg.norm(embedding)
                        
                        self.reference_embeddings[name] = embedding
                        print(f"✓ Loaded {name} (embedding shape: {embedding.shape})")
                    else:
                        print(f"✗ No face detected in {name}")
                        
                except Exception as e:
                    print(f"✗ Failed to process {name}: {str(e)}")
        
        print(f"Loaded {len(self.reference_embeddings)} reference faces")
    
    def cosine_similarity(self, emb1, emb2):
        """Fast cosine similarity calculation."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def get_face_embedding(self, face_img):
        """Extract face embedding using FaceNet."""
        try:
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(face_img)
            face_tensor = self.mtcnn(pil_img)
            
            if face_tensor is not None:
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.resnet(face_tensor)
                
                embedding = embedding.cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
            else:
                return None
                
        except Exception as e:
            return None
    
    def recognize_face_fast(self, face_img):
        """Ultra-fast face recognition using FaceNet embeddings."""
        try:
            face_embedding = self.get_face_embedding(face_img)
            
            if face_embedding is None:
                return "No Face", 0
            
            if not self.reference_embeddings:
                return "Unknown", 0
            
            best_similarity = -1
            best_name = "Unknown"
            
            for name, ref_embedding in self.reference_embeddings.items():
                similarity = self.cosine_similarity(face_embedding, ref_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name
            
            if best_similarity > 0.6:
                return best_name, best_similarity
            else:
                return "Unknown", best_similarity
                
        except Exception as e:
            return "Error", 0
    
    def speak_async(self, text):
        """Non-blocking speech."""
        if not self.speaking:
            self.speaking = True
            def speak():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                finally:
                    self.speaking = False
            threading.Thread(target=speak, daemon=True).start()
    
    def detect_and_process(self, frame, current_time):
        """Detect both objects and faces simultaneously."""
        detected_objects = set()
        face_results = []
        
        # SIMULTANEOUS DETECTION: Run both models on the same frame
        object_results = self.object_model(frame, conf=0.5, verbose=False)
        face_results_raw = self.face_model.predict(frame, conf=0.5, verbose=False)
        
        # Process object detections
        for result in object_results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.object_model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detected_objects.add(class_name)
                    
                    # Draw object detection (Blue boxes)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Object label
                    label = f"{class_name} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0] + 10, y1), (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Process face detections
        for result in face_results_raw:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Add padding around face
                    padding = 30
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    
                    # Extract face
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        # Recognize face
                        name, similarity = self.recognize_face_fast(face_crop)
                        
                        # Determine color and label
                        if name != "Unknown" and name != "Error" and "No Face" not in name:
                            color = (0, 255, 0)  # Green for recognized
                            label = f"{name} ({similarity:.3f})"
                            
                            # Announce face
                            if (name not in self.last_announcement or 
                                current_time - self.last_announcement[name] > 4):
                                self.speak_async(f"Hello {name}")
                                self.last_announcement[name] = current_time
                        else:
                            color = (0, 0, 255)  # Red for unknown
                            label = "Unknown"
                        
                        # Draw face detection (Green/Red boxes)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Face label with bigger text
                        font_scale = 0.8
                        thickness = 2
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                                     (x1 + label_size[0] + 15, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 7, y1 - 7), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                        
                        # Show detection confidence
                        cv2.putText(frame, f"Det: {confidence:.2f}", (x1, y2 + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Announce objects
        if (detected_objects != self.prev_objects and 
            current_time - self.last_object_announcement > 3):
            
            if detected_objects:
                objects_list = list(detected_objects)
                if len(objects_list) == 1:
                    announcement = f"I can see a {objects_list[0]}"
                else:
                    announcement = f"I can see {', '.join(objects_list[:-1])}, and a {objects_list[-1]}"
                
                self.speak_async(announcement)
                print(f"Objects: {', '.join(objects_list)}")
                
            self.prev_objects = detected_objects.copy()
            self.last_object_announcement = current_time
        
        return frame
    
    def run(self):
        """Main loop running both detections simultaneously."""
        cap = cv2.VideoCapture(0)
        
        # Set larger frame size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("🚀 Simultaneous Object Detection & Face Recognition Started")
        print(f"Frame size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"Reference faces: {len(self.reference_embeddings)}")
        print("\nBoth YOLOv11n (objects) and YOLOv8n-face run on EVERY frame!")
        print("Blue boxes = Objects | Green boxes = Recognized faces | Red boxes = Unknown faces")
        
        # Performance tracking
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            fps_counter += 1
            current_time = time.time()
            
            # Calculate FPS
            if fps_counter % 30 == 0:
                current_fps = 30 / (current_time - fps_start_time)
                fps_start_time = current_time
            
            # Process every nth frame
            if self.frame_count % self.FRAME_SKIP == 0:
                # SIMULTANEOUS DETECTION: Both models run on same frame
                frame = self.detect_and_process(frame, current_time)
            
            # Show performance info
            status = "Speaking..." if self.speaking else "Ready"
            processing = "Processing Both" if self.frame_count % self.FRAME_SKIP == 0 else "Skipping"
            
            cv2.putText(frame, f"FPS: {current_fps:.1f} | {status}", (15, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"{processing} | Refs: {len(self.reference_embeddings)}", (15, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Device: {self.device}", (15, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"YOLOv11n + YOLOv8n-face Simultaneous", (15, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Create resizable window
            cv2.namedWindow('Simultaneous Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Simultaneous Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Reloading reference faces...")
                self.load_reference_faces()
            elif key == ord('f'):
                cv2.setWindowProperty('Simultaneous Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord('w'):
                cv2.setWindowProperty('Simultaneous Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Simultaneous Detection System stopped")

# Usage
if __name__ == "__main__":
    try:
        system = SimultaneousDetectionSystem()
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: pip install facenet-pytorch ultralytics pyttsx3")