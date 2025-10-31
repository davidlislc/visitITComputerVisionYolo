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
import json
from datetime import datetime

class IntrusionDetectionSystem:
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
        
        # INTRUSION DETECTION SETTINGS
        self.intrusion_zones = []  # List of intrusion zones
        self.authorized_faces = set()  # Set of authorized face names
        self.intrusion_active = True
        self.intrusion_log = []
        self.last_intrusion_alert = 0
        self.intrusion_cooldown = 10  # seconds between intrusion alerts
        
        # Load configuration
        self.load_intrusion_config()
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
    def load_intrusion_config(self):
        """Load intrusion detection configuration."""
        config_file = 'intrusion_config.json'
        
        # Default configuration
        default_config = {
            "intrusion_zones": [
                {
                    "name": "Main Door",
                    "x1": 0.3, "y1": 0.3,
                    "x2": 0.7, "y2": 0.7
                }
            ],
            "authorized_faces": ["admin", "user1", "employee"],
            "alert_threshold": 0.5,  # Minimum confidence for person detection
            "save_intrusion_images": True
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                # Load intrusion zones (convert relative to absolute coordinates)
                self.intrusion_zones = config.get("intrusion_zones", default_config["intrusion_zones"])
                self.authorized_faces = set(config.get("authorized_faces", default_config["authorized_faces"]))
                self.save_intrusion_images = config.get("save_intrusion_images", True)
                
                print(f"✓ Loaded intrusion config: {len(self.intrusion_zones)} zones, {len(self.authorized_faces)} authorized faces")
                
            except Exception as e:
                print(f"Error loading config: {e}")
                self.create_default_config(config_file, default_config)
        else:
            self.create_default_config(config_file, default_config)
            
    def create_default_config(self, config_file, default_config):
        """Create default intrusion configuration file."""
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"✓ Created default intrusion config: {config_file}")
            
            self.intrusion_zones = default_config["intrusion_zones"]
            self.authorized_faces = set(default_config["authorized_faces"])
            self.save_intrusion_images = default_config["save_intrusion_images"]
            
        except Exception as e:
            print(f"Error creating config: {e}")

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
    
    def point_in_zone(self, point, zone, frame_width, frame_height):
        """Check if point is inside intrusion zone."""
        x, y = point
        # Convert relative coordinates to absolute
        zone_x1 = int(zone['x1'] * frame_width)
        zone_y1 = int(zone['y1'] * frame_height)
        zone_x2 = int(zone['x2'] * frame_width)
        zone_y2 = int(zone['y2'] * frame_height)
        
        return zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2
    
    def check_intrusion(self, person_bbox, face_name, frame_width, frame_height):
        """Check if person is intruding in restricted zones."""
        if not self.intrusion_active:
            return False, None
            
        # Get center point of person
        x1, y1, x2, y2 = person_bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Check each intrusion zone
        for zone in self.intrusion_zones:
            if self.point_in_zone((center_x, center_y), zone, frame_width, frame_height):
                # Person is in restricted zone
                if face_name not in self.authorized_faces:
                    return True, zone['name']
        
        return False, None
    
    def log_intrusion(self, zone_name, face_name, confidence, frame=None):
        """Log intrusion event."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        intrusion_event = {
            "timestamp": timestamp,
            "zone": zone_name,
            "face_name": face_name,
            "confidence": confidence,
            "authorized": face_name in self.authorized_faces
        }
        
        self.intrusion_log.append(intrusion_event)
        
        # Save to file
        log_file = f"logs/intrusions_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
                
            logs.append(intrusion_event)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Error saving log: {e}")
        
        # Save intrusion image
        if self.save_intrusion_images and frame is not None:
            img_filename = f"logs/intrusion_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
            cv2.imwrite(img_filename, frame)
        
        print(f"🚨 INTRUSION LOGGED: {zone_name} - {face_name} at {timestamp}")
    
    def draw_intrusion_zones(self, frame):
        """Draw intrusion zones on frame."""
        frame_height, frame_width = frame.shape[:2]
        
        for i, zone in enumerate(self.intrusion_zones):
            # Convert relative to absolute coordinates
            x1 = int(zone['x1'] * frame_width)
            y1 = int(zone['y1'] * frame_height)
            x2 = int(zone['x2'] * frame_width)
            y2 = int(zone['y2'] * frame_height)
            
            # Draw zone rectangle
            color = (0, 255, 255) if self.intrusion_active else (128, 128, 128)  # Yellow if active, gray if inactive
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw zone label
            label = f"Zone: {zone['name']}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def detect_and_process(self, frame, current_time):
        """Detect both objects and faces simultaneously with intrusion detection."""
        detected_objects = set()
        face_results = []
        frame_height, frame_width = frame.shape[:2]
        
        # SIMULTANEOUS DETECTION: Run both models on the same frame
        object_results = self.object_model(frame, conf=0.5, verbose=False)
        face_results_raw = self.face_model.predict(frame, conf=0.5, verbose=False)
        
        # Process object detections and check for intrusions
        persons_detected = []
        
        for result in object_results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.object_model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detected_objects.add(class_name)
                    
                    # Special handling for person detection
                    if class_name == 'person':
                        persons_detected.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence
                        })
                    
                    # Draw object detection (Blue boxes)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Object label
                    label = f"{class_name} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0] + 10, y1), (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Process face detections and check intrusions
        recognized_faces = {}
        
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
                        
                        # Store recognized face location
                        face_center_x = (x1 + x2) // 2
                        face_center_y = (y1 + y2) // 2
                        recognized_faces[(face_center_x, face_center_y)] = name
                        
                        # Check for intrusion
                        is_intrusion, zone_name = self.check_intrusion((x1, y1, x2, y2), name, frame_width, frame_height)
                        
                        # Determine color and label based on intrusion status
                        if is_intrusion:
                            color = (0, 0, 255)  # Red for intruder
                            label = f"🚨 INTRUDER: {name}"
                            
                            # Alert for intrusion
                            if current_time - self.last_intrusion_alert > self.intrusion_cooldown:
                                alert_msg = f"Security Alert! Unauthorized person {name} detected in {zone_name}"
                                self.speak_async(alert_msg)
                                self.log_intrusion(zone_name, name, similarity, frame)
                                self.last_intrusion_alert = current_time
                                
                        elif name != "Unknown" and name != "Error" and "No Face" not in name:
                            if name in self.authorized_faces:
                                color = (0, 255, 0)  # Green for authorized
                                label = f"✓ {name} ({similarity:.3f})"
                            else:
                                color = (0, 165, 255)  # Orange for recognized but not authorized
                                label = f" {name} ({similarity:.3f})"
                            
                            # Announce face
                            if (name not in self.last_announcement or 
                                current_time - self.last_announcement[name] > 4):
                                greeting = f"Hello {name}" 
                                self.speak_async(greeting)
                                self.last_announcement[name] = current_time
                        else:
                            color = (0, 0, 255)  # Red for unknown
                            label = "Unknownson"
                        
                        # Draw face detection
                        thickness = 4 if is_intrusion else 3
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Face label with bigger text
                        font_scale = 0.8
                        text_thickness = 2
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                                     (x1 + label_size[0] + 15, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 7, y1 - 7), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
                        
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
                
                #self.speak_async(announcement)
                print(f"Objects: {', '.join(objects_list)}")
                
            self.prev_objects = detected_objects.copy()
            self.last_object_announcement = current_time
        
        return frame
    
    def run(self):
        """Main loop running both detections simultaneously with intrusion detection."""
        cap = cv2.VideoCapture(0)
        
        # Set larger frame size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("🚀 Intrusion Detection System Started")
        print(f"Frame size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"Reference faces: {len(self.reference_embeddings)}")
        print(f"Authorized faces: {', '.join(self.authorized_faces)}")
        print(f"Intrusion zones: {len(self.intrusion_zones)}")
        print("\nControls:")
        print("  q - Quit")
        print("  r - Reload reference faces")
        print("  i - Toggle intrusion detection")
        print("  f - Fullscreen")
        print("  w - Windowed mode")
        print("  s - Save current frame")
        
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
            
            # Draw intrusion zones first
            self.draw_intrusion_zones(frame)
            
            # Process every nth frame
            if self.frame_count % self.FRAME_SKIP == 0:
                # SIMULTANEOUS DETECTION: Both models run on same frame
                frame = self.detect_and_process(frame, current_time)
            
            # Show performance info
            status = "Speaking..." if self.speaking else "Ready"
            processing = "Processing Both" if self.frame_count % self.FRAME_SKIP == 0 else "Skipping"
            intrusion_status = "ACTIVE" if self.intrusion_active else "INACTIVE"
            
            cv2.putText(frame, f"FPS: {current_fps:.1f} | {status}", (15, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Intrusion: {intrusion_status} | {processing}", (15, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Authorized: {len(self.authorized_faces)} | Refs: {len(self.reference_embeddings)}", (15, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Zones: {len(self.intrusion_zones)} | Alerts: {len(self.intrusion_log)}", (15, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Create resizable window
            cv2.namedWindow('Intrusion Detection System', cv2.WINDOW_NORMAL)
            cv2.imshow('Intrusion Detection System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Reloading reference faces...")
                self.load_reference_faces()
            elif key == ord('i'):
                self.intrusion_active = not self.intrusion_active
                status = "ACTIVE" if self.intrusion_active else "INACTIVE"
                print(f"Intrusion detection: {status}")
            elif key == ord('f'):
                cv2.setWindowProperty('Intrusion Detection System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord('w'):
                cv2.setWindowProperty('Intrusion Detection System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Intrusion Detection System stopped")
        print(f"Total intrusion events logged: {len(self.intrusion_log)}")

# Usage
if __name__ == "__main__":
    try:
        system = IntrusionDetectionSystem()
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: pip install facenet-pytorch ultralytics pyttsx3")