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

class UltraFastFaceRecognition:
    def __init__(self):
        # Check if CUDA is available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models
        self.yolo_model = YOLO('yolov8n-face.pt')  # Face-specific YOLO
        
        # Initialize MTCNN for face detection and alignment
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
            factor=0.709, 
            post_process=True,
            device=self.device,
            keep_all=False  # Only keep best face
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
        self.last_announcement = {}
        
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
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Extract face using MTCNN
                    img_cropped = self.mtcnn(img)
                    
                    if img_cropped is not None:
                        # Get embedding
                        img_cropped = img_cropped.unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            embedding = self.resnet(img_cropped)
                            
                        # Store normalized embedding
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
            # Convert BGR to RGB
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(face_img)
            
            # Extract and align face
            face_tensor = self.mtcnn(pil_img)
            
            if face_tensor is not None:
                # Get embedding
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.resnet(face_tensor)
                
                # Normalize embedding
                embedding = embedding.cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
            else:
                return None
                
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def recognize_face_fast(self, face_img):
        """Ultra-fast face recognition using FaceNet embeddings."""
        try:
            # Get face embedding
            face_embedding = self.get_face_embedding(face_img)
            
            if face_embedding is None:
                return "No Face", 0
            
            if not self.reference_embeddings:
                return "No References", 0
            
            best_similarity = -1
            best_name = "Unknown"
            
            # Compare with all reference embeddings
            for name, ref_embedding in self.reference_embeddings.items():
                similarity = self.cosine_similarity(face_embedding, ref_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name
            
            # Threshold for recognition (FaceNet typically uses 0.6-0.8)
            if best_similarity > 0.6:
                return best_name, best_similarity
            else:
                return f"Unknown ({best_name}?)", best_similarity
                
        except Exception as e:
            print(f"Recognition error: {e}")
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
    
    def run(self):
        """Main loop with FaceNet optimization."""
        cap = cv2.VideoCapture(0)
        
        # Set larger frame size for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Increased from 640
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Increased from 480
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Alternative resolutions you can try:
        # HD: 1280x720
        # Full HD: 1920x1080
        # 4K: 3840x2160 (if your camera supports it)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("🚀 Ultra-Fast FaceNet Recognition Started")
        print(f"Frame size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"References loaded: {len(self.reference_embeddings)}")
        
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
                # Detect faces with YOLO
                results = self.yolo_model.predict(frame, conf=0.5, verbose=False)
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            
                            # Add some padding around face
                            padding = 30  # Increased padding for bigger frames
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(frame.shape[1], x2 + padding)
                            y2 = min(frame.shape[0], y2 + padding)
                            
                            # Extract face
                            face_crop = frame[y1:y2, x1:x2]
                            
                            if face_crop.size > 0:
                                # Fast recognition using FaceNet
                                name, similarity = self.recognize_face_fast(face_crop)
                                
                                # Draw results with bigger text for larger frame
                                if name != "Unknown" and "Error" not in name and "No" not in name:
                                    if "?" not in name:  # High confidence match
                                        color = (0, 255, 0)  # Green
                                        label = f"{name} ({similarity:.3f})"
                                        
                                        # Announce
                                        if (name not in self.last_announcement or 
                                            current_time - self.last_announcement[name] > 4):
                                            self.speak_async(f"Hello {name}")
                                            self.last_announcement[name] = current_time
                                    else:  # Low confidence match
                                        color = (0, 165, 255)  # Orange
                                        label = name
                                else:
                                    color = (0, 0, 255)  # Red
                                    label = name
                                
                                # Draw bounding box with thicker lines for bigger frame
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                
                                # Label background with bigger text
                                font_scale = 0.8  # Increased from 0.6
                                thickness = 2
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                                cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                                            (x1 + label_size[0] + 15, y1), color, -1)
                                
                                cv2.putText(frame, label, (x1 + 7, y1 - 7), 
                                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                                
                                # Show YOLO confidence with bigger text
                                cv2.putText(frame, f"Det: {confidence:.2f}", (x1, y2 + 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show performance info with bigger text for larger frame
            status = "Speaking..." if self.speaking else "Ready"
            processing = "Processing" if self.frame_count % self.FRAME_SKIP == 0 else "Skipping"
            
            cv2.putText(frame, f"FPS: {current_fps:.1f} | {status}", (15, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"{processing} | Refs: {len(self.reference_embeddings)}", (15, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Device: {self.device}", (15, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame size info
            cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", (15, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Create resizable window for better viewing
            cv2.namedWindow('FaceNet Recognition', cv2.WINDOW_NORMAL)
            cv2.imshow('FaceNet Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Reloading reference faces...")
                self.load_reference_faces()
            elif key == ord('f'):
                # Toggle fullscreen
                cv2.setWindowProperty('FaceNet Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord('w'):
                # Return to windowed mode
                cv2.setWindowProperty('FaceNet Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ FaceNet Recognition stopped")

# Usage
if __name__ == "__main__":
    try:
        recognizer = UltraFastFaceRecognition()
        recognizer.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: pip install facenet-pytorch")
