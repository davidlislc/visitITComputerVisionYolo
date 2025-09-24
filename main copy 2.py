import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading
from collections import Counter

class ObjectDetectionSystem:
    def __init__(self, model_path='yolo11n.pt', confidence_threshold=0.5):
        """Initialize the object detection system."""
        self.model = YOLO(model_path)
        self.engine = pyttsx3.init()
        self.confidence_threshold = confidence_threshold
        self.prev_objects = set()
        self.last_announcement_time = 0
        self.announcement_cooldown = 3  # seconds
        self.speaking = False
        
        # Configure TTS settings
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
        
    def speak_async(self, text):
        """Speak text asynchronously to avoid blocking the main thread."""
        if not self.speaking:
            self.speaking = True
            def speak():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                finally:
                    self.speaking = False
            
            thread = threading.Thread(target=speak, daemon=True)
            thread.start()
    
    def process_detections(self, results):
        """Process YOLO detection results and return detected objects."""
        detected_objects = []
        current_time = time.time()
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Filter by confidence
                    if box.conf[0] >= self.confidence_threshold:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        detected_objects.append((class_name, confidence))
        
        return detected_objects, current_time
    
    def should_announce(self, current_objects, current_time):
        """Determine if we should make an announcement."""
        # Convert to set for comparison
        current_set = set(obj[0] for obj in current_objects)
        
        # Check if enough time has passed since last announcement
        time_passed = current_time - self.last_announcement_time > self.announcement_cooldown
        
        # Check if objects have changed
        objects_changed = current_set != self.prev_objects
        
        return time_passed and objects_changed and not self.speaking
    
    def create_announcement(self, detected_objects):
        """Create announcement text from detected objects."""
        if not detected_objects:
            return ""
        
        # Count occurrences of each object
        object_counts = Counter(obj[0] for obj in detected_objects)
        
        # Create announcement
        announcement_parts = []
        for obj_name, count in object_counts.items():
            if count == 1:
                announcement_parts.append(f"one {obj_name}")
            else:
                announcement_parts.append(f"{count} {obj_name}s")
        
        if len(announcement_parts) == 1:
            return f"I can see {announcement_parts[0]}"
        elif len(announcement_parts) == 2:
            return f"I can see {announcement_parts[0]} and {announcement_parts[1]}"
        else:
            return f"I can see {', '.join(announcement_parts[:-1])}, and {announcement_parts[-1]}"
    
    def run(self, camera_index=0):
        """Main detection loop."""
        camera = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not camera.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Object Detection System Started. Press 'q' to quit.")
        frame_count = 0
        
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                frame_count += 1
                
                # Run detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Process detections
                detected_objects, current_time = self.process_detections(results)
                
                # Create annotated frame
                annotated_frame = results[0].plot()
                
                # Add detection count to frame
                cv2.putText(annotated_frame, f"Objects: {len(detected_objects)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('YOLOv11 Object Detection', annotated_frame)
                
                # Handle announcements
                if detected_objects and self.should_announce(detected_objects, current_time):
                    announcement = self.create_announcement(detected_objects)
                    print(f"Announcing: {announcement}")
                    self.speak_async(announcement)
                    
                    # Update state
                    self.prev_objects = set(obj[0] for obj in detected_objects)
                    self.last_announcement_time = current_time
                
                # Print detection info every 30 frames
                if frame_count % 30 == 0 and detected_objects:
                    print(f"Currently detecting: {[f'{obj[0]} ({obj[1]:.2f})' for obj in detected_objects]}")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping detection system...")
        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            camera.release()
            cv2.destroyAllWindows()
            print("Detection system stopped.")

def main():
    """Main function to run the object detection system."""
    detector = ObjectDetectionSystem(
        model_path='yolo11n.pt',
        confidence_threshold=0.5
    )
    detector.run(camera_index=0)

if __name__ == "__main__":
    main()