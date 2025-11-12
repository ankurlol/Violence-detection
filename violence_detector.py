import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class SimpleViolenceDetector:
    def __init__(self):
        print("Initializing Simple Violence Detector...")
        # Use OpenCV's built-in face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize motion detection
        self.prev_gray = None
        print("Violence Detector ready!")
    
    def detect_violence_features(self, frame):
        """Detect features that might indicate violence"""
        features = {
            'rapid_movement': False,
            'multiple_faces': False
        }
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 1:
            features['multiple_faces'] = True
        
        # Motion detection using optical flow
        if self.prev_gray is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_magnitude = np.mean(magnitude)
            
            if avg_magnitude > 8.0:  # Threshold for rapid movement
                features['rapid_movement'] = True
            
            # Draw motion vectors (optional visualization)
            h, w = gray.shape
            step = 15
            for y in range(0, h, step):
                for x in range(0, w, step):
                    fx, fy = flow[y, x]
                    if abs(fx) > 1 or abs(fy) > 1:  # Only draw significant movement
                        cv2.arrowedLine(
                            frame, (x, y), (int(x + fx), int(y + fy)), 
                            (0, 255, 255), 1, tipLength=0.3
                        )
        
        self.prev_gray = gray
        return features, frame
    
    def predict_violence(self, frame):
        """Predict if frame contains violent activity"""
        features, processed_frame = self.detect_violence_features(frame)
        
        # Calculate violence score
        violence_score = 0
        if features['rapid_movement']:
            violence_score += 0.6
        if features['multiple_faces']:
            violence_score += 0.4
        
        is_violent = violence_score > 0.5
        
        return is_violent, violence_score, processed_frame

def main():
    print("Simple Violence Detection System")
    print("=" * 40)
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    # Initialize detector
    detector = SimpleViolenceDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect violence
        is_violent, score, processed_frame = detector.predict_violence(frame)
        
        # Display results
        if is_violent:
            color = (0, 0, 255)  # Red
            status = "VIOLENCE DETECTED!"
            border_color = (0, 0, 255)
        else:
            color = (0, 255, 0)  # Green
            status = "Normal"
            border_color = (0, 255, 0)
        
        # Add colored border based on detection
        cv2.rectangle(processed_frame, (0, 0), 
                     (processed_frame.shape[1], processed_frame.shape[0]), 
                     border_color, 10)
        
        # Add text information
        cv2.putText(processed_frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(processed_frame, f"Violence Score: {score:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(processed_frame, "Press 'Q' to quit, 'S' for screenshot", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Simple Violence Detection', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f"violence_screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Screenshot saved as {filename}")
            screenshot_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped. Thank you for using Violence Detection System!")

if __name__ == "__main__":
    main()