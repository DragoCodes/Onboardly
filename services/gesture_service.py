import cv2
import mediapipe as mp
import random
import face_recognition
from typing import List, Dict

mp_hands = mp.solutions.hands

class GestureService:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def detect_number_gesture(self,hand_landmarks):
        """
        Detect which number (0-9) is being shown by the hand
        This is a simplified implementation - a production system would need more robust detection
        """
        # Get landmark positions
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        
        # Check if fingers are extended
        # For simplicity, we'll just check if the tip of each finger is higher than its base
        # In a real system, this would be much more sophisticated
        
        # Thumb extended?
        thumb_extended = landmarks[4][0] < landmarks[3][0]
        
        # Index finger extended?
        index_extended = landmarks[8][1] < landmarks[5][1]
        
        # Middle finger extended?
        middle_extended = landmarks[12][1] < landmarks[9][1]
        
        # Ring finger extended?
        ring_extended = landmarks[16][1] < landmarks[13][1]
        
        # Pinky extended?
        pinky_extended = landmarks[20][1] < landmarks[17][1]
        
        # Count extended fingers (excluding thumb for numbers 1-5)
        extended_fingers = sum([index_extended, middle_extended, ring_extended, pinky_extended])
        
        # Simple gesture recognition
        if not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            return 0
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 1
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return 2
        elif index_extended and middle_extended and ring_extended and not pinky_extended:
            return 3
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return 4
        elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return 5
        
        # More complex gestures for 6-9 would go here
        # For simplicity, we'll just return None for unrecognized gestures
        return None

    def extract_gesture_sequence(self,gesture_detections):
        """
        Extract the sequence of gestures from frame-by-frame detections
        This handles cases where the same number is shown for multiple frames
        """
        if not gesture_detections:
            return []
        
        sequence = []
        last_number = None
        
        for detection in sorted(gesture_detections, key=lambda x: x["frame"]):
            if detection["number"] != last_number:
                sequence.append(detection["number"])
                last_number = detection["number"]
        
        return sequence

    
    def process_video(self, video_path: str, expected_gestures: List[int]):
        """
        Process video to detect hand gestures
        expected_gestures: List of integers representing the expected sequence
        """
        try:
            # Results container
            gesture_results = {
                "frames_processed": 0,
                "gestures_detected": [],
                "success": False,
                "confidence": 0.0,
                "face_frames": []
            }
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            # Initialize hand detector
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:
                
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Sample frames for face verification (5-10 frames)
                face_frame_indices = sorted(random.sample(range(total_frames), min(10, total_frames)))
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Save certain frames for face verification
                    if frame_count in face_frame_indices:
                        # Convert BGR to RGB for face_recognition
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Save frame for later face comparison
                        gesture_results["face_frames"].append(rgb_frame)
                    
                    # Process every 5th frame for gestures to improve performance
                    if frame_count % 5 == 0:
                        # Convert BGR to RGB
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Flip the image horizontally for a later selfie-view display
                        image = cv2.flip(image, 1)
                        
                        # To improve performance
                        image.flags.writeable = False
                        results = hands.process(image)
                        image.flags.writeable = True
                        
                        if results.multi_hand_landmarks:
                            # Get gesture number
                            detected_number = self.detect_number_gesture(results.multi_hand_landmarks[0])
                            if detected_number is not None:
                                gesture_results["gestures_detected"].append({
                                    "frame": frame_count,
                                    "number": detected_number
                                })
                    
                    frame_count += 1
                
                gesture_results["frames_processed"] = frame_count
                
                # Check if the expected sequence was detected
                detected_sequence = self.extract_gesture_sequence(gesture_results["gestures_detected"])
                
                # Compare with expected sequence
                if detected_sequence == expected_gestures:
                    gesture_results["success"] = True
                    gesture_results["confidence"] = 100.0
                else:
                    # Calculate partial match
                    gesture_results["success"] = False
                    
                    # Simple metric: percentage of correct gestures in order
                    correct = 0
                    for i, num in enumerate(detected_sequence):
                        if i < len(expected_gestures) and num == expected_gestures[i]:
                            correct += 1
                    
                    if len(expected_gestures) > 0:
                        gesture_results["confidence"] = (correct / len(expected_gestures)) * 100
                    
                cap.release()
                return gesture_results
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {"error": str(e)}
        
    
    def verify_faces_from_video(video_frames, reference_face_encoding):
        """
        Verify faces from sampled video frames against reference face
        """
        results = {
            "frames_checked": len(video_frames),
            "matches": 0,
            "match_percentage": 0.0,
            "overall_match": False
        }
        
        if not video_frames:
            return results
        
        match_count = 0
        total_similarity = 0.0
        
        for frame in video_frames:
            # Extract face from frame
            face_locations = face_recognition.face_locations(frame)
            
            if not face_locations:
                continue
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            if not face_encodings:
                continue
            
            # Compare with reference face
            face_distances = face_recognition.face_distance([reference_face_encoding], face_encodings[0])
            similarity = (1.0 - face_distances[0]) * 100
            total_similarity += similarity
            
            if similarity > 70:  # 70% threshold for a match
                match_count += 1
        
        # Calculate results
        if results["frames_checked"] > 0:
            results["matches"] = match_count
            results["match_percentage"] = (match_count / results["frames_checked"]) * 100
            results["avg_similarity"] = total_similarity / results["frames_checked"] if results["frames_checked"] > 0 else 0
            results["overall_match"] = results["match_percentage"] > 70  # 70% threshold
        
        return results
