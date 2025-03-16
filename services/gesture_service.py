import base64
import io
import random
import time
from typing import Any, Dict, List, Optional

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
from PIL import Image

# Import FaceService
from services.face_service import FaceService

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class GestureService:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )
        # Initialize FaceService
        self.face_service = FaceService()

    def detect_number_gesture(self, hand_landmarks) -> Optional[int]:
        """
        Detect which number (1-7) is being shown by the hand
        With custom gestures:
        - 5: All fingers extended (improved detection)
        - 6: Fist (no fingers extended)
        - 7: Only thumb and pinky extended
        """
        # Get landmark positions
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

        # For better 5 detection, we'll use more robust checks

        # Define finger base and tip indices
        finger_indices = {
            "thumb": {"base": 2, "tip": 4},
            "index": {"base": 5, "tip": 8},
            "middle": {"base": 9, "tip": 12},
            "ring": {"base": 13, "tip": 16},
            "pinky": {"base": 17, "tip": 20},
        }

        # Check if fingers are extended using distance and angle checks
        finger_extended = {}

        # Special check for thumb (different orientation)
        thumb_tip_x = landmarks[finger_indices["thumb"]["tip"]][0]
        thumb_base_x = landmarks[finger_indices["thumb"]["base"]][0]
        wrist_x = landmarks[0][0]

        # Thumb is extended if tip is further from wrist than base (in x-coordinate)
        # This works better when palm is facing the camera
        if abs(thumb_tip_x - wrist_x) > abs(thumb_base_x - wrist_x):
            finger_extended["thumb"] = True
        else:
            finger_extended["thumb"] = False

        # Check other fingers
        for finger in ["index", "middle", "ring", "pinky"]:
            base_idx = finger_indices[finger]["base"]
            tip_idx = finger_indices[finger]["tip"]

            # Finger is extended if tip is higher than base (lower y value)
            if landmarks[tip_idx][1] < landmarks[base_idx][1]:
                finger_extended[finger] = True
            else:
                finger_extended[finger] = False

        # Enhanced detection for number 5
        # For number 5, we also check the spread between fingers
        if all(finger_extended.values()):
            # Calculate distances between adjacent fingertips
            thumb_tip = np.array([landmarks[4][0], landmarks[4][1]])
            index_tip = np.array([landmarks[8][0], landmarks[8][1]])
            middle_tip = np.array([landmarks[12][0], landmarks[12][1]])
            ring_tip = np.array([landmarks[16][0], landmarks[16][1]])
            pinky_tip = np.array([landmarks[20][0], landmarks[20][1]])

            # Check if fingers are spread apart (not touching)
            thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
            index_middle_dist = np.linalg.norm(index_tip - middle_tip)
            middle_ring_dist = np.linalg.norm(middle_tip - ring_tip)
            ring_pinky_dist = np.linalg.norm(ring_tip - pinky_tip)

            # If all distances are above a threshold, fingers are spread
            min_spread_distance = 0.03  # Adjust based on testing
            fingers_spread = all(
                [
                    thumb_index_dist > min_spread_distance,
                    index_middle_dist > min_spread_distance,
                    middle_ring_dist > min_spread_distance,
                    ring_pinky_dist > min_spread_distance,
                ]
            )

            if fingers_spread:
                return 5

        # Detection logic for other numbers (1-4, 6-7)
        if (
            finger_extended["index"]
            and not finger_extended["middle"]
            and not finger_extended["ring"]
            and not finger_extended["pinky"]
        ):
            return 1
        elif (
            finger_extended["index"]
            and finger_extended["middle"]
            and not finger_extended["ring"]
            and not finger_extended["pinky"]
        ):
            return 2
        elif (
            finger_extended["index"]
            and finger_extended["middle"]
            and finger_extended["ring"]
            and not finger_extended["pinky"]
        ):
            return 3
        elif (
            finger_extended["index"]
            and finger_extended["middle"]
            and finger_extended["ring"]
            and finger_extended["pinky"]
            and not finger_extended["thumb"]
        ):
            return 4
        elif not any(finger_extended.values()):
            # Fist - no fingers extended
            return 6
        elif (
            finger_extended["thumb"]
            and not finger_extended["index"]
            and not finger_extended["middle"]
            and not finger_extended["ring"]
            and finger_extended["pinky"]
        ):
            # Only thumb and pinky extended
            return 7

        # If no match found or not confident enough
        return None

    def generate_random_digits(self, count: int = 4) -> List[int]:
        """Generate random sequence of digits from 1-7"""
        return random.choices(range(1, 8), k=count)

    def _encode_image_to_base64(self, frame):
        """Convert OpenCV image to base64 string for JSON serialization"""
        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Create a bytes buffer
        buffer = io.BytesIO()

        # Save the image to the buffer in JPEG format with compression
        pil_image.save(buffer, format="JPEG", quality=75)

        # Get the bytes from the buffer and encode to base64
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_str

    def verify_liveness(self, reference_face_encoding) -> Dict[str, Any]:
        """Perform liveness check with random number sequence and face matching"""
        # Generate random digit sequence
        digit_sequence = self.generate_random_digits(4)

        # Initialize results
        results = {
            "expected_sequence": digit_sequence,
            "detected_sequence": [],
            "success": False,
            "face_matches": [],
            "face_match_timestamps": [],
            "overall_face_match": False,
            "face_images": [],  # Store base64 encoded images instead of raw frames
        }

        print(
            f"\nLiveness Check: Please show these numbers with your hand: {' '.join(map(str, digit_sequence))}"
        )
        print("Instructions:")
        print("1: Show INDEX finger only")
        print("2: Show INDEX and MIDDLE fingers")
        print("3: Show INDEX, MIDDLE, and RING fingers")
        print("4: Show INDEX, MIDDLE, RING, and PINKY fingers")
        print("5: Show ALL FIVE fingers including THUMB")
        print("6: Fist")
        print("7: Show THUMB and Pinky only")

        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            results["error"] = "Cannot open webcam"
            return results

        # Variables for tracking gesture detection
        current_digit_index = 0
        current_digit = digit_sequence[current_digit_index]
        stable_detection_frames = 0
        last_detected_number = None

        # Variables for face verification
        total_face_frames = 0
        total_face_matches = 0
        face_frame_counter = 0
        face_check_frequency = 15  # Check face every N frames
        max_stored_images = 10  # Maximum number of face images to store

        # Main capture loop
        with mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        ) as hands:
            instruction_text = f"Show digit {current_digit} with your hand"
            feedback_text = ""
            face_feedback_text = ""

            start_time = time.time()

            while cap.isOpened() and current_digit_index < len(digit_sequence):
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror frame for more intuitive experience
                frame = cv2.flip(frame, 1)
                original_frame = frame.copy()

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame for hand landmarks
                hand_results = hands.process(rgb_frame)

                # Check face every N frames
                face_frame_counter += 1
                current_time = time.time() - start_time

                if face_frame_counter >= face_check_frequency:
                    face_frame_counter = 0

                    # Convert to RGB for face_recognition
                    face_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

                    # Detect faces
                    face_locations = face_recognition.face_locations(face_rgb)
                    if face_locations:
                        # Get face encoding
                        face_encodings = face_recognition.face_encodings(
                            face_rgb, face_locations
                        )
                        if face_encodings:
                            # Compare with reference face
                            comparison = self.face_service.compare_faces(
                                reference_face_encoding, face_encodings[0]
                            )

                            if "error" not in comparison:
                                match_result = {
                                    "similarity": comparison["similarity"],
                                    "match": comparison["match"],
                                    "timestamp": current_time,
                                    "gesture_index": current_digit_index,
                                }

                                results["face_matches"].append(match_result)
                                results["face_match_timestamps"].append(current_time)

                                if comparison["match"]:
                                    total_face_matches += 1
                                    face_feedback_text = (
                                        f"Face match: {comparison['similarity']:.2f}"
                                    )
                                else:
                                    face_feedback_text = f"Face not recognized: {comparison['similarity']:.2f}"

                                total_face_frames += 1

                                # Store face image (only store up to max_stored_images)
                                if len(results["face_images"]) < max_stored_images:
                                    # Draw rectangle around face
                                    top, right, bottom, left = face_locations[0]
                                    cv2.rectangle(
                                        original_frame,
                                        (left, top),
                                        (right, bottom),
                                        (0, 255, 0),
                                        2,
                                    )

                                    # Convert frame to base64 for JSON serialization
                                    img_base64 = self._encode_image_to_base64(
                                        original_frame
                                    )

                                    # Store image with metadata
                                    face_img_data = {
                                        "image": img_base64,
                                        "timestamp": current_time,
                                        "digit_index": current_digit_index,
                                        "similarity": comparison["similarity"],
                                        "match": comparison["match"],
                                    }
                                    results["face_images"].append(face_img_data)
                            else:
                                face_feedback_text = "Face comparison error"
                        else:
                            face_feedback_text = "No face encoding"
                    else:
                        face_feedback_text = "No face detected"

                # Display instructions and feedback
                cv2.putText(
                    frame,
                    instruction_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    feedback_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    face_feedback_text,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 165, 0),  # Orange color
                    2,
                )

                # Progress indicator
                progress = f"Progress: {current_digit_index}/{len(digit_sequence)}"
                cv2.putText(
                    frame,
                    progress,
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

                # Process hand landmarks if detected
                detected_number = None
                if hand_results.multi_hand_landmarks:
                    # Draw hand landmarks
                    for hand_landmark in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmark, mp_hands.HAND_CONNECTIONS
                        )
                        detected_number = self.detect_number_gesture(hand_landmark)

                    # Display detected number
                    if detected_number is not None:
                        cv2.putText(
                            frame,
                            f"Detected: {detected_number}",
                            (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 255),
                            2,
                        )

                # Gesture detection logic
                if detected_number is not None:
                    # Check if detection is stable
                    if detected_number == last_detected_number:
                        stable_detection_frames += 1
                    else:
                        stable_detection_frames = 0

                    last_detected_number = detected_number

                    # If detection is stable for sufficient frames
                    if stable_detection_frames >= 15:  # About 0.5 seconds
                        if detected_number == current_digit:
                            # Correct gesture
                            feedback_text = f"Correct! {detected_number} recognized."
                            results["detected_sequence"].append(detected_number)

                            # Move to next digit
                            current_digit_index += 1

                            # Reset detection
                            stable_detection_frames = 0
                            last_detected_number = None

                            # Check if sequence complete
                            if current_digit_index < len(digit_sequence):
                                current_digit = digit_sequence[current_digit_index]
                                instruction_text = (
                                    f"Show digit {current_digit} with your hand"
                                )
                            else:
                                feedback_text = "All digits correctly entered!"
                                results["success"] = True
                                # Display success message for 2 seconds
                                cv2.putText(
                                    frame,
                                    "SEQUENCE COMPLETE!",
                                    (80, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    (0, 255, 0),
                                    3,
                                )
                                cv2.imshow("Liveness Check", frame)
                                cv2.waitKey(2000)
                        else:
                            # Wrong gesture
                            feedback_text = f"Not correct. Please show {current_digit}"
                            stable_detection_frames = 0

                # Display frame
                cv2.imshow("Liveness Check", frame)

                # Check for exit
                if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                    break

            # Clean up
            cap.release()
            cv2.destroyAllWindows()

        # Calculate overall face match statistics if any face checks were performed
        if total_face_frames > 0:
            # Calculate match percentage
            match_percentage = (total_face_matches / total_face_frames) * 100
            results["match_percentage"] = round(match_percentage, 2)
            results["total_face_checks"] = total_face_frames
            results["successful_face_matches"] = total_face_matches

            # Calculate average similarity from all matches
            if results["face_matches"]:
                total_similarity = sum(
                    match["similarity"] for match in results["face_matches"]
                )
                avg_similarity = total_similarity / len(results["face_matches"])
                results["average_similarity"] = round(avg_similarity, 2)

            # Determine overall match (requiring at least 50% match rate)
            results["overall_face_match"] = match_percentage >= 50
        else:
            results["overall_face_match"] = False
            results["match_percentage"] = 0
            results["average_similarity"] = 0

        del results["face_images"]

        return results
