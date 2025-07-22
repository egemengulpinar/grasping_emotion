import cv2
import mediapipe as mp
import numpy as np
import time
import argparse

class HandDetector:
    def __init__(self, static_mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """
        Initialize the hand detector
        
        Args:
            static_mode: If True, detection runs on every frame
            max_hands: Maximum number of hands to detect
            detection_con: Minimum detection confidence
            track_con: Minimum tracking confidence
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Fingertip landmark indices
        self.fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        
    def find_hands(self, img, draw=True, flip_type=True):
        """
        Find hands in the image
        
        Args:
            img: Image to process
            draw: Whether to draw landmarks and connections
            flip_type: Whether to flip the image horizontally
            
        Returns:
            img: Processed image
            results: MediaPipe hand detection results
        """
        # Flip the image horizontally for a later selfie-view display
        if flip_type:
            img = cv2.flip(img, 1)
            
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if detected
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return img
    
    def find_positions(self, img, hand_no=0):
        """
        Find the position of hand landmarks
        
        Args:
            img: Image to process
            hand_no: Hand number to get positions for
            
        Returns:
            landmark_list: List of landmark positions [id, x, y]
        """
        landmark_list = []
        h, w, c = img.shape
        
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(my_hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                    
        return landmark_list
    
    def find_hand_type(self, hand_no=0):
        """
        Determine if hand is left or right
        
        Args:
            hand_no: Hand number to identify
            
        Returns:
            hand_type: "Left" or "Right"
        """
        hand_type = None
        if self.results.multi_handedness:
            if len(self.results.multi_handedness) > hand_no:
                hand_type = self.results.multi_handedness[hand_no].classification[0].label
        return hand_type
    
    def count_fingers_up(self, landmark_list, hand_type="Right"):
        """
        Count how many fingers are up
        
        Args:
            landmark_list: List of landmark positions
            hand_type: "Left" or "Right"
            
        Returns:
            fingers: List of which fingers are up (1) or down (0)
            count: Total number of fingers up
        """
        if not landmark_list:
            return [0, 0, 0, 0, 0], 0
            
        fingers = []
        
        # Thumb (different rule based on hand type)
        if hand_type == "Right":
            # For right hand, thumb is up if it's to the left of thumb base
            if landmark_list[self.fingertips[0]][1] < landmark_list[self.fingertips[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # For left hand, thumb is up if it's to the right of thumb base
            if landmark_list[self.fingertips[0]][1] > landmark_list[self.fingertips[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Other 4 fingers
        for id in range(1, 5):
            if landmark_list[self.fingertips[id]][2] < landmark_list[self.fingertips[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers, fingers.count(1)
    
    def get_hand_gesture(self, fingers):
        """
        Identify basic hand gestures based on fingers up
        
        Args:
            fingers: List of which fingers are up (1) or down (0)
            
        Returns:
            gesture: Name of recognized gesture
        """
        # All fingers down - Fist
        if fingers == [0, 0, 0, 0, 0]:
            return "Fist"
        
        # All fingers up - Open Hand
        elif fingers == [1, 1, 1, 1, 1]:
            return "Open Hand"
        
        # Index finger only - Pointing
        elif fingers == [0, 1, 0, 0, 0]:
            return "Pointing"
        
        # Thumb and pinky only - "Call me"
        elif fingers == [1, 0, 0, 0, 1]:
            return "Call me"
        
        # Thumb, index, pinky up - "Rock on"
        elif fingers == [1, 1, 0, 0, 1]:
            return "Rock on"
        
        # Thumb up
        elif fingers == [1, 0, 0, 0, 0]:
            return "Thumbs up"
            
        # Thumb down
        elif fingers == [0, 0, 0, 0, 0] and fingers[0] == -1:  # Special case
            return "Thumbs down"
            
        # Victory/Peace
        elif fingers == [0, 1, 1, 0, 0]:
            return "Peace"
            
        # OK sign (custom detection would be better)
        elif fingers == [1, 1, 0, 0, 0]:
            return "OK"
            
        else:
            return "Unknown"

def process_video(source, detector, output=None, display=True):
    """
    Process video from a file or camera
    
    Args:
        source: Camera index (int) or video file path (str)
        detector: HandDetector instance
        output: Path to save output video (optional)
        display: Whether to display the processed video
        
    Returns:
        None
    """
    # Initialize video capture
    if isinstance(source, int) or source.isdigit():
        cap = cv2.VideoCapture(int(source))
        source_name = f"Camera {source}"
    else:
        cap = cv2.VideoCapture(source)
        source_name = source.split('/')[-1]
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output specified
    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
    
    # Process video frames
    prev_time = 0
    while cap.isOpened():
        # Read frame
        success, frame = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break
        
        # Find hands
        frame = detector.find_hands(frame)
        
        # Process both hands
        for hand_no in range(2):  # Process up to 2 hands
            # Get landmark positions for this hand
            landmark_list = detector.find_positions(frame, hand_no)
            
            if landmark_list:
                # Get hand type
                hand_type = detector.find_hand_type(hand_no)
                
                # Count fingers up
                fingers, count = detector.count_fingers_up(landmark_list, hand_type)
                
                # Get hand gesture
                gesture = detector.get_hand_gesture(fingers)
                
                # Draw rectangle for hand info
                cv2.rectangle(frame, (20, 20 + 150 * hand_no), (300, 150 + 150 * hand_no), (0, 255, 0), 2)
                
                # Display hand information
                cv2.putText(frame, f"Hand {hand_no+1} ({hand_type})", (30, 50 + 150 * hand_no), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.putText(frame, f"Gesture: {gesture}", (30, 80 + 150 * hand_no), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Display finger states
                finger_text = ", ".join([name for i, name in enumerate(detector.finger_names) if fingers[i] == 1])
                if not finger_text:
                    finger_text = "None"
                cv2.putText(frame, f"Fingers up: {finger_text}", (30, 110 + 150 * hand_no), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                cv2.putText(frame, f"Count: {count}", (30, 140 + 150 * hand_no), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (width - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display source name
        cv2.putText(frame, f"Source: {source_name}", (width - 350, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write frame to output if specified
        if writer:
            writer.write(frame)
        
        # Display frame if needed
        if display:
            cv2.imshow("Hand Tracking", frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hand tracking and gesture recognition')
    parser.add_argument('--source', default='0', help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--output', default=None, help='Output video file path')
    parser.add_argument('--no-display', action='store_true', help='Do not display video')
    parser.add_argument('--max-hands', type=int, default=2, help='Maximum number of hands to detect')
    args = parser.parse_args()
    
    # Initialize hand detector
    detector = HandDetector(max_hands=args.max_hands)
    
    # Process video
    process_video(args.source, detector, args.output, not args.no_display) 