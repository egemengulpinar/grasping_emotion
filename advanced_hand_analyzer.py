import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import argparse
from collections import Counter
import seaborn as sns
from pathlib import Path
import glob
from scipy import stats

class HandMovementAnalyzer:
    def __init__(self, detection_con=0.7, track_con=0.7, show_realtime=False):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Landmark indices
        self.fingertips = [4, 8, 12, 16, 20]
        self.wrist = 0
        self.middle_base = 9
        
        # Realtime visualization
        self.show_realtime = show_realtime
        self.trajectory_points = []
        self.grasping_events = []
        
        # Video orientation tracking
        self.video_orientation = "landscape"  # "landscape" or "portrait"
        self.frame_width = 0
        self.frame_height = 0
        
        # Enhanced gesture thresholds
        self.gesture_thresholds = {
            'portrait': {
                'grasping_relaxed': 0.15,      # More relaxed for portrait
                'grasping_close': 0.10,
                'peace_sign_distance': 0.08,   # Stricter peace sign
                'peace_sign_angle_tolerance': 25,  # Degrees
                'fist_openness': 0.20,
                'open_hand_openness': 0.85
            },
            'landscape': {
                'grasping_relaxed': 0.12,
                'grasping_close': 0.08,
                'peace_sign_distance': 0.06,
                'peace_sign_angle_tolerance': 20,
                'fist_openness': 0.25,
                'open_hand_openness': 0.85
            }
        }
        
        # Results storage
        self.results = {
            'frame_data': [],
            'summary': {},
            'trajectory': []
        }
        
        # Survey data integration
        self.survey_data = None
        self.load_survey_data()
        
    def load_survey_data(self):
        """Load survey data for participant analysis"""
        try:
            survey_path = "survey results.csv"
            if os.path.exists(survey_path):
                self.survey_data = pd.read_csv(survey_path)
                print("✅ Survey data loaded successfully")
                
                # Create object mapping
                self.object_mapping = {
                    'toy': 'Dog Toy',
                    'donut': 'Donut', 
                    'pig': 'Plush',
                    'spider': 'Spider',
                    'box': 'Cube'
                }
            else:
                print("⚠️ Survey data not found")
        except Exception as e:
            print(f"❌ Error loading survey data: {e}")
            self.survey_data = None
    
    def get_participant_info(self, video_label):
        """Extract participant ID and experiment type from video label or path"""
        try:
            # Extract ID from video label (e.g., "ID4/toy" -> "ID04")
            if '/' in video_label:
                folder_part = video_label.split('/')[0]
            else:
                folder_part = video_label
            
            # Extract numeric part and format as ID
            import re
            match = re.search(r'ID(\d+)', folder_part, re.IGNORECASE)
            if match:
                id_num = int(match.group(1))
                participant_id = f"ID{id_num:02d}"  # Format as ID01, ID02, etc.
                
                # Determine experiment type based on ID
                if id_num <= 10:
                    exp_type = "SEE"
                else:
                    exp_type = "BLIND"
                    
                return participant_id, exp_type
        except Exception as e:
            print(f"⚠️ Error extracting participant info from {video_label}: {e}")
        
        return None, None
    
    def get_survey_data_for_video(self, video_label):
        """Get survey data for specific video and participant"""
        if self.survey_data is None:
            return None
        
        try:
            participant_id, exp_type = self.get_participant_info(video_label)
            if not participant_id:
                return None
            
            # Extract object name from video label
            object_name = video_label.split('/')[-1] if '/' in video_label else video_label
            survey_object = self.object_mapping.get(object_name.lower())
            
            if not survey_object:
                return None
            
            # Find matching survey entry
            survey_entry = self.survey_data[
                (self.survey_data['Participant ID'] == participant_id) &
                (self.survey_data['Object'] == survey_object)
            ]
            
            if not survey_entry.empty:
                return {
                    'participant_id': participant_id,
                    'experiment_type': exp_type,
                    'object': survey_object,
                    'emotion_felt': survey_entry.iloc[0]['Emotion Felt'],
                    'intensity': survey_entry.iloc[0]['Intensity'],
                    'familiarity': survey_entry.iloc[0]['Familiarity'],
                    'comfort_during_grasping': survey_entry.iloc[0]['Comfort During Grasping'],
                    'would_react_same_way': survey_entry.iloc[0]['Would React Same Way Again']
                }
        except Exception as e:
            print(f"⚠️ Error getting survey data for {video_label}: {e}")
        
        return None
    
    def calculate_hand_detection_time(self, trajectory_data, fps):
        """Calculate the time range when hand was detected (start to end time)"""
        if not trajectory_data:
            return 0.0
        
        # Get first and last frame times
        start_time = trajectory_data[0]['time']
        end_time = trajectory_data[-1]['time']
        
        return end_time - start_time
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def get_hand_center(self, landmarks):
        """Calculate hand center using wrist and middle finger base"""
        wrist = landmarks[self.wrist]
        middle_base = landmarks[self.middle_base]
        center_x = (wrist.x + middle_base.x) / 2
        center_y = (wrist.y + middle_base.y) / 2
        return center_x, center_y
    
    def calculate_hand_openness(self, landmarks):
        """Calculate hand openness as ratio of finger extension"""
        thumb_tip = landmarks[4]
        thumb_base = landmarks[1]
        
        openness_scores = []
        
        # Calculate for each finger
        finger_pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]  # tip, base pairs
        
        for tip_idx, base_idx in finger_pairs:
            tip = landmarks[tip_idx]
            base = landmarks[base_idx]
            finger_length = self.calculate_distance(tip, base)
            openness_scores.append(finger_length)
        
        # Normalize to 0-1 scale
        avg_openness = np.mean(openness_scores)
        return min(1.0, avg_openness * 5)  # Scaling factor
    
    def detect_video_orientation(self, frame):
        """Detect if video is portrait or landscape"""
        height, width = frame.shape[:2]
        self.frame_width = width
        self.frame_height = height
        
        if height > width:
            self.video_orientation = "portrait"
        else:
            self.video_orientation = "landscape"
        
        return self.video_orientation

    def calculate_finger_angle(self, p1, p2, p3):
        """Calculate angle between three points (p1-p2-p3)"""
        import math
        
        # Vector from p2 to p1
        v1 = [p1.x - p2.x, p1.y - p2.y]
        # Vector from p2 to p3  
        v2 = [p3.x - p2.x, p3.y - p2.y]
        
        # Calculate angle
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag_v1 == 0 or mag_v2 == 0:
            return 0
        
        cos_angle = dot_product / (mag_v1 * mag_v2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        
        angle = math.acos(cos_angle)
        return math.degrees(angle)

    def is_finger_extended(self, landmarks, finger_indices):
        """Check if finger is extended based on joint positions"""
        tip, pip, mcp = finger_indices
        
        # For thumb, use different logic
        if finger_indices[0] == 4:  # Thumb
            return landmarks[tip].x > landmarks[pip].x  # Simple x-position check
        
        # For other fingers, check if tip is above pip and pip is above mcp
        return (landmarks[tip].y < landmarks[pip].y and 
                landmarks[pip].y < landmarks[0].y + 0.1)  # Relative to wrist (index 0)

    def detect_improved_peace_sign(self, landmarks):
        """Improved peace sign detection with strict criteria"""
        thresholds = self.gesture_thresholds[self.video_orientation]
        
        # Check if only index and middle fingers are extended
        fingers_status = {
            'thumb': self.is_finger_extended(landmarks, [4, 3, 2]),
            'index': self.is_finger_extended(landmarks, [8, 6, 5]),
            'middle': self.is_finger_extended(landmarks, [12, 10, 9]),
            'ring': self.is_finger_extended(landmarks, [16, 14, 13]),
            'pinky': self.is_finger_extended(landmarks, [20, 18, 17])
        }
        
        # Peace sign: Only index and middle extended, others folded
        if not (fingers_status['index'] and fingers_status['middle']):
            return False
        
        if fingers_status['ring'] or fingers_status['pinky']:
            return False
        
        # Check distance between index and middle fingertips
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        distance = self.calculate_distance(index_tip, middle_tip)
        
        if distance < thresholds['peace_sign_distance']:
            return False  # Too close together
        
        # Check angle between fingers (should be roughly V-shaped)
        wrist = landmarks[0]
        angle = self.calculate_finger_angle(index_tip, wrist, middle_tip)
        
        # Peace sign should have fingers spread apart (15-45 degrees)
        if angle < 15 or angle > 45:
            return False
        
        return True

    def calculate_finger_curvature(self, landmarks):
        """Calculate finger curvature to detect holding vs grasping"""
        curvatures = []
        
        # Finger definitions: [tip, pip, mcp] for each finger
        fingers = {
            'index': [8, 6, 5],
            'middle': [12, 10, 9], 
            'ring': [16, 14, 13],
            'pinky': [20, 18, 17]
        }
        
        for finger_name, (tip, pip, mcp) in fingers.items():
            # Calculate angle at PIP joint
            angle = self.calculate_finger_angle(
                landmarks[tip], landmarks[pip], landmarks[mcp]
            )
            
            # Curvature = how much finger is bent (180° = straight, smaller = more bent)
            curvature = 180 - angle
            curvatures.append(curvature)
        
        return curvatures
    
    def detect_holding_vs_grasping(self, landmarks):
        """
        Enhanced detection: Distinguish between holding and grasping (2 categories only)
        Holding: Object manipulation with moderate grip
        Grasping: Strong pressure contact, tight grip
        """
        thresholds = self.gesture_thresholds[self.video_orientation]
        
        thumb_tip = landmarks[4]
        finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky
        
        # Calculate finger-thumb distances
        finger_distances = []
        for finger_tip_idx in finger_tips:
            finger_tip = landmarks[finger_tip_idx]
            distance = self.calculate_distance(thumb_tip, finger_tip)
            finger_distances.append(distance)
        
        # Calculate finger curvatures
        curvatures = self.calculate_finger_curvature(landmarks)
        avg_curvature = np.mean(curvatures)
        
        # Hand openness
        hand_openness = self.calculate_hand_openness(landmarks)
        
        # Enhanced criteria for holding/grasping
        avg_distance = np.mean(finger_distances)
        min_distance = min(finger_distances)
        
        # Orientation-adjusted thresholds
        if self.video_orientation == "portrait":
            holding_distance_threshold = 0.16    # Relaxed threshold for holding
            grasping_distance_threshold = 0.11   # Stricter for strong grasping
            curvature_threshold = 15             # Minimum bend for any interaction
            openness_holding_max = 0.75          # Max openness for holding
            openness_grasping_max = 0.6          # Max openness for grasping
        else:
            holding_distance_threshold = 0.14
            grasping_distance_threshold = 0.09
            curvature_threshold = 20
            openness_holding_max = 0.70
            openness_grasping_max = 0.55
        
        # Count fingers in different ranges
        holding_fingers = sum(1 for d in finger_distances if d < holding_distance_threshold)
        grasping_fingers = sum(1 for d in finger_distances if d < grasping_distance_threshold)
        
        # Detection logic with only 2 categories
        
        # Strong grasping (tight grip with high pressure)
        is_strong_grasping = (
            grasping_fingers >= 3 and  # At least 3 fingers very close
            hand_openness < openness_grasping_max and
            avg_curvature > curvature_threshold and
            avg_distance < grasping_distance_threshold
        )
        
        # Holding detection (moderate grip, object manipulation)
        is_holding = (
            not is_strong_grasping and  # Not already classified as strong grasping
            holding_fingers >= 2 and    # At least 2 fingers in holding range
            hand_openness < openness_holding_max and  # Hand not too open
            avg_curvature > curvature_threshold  # Fingers have some curve
        )
        
        # Final classification - only 2 categories
        if is_strong_grasping:
            is_grasping = True
            grasp_type = "Grasping"
            # High strength for strong grasping
            grasp_strength = min(1.0, (1 - avg_distance / holding_distance_threshold) * 0.8 + 0.3)
        elif is_holding:
            is_grasping = True  # Consider holding as grasping for analysis
            grasp_type = "Holding"
            # Moderate strength for holding
            grasp_strength = min(1.0, (1 - avg_distance / holding_distance_threshold) * 0.6 + 0.2)
        else:
            is_grasping = False
            grasp_type = "None"
            grasp_strength = 0.0
        
        return is_grasping, avg_distance, min_distance, grasp_strength, holding_fingers, grasp_type

    def detect_grasping(self, landmarks):
        """
        🆕 ENHANCED GRASPING DETECTION: Now includes holding detection
        References: MDPI Sensors 2024, Scientific Reports 2024
        """
        return self.detect_holding_vs_grasping(landmarks)

    def detect_gesture(self, landmarks):
        """Enhanced gesture detection with 3-category movement system: Grasping, Holding, Other"""
        thresholds = self.gesture_thresholds[self.video_orientation]
        
        hand_openness = self.calculate_hand_openness(landmarks)
        is_grasping, grasp_avg_dist, grasp_min_dist, grasp_strength, close_fingers, grasp_type = self.detect_grasping(landmarks)
        
        # 3-CATEGORY SYSTEM: Grasping, Holding, Other
        # Priority-based gesture classification
        if is_grasping and grasp_strength > 0.2:  # Any type of grasping/holding
            if grasp_type == "Grasping":
                return "Grasping"  # Category 1: Strong pressure, tight grip
            elif grasp_type == "Holding":
                return "Holding"   # Category 2: Object manipulation, moderate grip
        
        # Category 3: Other (all non-grasping/holding movements)
        # This includes: Fist, Peace Sign, Open Hand, Pointing, Thumbs Up, etc.
        return "Other"
    
    def calculate_velocity(self, current_center, previous_center, time_diff):
        """Calculate velocity between two points"""
        if previous_center is None or time_diff == 0:
            return 0
        
        distance = np.sqrt((current_center[0] - previous_center[0])**2 + 
                          (current_center[1] - previous_center[1])**2)
        return distance / time_diff
    
    def process_video(self, video_path, show_video=False, video_label="Video"):
        """Enhanced video processing with orientation detection and improved analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        
        # Initialize tracking variables
        frame_count = 0
        previous_center = None
        trajectory_data = []
        grasping_events = []
        
        # Reset trajectory points for real-time visualization
        self.trajectory_points = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps
            
            # Detect video orientation on first frame
            if frame_count == 1:
                orientation = self.detect_video_orientation(frame)
                print(f"Video orientation detected: {orientation}")
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Calculate hand center (enhanced)
                    wrist = hand_landmarks.landmark[self.wrist]
                    middle_base = hand_landmarks.landmark[self.middle_base]
                    current_center = (
                        (wrist.x + middle_base.x) / 2,
                        (wrist.y + middle_base.y) / 2
                    )
                    
                    # Convert to pixel coordinates for real-time display
                    h, w = frame.shape[:2]
                    center_px = (int(current_center[0] * w), int(current_center[1] * h))
                    
                    # Add to trajectory (limit to last 50 points for memory)
                    self.trajectory_points.append(center_px)
                    if len(self.trajectory_points) > 50:
                        self.trajectory_points.pop(0)
                    
                    # Calculate velocity
                    velocity = 0
                    if previous_center is not None:
                        velocity = self.calculate_distance(
                            type('obj', (object,), {'x': previous_center[0], 'y': previous_center[1]})(),
                            type('obj', (object,), {'x': current_center[0], 'y': current_center[1]})()
                        ) * fps  # Convert to per-second
                    
                    # Enhanced gesture and grasping detection
                    gesture = self.detect_gesture(hand_landmarks.landmark)
                    hand_openness = self.calculate_hand_openness(hand_landmarks.landmark)
                    
                    # Track grasping events with enhanced algorithm
                    is_grasping, grasp_avg_dist, grasp_min_dist, grasp_strength, close_fingers, grasp_type = self.detect_grasping(hand_landmarks.landmark)
                    if is_grasping:
                        grasping_events.append({
                            'frame': frame_count,
                            'time': current_time,
                            'avg_distance': grasp_avg_dist,
                            'min_distance': grasp_min_dist,
                            'grasp_strength': grasp_strength,
                            'close_fingers': close_fingers,
                            'grasp_type': grasp_type
                        })
                    
                    # Store trajectory data
                    trajectory_data.append({
                        'frame': frame_count,
                        'time': current_time,
                        'center_x': current_center[0],
                        'center_y': current_center[1],
                        'velocity': velocity,
                        'hand_openness': hand_openness,
                        'gesture': gesture,
                        'is_grasping': is_grasping,
                        'grasp_strength': grasp_strength,
                        'grasp_type': grasp_type,
                        'video_label': video_label
                    })
                    
                    # Real-time visualization with enhanced display
                    if show_video:
                        frame = self.draw_real_time_info(frame, hand_landmarks, center_px, velocity, gesture, is_grasping, grasp_strength, grasp_type)
                    
                    previous_center = current_center
            
            # Show frame if real-time mode
            if show_video:
                cv2.imshow('Hand Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:  # 'q' or ESC
                    break
        
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Calculate enhanced summary statistics
        if not trajectory_data:
            print(f"⚠️  No hand detected in {video_path}")
            return None
        
        df = pd.DataFrame(trajectory_data)
        
        # Calculate hand detection time (time when hand was actually detected)
        hand_detection_time = self.calculate_hand_detection_time(trajectory_data, fps)
        
        # Get survey data for this video
        survey_info = self.get_survey_data_for_video(video_label)
        
        # Enhanced summary with orientation info, frame data, and survey integration
        video_summary = {
            'video_label': video_label,
            'video_path': video_path,
            'orientation': self.video_orientation,
            'duration': duration,  # Total video duration
            'hand_detection_time': hand_detection_time,  # Time when hand was detected
            'total_frames': len(trajectory_data),
            'fps': fps,
            'avg_velocity': df['velocity'].mean(),
            'max_velocity': df['velocity'].max(),
            'total_path_length': df['velocity'].sum() / fps,
            'avg_hand_openness': df['hand_openness'].mean(),
            'gesture_counts': df['gesture'].value_counts().to_dict(),
            'most_common_gesture': [df['gesture'].mode().iloc[0], df['gesture'].value_counts().iloc[0]],
            'grasping_events_count': len(grasping_events),
            'grasping_frequency': len(grasping_events) / hand_detection_time if hand_detection_time > 0 else 0,
            'grasping_events': grasping_events[:10],  # Store first 10 events
            'frame_data': trajectory_data,  # Include frame data for analysis
            'survey_data': survey_info  # Survey information for this participant/object
        }
        
        print(f"Video {video_label} processed successfully!")
        print(f"Orientation: {self.video_orientation}")
        print(f"Grasping events detected: {len(grasping_events)}")
        print(f"Grasping frequency: {video_summary['grasping_frequency']:.2f} events/second")
        
        return video_summary
    
    def draw_real_time_info(self, frame, hand_landmarks, center, velocity, gesture, is_grasping, grasp_strength, grasp_type=""):
        """Enhanced real-time info display with holding/grasping distinction"""
        if hand_landmarks is None:
            return frame
        
        # Draw hand landmarks with enhanced visibility
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Draw hand center with orientation-adjusted size
        center_size = 8 if self.video_orientation == "portrait" else 6
        cv2.circle(frame, center, center_size, (0, 255, 0), -1)
        
        # Add trajectory trail (last 20 points)
        if len(self.trajectory_points) > 1:
            for i in range(1, min(len(self.trajectory_points), 20)):
                alpha = i / 20.0
                color_intensity = int(255 * alpha)
                cv2.circle(frame, self.trajectory_points[-i], 2, (0, color_intensity, 0), -1)
        
        # Enhanced grasping indication with type-specific colors
        if is_grasping:
            grasp_size = 12 if self.video_orientation == "portrait" else 10
            if grasp_type == "Grasping":
                color = (0, 0, 255)  # Red for strong grasping
            else:
                color = (0, 165, 255)  # Orange for holding
            cv2.circle(frame, center, grasp_size, color, 3)
        
        # Enhanced text display with orientation-adjusted positioning and size
        font_scale = 0.8 if self.video_orientation == "portrait" else 0.6
        thickness = 2 if self.video_orientation == "portrait" else 1
        
        # Position text based on orientation
        if self.video_orientation == "portrait":
            text_start_y = 30
            text_spacing = 35
        else:
            text_start_y = 25
            text_spacing = 25
        
        # Display information with enhanced grasping details
        cv2.putText(frame, f"Gesture: {gesture}", (10, text_start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        cv2.putText(frame, f"Velocity: {velocity:.2f}", (10, text_start_y + text_spacing), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        cv2.putText(frame, f"Grasping: {'YES' if is_grasping else 'NO'}", 
                   (10, text_start_y + 2*text_spacing), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
        
        # Add orientation indicator
        cv2.putText(frame, f"Mode: {self.video_orientation.title()}", 
                   (10, text_start_y + 3*text_spacing), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 255, 255), thickness)
        
        # Enhanced grasping info
        if is_grasping:
            cv2.putText(frame, f"Strength: {grasp_strength:.2f}", 
                       (10, text_start_y + 4*text_spacing), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (0, 0, 255), thickness)
            if grasp_type and grasp_type != "None":
                cv2.putText(frame, f"Type: {grasp_type}", 
                           (10, text_start_y + 5*text_spacing), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (255, 100, 0), thickness)
        
        return frame
    
    def process_folder(self, folder_path, show_video=False):
        """Process all videos in a folder with custom video labels from filenames"""
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return
        
        # Get all video files
        video_extensions = ['.mov', '.mp4', '.avi', '.mkv', '.wmv']
        video_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file)
        
        if not video_files:
            print(f"❌ No video files found in: {folder_path}")
            return
        
        # Sort videos by filename for consistent order
        video_files.sort()
        
        print(f"Found {len(video_files)} videos in folder")
        for i, video_file in enumerate(video_files, 1):
            # Extract label from filename (remove extension)
            video_label = os.path.splitext(video_file)[0]
            print(f"   {i}. {video_file} -> Label: {video_label}")
        
        summaries = []
        
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            # Use folder/filename format for participant ID extraction
            folder_name = os.path.basename(folder_path.rstrip('/'))  # Remove trailing slash
            file_name = os.path.splitext(video_file)[0]
            video_label = f"{folder_name}/{file_name}"
            
            print(f"\nProcessing video: {video_path}")
            print(f"Video label: {video_label}")
            
            summary = self.process_video(video_path, show_video=show_video, video_label=video_label)
            if summary:
                summaries.append(summary)
        
        if not summaries:
            print("❌ No videos were processed successfully!")
            return
        
        # Create comprehensive analysis
        print(f"\nCreating comprehensive analysis for {len(summaries)} videos...")
        
        # Extract folder name for output directory
        folder_name = os.path.basename(folder_path)
        output_dir = os.path.join('analysis_results', folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Output directory: {output_dir}")
        
        # Create comprehensive DataFrame
        all_data = []
        for summary in summaries:
            video_data = pd.DataFrame(summary['frame_data'])
            video_data['video_label'] = summary['video_label']
            all_data.append(video_data)
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Save comprehensive data
        csv_path = os.path.join(output_dir, 'detailed_analysis.csv')
        df.to_csv(csv_path, index=False)
        print(f"Detailed data saved: {csv_path}")
        
        # Create plots for comprehensive analysis
        self.create_main_analysis_plots(df, summaries, output_dir)
        self.create_grasping_analysis_plots(df, summaries, output_dir)
        self.create_temporal_analysis_plots(df, summaries, output_dir)
        self.create_summary_table_plot(summaries, output_dir)
        print(f"Enhanced analysis plots created in 4 separate figures!")
        
        # Save enhanced summary
        self.save_enhanced_summary(summaries, output_dir)
        
        # Create analysis report
        self.create_analysis_report(summaries, output_dir)
        
        # Create LLM-optimized analysis package
        llm_analysis_dir = self.create_llm_optimized_analysis(summaries, output_dir)
        
        print(f"Comprehensive analysis saved to: {output_dir}")
        
        total_grasping = sum(s['grasping_events_count'] for s in summaries)
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"Total grasping events detected: {total_grasping}")
        print(f"🤖 LLM-optimized analysis package: {llm_analysis_dir}")

    def create_main_analysis_plots(self, df, summaries, output_dir):
        """Figure 1: Main Analysis - Single Grasping Frequency Plot"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Enhanced color palette - More distinct and vibrant
        video_colors = {
            'toy': '#E74C3C',      # Bright Red
            'spider': '#8E44AD',   # Purple  
            'box': '#3498DB',      # Blue
            'donut': '#F39C12',    # Orange
            'pig': '#27AE60'       # Green
        }
        
        # Single Enhanced Grasping Frequency Comparison Plot
        video_labels = [s['video_label'] for s in summaries]
        grasping_frequencies = [s['grasping_frequency'] for s in summaries]
        bar_colors = [video_colors.get(label, 'gray') for label in video_labels]
        
        bars = ax.bar(range(len(video_labels)), grasping_frequencies, 
                      color=bar_colors, alpha=0.9, width=0.7, 
                      edgecolor='white', linewidth=3)
        
        ax.set_xlabel('Object Types', fontsize=18, fontweight='bold')
        ax.set_ylabel('Grasping Frequency (events/s)', fontsize=18, fontweight='bold')
        ax.set_title('Grasping Frequency by Object Type', fontsize=20, fontweight='bold', pad=25)
        ax.set_xticks(range(len(video_labels)))
        ax.set_xticklabels([label.title() for label in video_labels], 
                           rotation=0, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.4, axis='y', linewidth=1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add enhanced value labels on bars
        for bar, freq in zip(bars, grasping_frequencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(grasping_frequencies)*0.02,
                    f'{freq:.1f}', ha='center', va='bottom', fontweight='bold', 
                    fontsize=14, color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout(pad=3.0)
        plot_path = os.path.join(output_dir, '1_main_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Figure 1 - Main Analysis saved: {plot_path}")

    def create_grasping_analysis_plots(self, df, summaries, output_dir):
        """Figure 2: 3-Category Movement System Analysis (Grasping, Holding, Other)"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Enhanced color palette
        video_colors = {
            'toy': '#E74C3C',      # Bright Red
            'spider': '#8E44AD',   # Purple  
            'box': '#3498DB',      # Blue
            'donut': '#F39C12',    # Orange
            'pig': '#27AE60'       # Green
        }
        
        # 3-Category Movement System Analysis
        video_labels = [s['video_label'] for s in summaries]
        
        # Calculate 3-category frequencies per object 
        category_data = {}
        for summary in summaries:
            video_data = df[df['video_label'] == summary['video_label']]
            gesture_counts = video_data['gesture'].value_counts()
            
            category_data[summary['video_label']] = {
                'Grasping': gesture_counts.get('Grasping', 0),
                'Holding': gesture_counts.get('Holding', 0),
                'Other': gesture_counts.get('Other', 0)
            }
        
        # Create enhanced stacked bar chart with 3 categories
        grasping_counts = [category_data[label]['Grasping'] for label in video_labels]
        holding_counts = [category_data[label]['Holding'] for label in video_labels]
        other_counts = [category_data[label]['Other'] for label in video_labels]
        
        x_pos = range(len(video_labels))
        
        # 3-category stacked bars
        bars1 = ax.bar(x_pos, grasping_counts, label='Category 1: Grasping', 
                       color='#C0392B', alpha=0.9, width=0.7, edgecolor='white', linewidth=2)
        bars2 = ax.bar(x_pos, holding_counts, bottom=grasping_counts, label='Category 2: Holding', 
                       color='#E67E22', alpha=0.9, width=0.7, edgecolor='white', linewidth=2)
        bars3 = ax.bar(x_pos, other_counts, bottom=[g+h for g,h in zip(grasping_counts, holding_counts)], 
                       label='Category 3: Other', color='#95A5A6', alpha=0.9, width=0.7, edgecolor='white', linewidth=2)
        
        ax.set_xlabel('Object Types', fontsize=18, fontweight='bold')
        ax.set_ylabel('Movement Events Count', fontsize=18, fontweight='bold') 
        ax.set_title('3-Category Movement System by Object\n(Grasping, Holding, Other)', fontsize=20, fontweight='bold', pad=25)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([label.title() for label in video_labels], 
                           rotation=0, fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=14, framealpha=0.9, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.4, axis='y', linewidth=1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add enhanced value annotations on bars (only show if significant)
        for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            height3 = bar3.get_height()
            
            if height1 > 50:  # Show grasping count
                ax.text(bar1.get_x() + bar1.get_width()/2., height1/2, f'{int(height1)}',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=12)
            if height2 > 50:  # Show holding count
                ax.text(bar2.get_x() + bar2.get_width()/2., height1 + height2/2, f'{int(height2)}',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=12)
            if height3 > 100:  # Show other count (higher threshold as usually more)
                ax.text(bar3.get_x() + bar3.get_width()/2., height1 + height2 + height3/2, f'{int(height3)}',
                        ha='center', va='center', fontweight='bold', color='black', fontsize=12)
        
        plt.tight_layout(pad=3.0)
        plot_path = os.path.join(output_dir, '2_grasping_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Figure 2 - 3-Category Movement Analysis saved: {plot_path}")

    def create_temporal_analysis_plots(self, df, summaries, output_dir):
        """Figure 3: Enhanced Temporal Analysis - Temporal Grasping Patterns by Object (Single Large Plot)"""
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        
        # Enhanced color palette
        video_colors = {
            'toy': '#E74C3C',      # Bright Red
            'spider': '#8E44AD',   # Purple  
            'box': '#3498DB',      # Blue
            'donut': '#F39C12',    # Orange
            'pig': '#27AE60'       # Green
        }
        
        # Enhanced Temporal Pattern of Grasping Events by Object (Large Display)
        for i, summary in enumerate(summaries):
            video_data = df[df['video_label'] == summary['video_label']].copy()
            video_label = summary['video_label']
            color = video_colors.get(video_label, f'C{i}')
            
            # Create time-binned grasping activity
            if len(video_data) > 0:
                # Create 15 time bins for smooth analysis
                max_time = video_data['time'].max()
                time_bins = np.linspace(0, max_time, 16)
                video_data['time_bin'] = pd.cut(video_data['time'], bins=time_bins)
                
                # Calculate grasping rate per time bin
                binned_data = video_data.groupby('time_bin')['is_grasping'].agg(['mean', 'count']).reset_index()
                binned_data = binned_data[binned_data['count'] > 0]
                
                # Get bin centers for plotting
                bin_centers = [interval.mid for interval in binned_data['time_bin']]
                grasping_rates = binned_data['mean'].values
                
                ax.plot(bin_centers, grasping_rates, 'o-', linewidth=5, markersize=12,
                        label=f'{video_label.title()}', color=color, alpha=0.9,
                        markeredgecolor='white', markeredgewidth=3)
        
        ax.set_xlabel('Time (seconds)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Grasping Activity Rate', fontsize=20, fontweight='bold')
        ax.set_title('Temporal Grasping Patterns by Object', fontsize=24, fontweight='bold', pad=30)
        ax.legend(fontsize=18, framealpha=0.9, fancybox=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.4, linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Add enhanced styling for single large plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        plt.tight_layout(pad=3.0)
        plot_path = os.path.join(output_dir, '3_temporal_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Figure 3 - Enhanced Temporal Analysis (Large Display) saved: {plot_path}")

    def create_summary_table_plot(self, summaries, output_dir):
        """Figure 4: Enhanced Summary Statistics Table with scientific formatting"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.axis('off')
        
        # Enhanced color palette
        video_colors = {
            'toy': '#E74C3C',      # Bright Red
            'spider': '#8E44AD',   # Purple  
            'box': '#3498DB',      # Blue
            'donut': '#F39C12',    # Orange
            'pig': '#27AE60'       # Green
        }
        
        # Calculate performance metrics
        metrics_data = []
        for summary in summaries:
            # Calculate additional metrics
            avg_grasp_duration = 0
            if summary['grasping_events_count'] > 0:
                # Estimate average grasp duration
                total_grasp_time = summary['grasping_events_count'] / summary['grasping_frequency']
                avg_grasp_duration = total_grasp_time / summary['grasping_events_count'] * summary['fps']
            
            # Engagement level
            avg_freq = sum(s['grasping_frequency'] for s in summaries) / len(summaries)
            if summary['grasping_frequency'] > avg_freq * 1.2:
                engagement = "High"
            elif summary['grasping_frequency'] < avg_freq * 0.8:
                engagement = "Low"
            else:
                engagement = "Moderate"
            
            # Calculate total events (including other gestures beyond grasping)
            total_frames = summary.get('total_frames', 0)
            grasping_events = summary['grasping_events_count']
            
            # Use hand detection time instead of total duration
            hand_detection_time = summary.get('hand_detection_time', summary['duration'])
            
            metrics_data.append([
                summary['video_label'].title(),
                f"{hand_detection_time:.1f}s",
                f"{grasping_events}",
                f"{total_frames}",
                f"{summary['grasping_frequency']:.1f}/s",
                f"{summary['avg_velocity']:.2f}",
                f"{summary['avg_hand_openness']:.2f}",
                engagement
            ])
        
        # Create enhanced metrics table with hand detection time instead of duration
        headers = ['Object Type', 'Hand Detection\nTime', 'Grasping\nEvents', 'Total\nEvents',
                  'Frequency', 'Avg\nVelocity', 'Hand\nOpenness', 'Engagement']
        
        table = ax.table(cellText=metrics_data, colLabels=headers, 
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)  # Smaller font size to prevent overflow
        table.scale(1.2, 3.5)  # Better scaling for readability
        
        # Set column widths to prevent text overflow
        cellDict = table.get_celld()
        for i in range(len(headers)):
            for j in range(len(metrics_data) + 1):  # +1 for header
                cellDict[(j,i)].set_width(0.12)  # Equal width columns
        
        # Enhanced table styling
        for i, label in enumerate([s['video_label'] for s in summaries]):
            color = video_colors.get(label, 'lightgray')
            for j in range(len(headers)):
                cell = table[(i+1, j)]
                cell.set_facecolor(color)
                cell.set_alpha(0.3)
                cell.set_edgecolor('white')
                cell.set_linewidth(2)
                cell.set_text_props(weight='bold', fontsize=14)
        
        # Style header
        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white', fontsize=15)
            cell.set_edgecolor('white')
            cell.set_linewidth(2)
        
        ax.set_title('Enhanced Performance Metrics by Object Type\nComplete statistical summary of hand-object interactions', 
                    fontsize=20, fontweight='bold', pad=40)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, '4_summary_table.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Figure 4 - Enhanced Summary Table saved: {plot_path}")

    def create_analysis_report(self, summaries, output_dir):
        """Create enhanced text-based analysis report with survey integration and 3-category system"""
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED HAND MOVEMENT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overview
            f.write("Analysis Overview:\n")
            f.write(f"   • Total objects analyzed: {len(summaries)}\n")
            f.write(f"   • Total video duration: {sum(s['duration'] for s in summaries):.2f} seconds\n")
            f.write(f"   • Total hand detection time: {sum(s.get('hand_detection_time', s['duration']) for s in summaries):.2f} seconds\n")
            f.write(f"   • Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"   • Object types: {', '.join([s['video_label'].title() for s in summaries])}\n\n")
            
            # Enhanced grasping summary with 3-category system
            total_grasping = sum(s['grasping_events_count'] for s in summaries)
            avg_frequency = sum(s['grasping_frequency'] for s in summaries) / len(summaries)
            
            f.write("Enhanced Grasping Analysis:\n")
            f.write(f"   • Total grasping events: {total_grasping}\n")
            f.write(f"   • Average grasping frequency: {avg_frequency:.3f} events/second\n\n")
            
            # 3-Category Movement System
            f.write("Movement Category System (3 Categories - LLM Optimized):\n")
            f.write("   • Category 1 - Grasping: High pressure, tight grip contact\n")
            f.write("   • Category 2 - Holding: Object manipulation, moderate grip\n")
            f.write("   • Category 3 - Other: All non-contact movements (gestures, pointing, etc.)\n\n")
            
            # Survey integration summary
            see_videos = [s for s in summaries if s.get('survey_data') and s['survey_data'].get('experiment_type') == 'SEE']
            blind_videos = [s for s in summaries if s.get('survey_data') and s['survey_data'].get('experiment_type') == 'BLIND']
            
            if see_videos or blind_videos:
                f.write("Survey Integration Summary:\n")
                f.write(f"   • SEE group videos (ID01-ID10): {len(see_videos)} analyzed\n")
                f.write(f"   • BLIND group videos (ID11-ID20): {len(blind_videos)} analyzed\n")
                if see_videos:
                    see_avg_freq = sum(s['grasping_frequency'] for s in see_videos) / len(see_videos)
                    f.write(f"   • SEE group average grasping frequency: {see_avg_freq:.2f} events/s\n")
                if blind_videos:
                    blind_avg_freq = sum(s['grasping_frequency'] for s in blind_videos) / len(blind_videos)
                    f.write(f"   • BLIND group average grasping frequency: {blind_avg_freq:.2f} events/s\n")
                f.write("\n")
            
            # Per-object results with survey data
            f.write("Per-Object Results with Survey Integration:\n\n")
            
            for i, summary in enumerate(summaries, 1):
                f.write(f"   {summary['video_label'].title()} Object:\n")
                
                # Video metrics
                hand_detection_time = summary.get('hand_detection_time', summary['duration'])
                f.write(f"      - Video duration: {summary['duration']:.2f}s\n")
                f.write(f"      - Hand detection time: {hand_detection_time:.2f}s\n")
                if 'orientation' in summary:
                    f.write(f"      - Orientation: {summary['orientation']}\n")
                f.write(f"      - Grasping events: {summary['grasping_events_count']}\n")
                f.write(f"      - Grasping frequency: {summary['grasping_frequency']:.3f}/s\n")
                f.write(f"      - Max velocity: {summary['max_velocity']:.2f} px/frame\n")
                f.write(f"      - Most common gesture: {summary['most_common_gesture'][0]}\n")
                
                # Survey data integration
                if summary.get('survey_data'):
                    survey = summary['survey_data']
                    f.write(f"      - Participant: {survey['participant_id']} ({survey['experiment_type']} group)\n")
                    f.write(f"      - Emotion felt: {survey['emotion_felt']} (Intensity: {survey['intensity']})\n")
                    f.write(f"      - Familiarity: {survey['familiarity']}\n")
                    f.write(f"      - Comfort during grasping: {survey['comfort_during_grasping']}\n")
                else:
                    f.write(f"      - Survey data: Not available\n")
                
                # Add grasping efficiency metric
                efficiency = summary['grasping_events_count'] / hand_detection_time if hand_detection_time > 0 else 0
                f.write(f"      - Grasping efficiency: {efficiency:.2f} events/second\n\n")
            
            # Analysis insights with object-specific findings
            f.write("Object-Specific Analysis Insights:\n")
            max_grasping_video = max(summaries, key=lambda x: x['grasping_frequency'])
            min_grasping_video = min(summaries, key=lambda x: x['grasping_frequency'])
            
            f.write(f"   • Most engaging object: {max_grasping_video['video_label'].title()} ")
            f.write(f"({max_grasping_video['grasping_frequency']:.2f} events/s)\n")
            f.write(f"   • Least engaging object: {min_grasping_video['video_label'].title()} ")
            f.write(f"({min_grasping_video['grasping_frequency']:.2f} events/s)\n")
            
            # Calculate activity ratio
            activity_ratio = max_grasping_video['grasping_frequency'] / min_grasping_video['grasping_frequency'] if min_grasping_video['grasping_frequency'] > 0 else 0
            f.write(f"   • Engagement ratio: {activity_ratio:.1f}x difference between most/least engaging objects\n\n")
            
            # Object-specific insights with survey correlation
            f.write("Object-Specific Behavior Patterns with Survey Correlation:\n")
            for summary in sorted(summaries, key=lambda x: x['grasping_frequency'], reverse=True):
                obj_name = summary['video_label'].title()
                freq = summary['grasping_frequency']
                if freq > avg_frequency * 1.2:
                    engagement_level = "High engagement"
                elif freq < avg_frequency * 0.8:
                    engagement_level = "Low engagement"
                else:
                    engagement_level = "Moderate engagement"
                
                # Add survey correlation if available
                survey_note = ""
                if summary.get('survey_data'):
                    survey = summary['survey_data']
                    survey_note = f" | Survey: {survey['emotion_felt']} emotion, {survey['comfort_during_grasping']} comfort"
                
                f.write(f"   • {obj_name}: {engagement_level} ({freq:.1f} events/s){survey_note}\n")
            
            # LLM optimization notes
            f.write("\nLLM Analysis Optimization Notes:\n")
            f.write("   • 3-category movement system: Grasping, Holding, Other\n")
            f.write("   • Hand detection time used instead of total video duration\n")
            f.write("   • Survey data integrated for participant analysis\n")
            f.write("   • SEE vs BLIND group comparison available\n")
            f.write("   • Object-specific emotional and comfort responses included\n")
            f.write("   • Engagement levels based on grasping frequency relative to average\n")
            f.write("   • Analysis optimized for LLM correlation and scientific research\n")
        
        print(f"Enhanced analysis report with survey integration saved: {report_path}")

    def save_enhanced_summary(self, summaries, output_dir):
        """Save enhanced JSON summary with 2-category grasp breakdown"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Clean summaries for JSON serialization
        clean_summaries = convert_numpy_types(summaries)
        
        # Calculate aggregate metrics
        total_duration = sum(s['duration'] for s in summaries)
        total_grasping_events = sum(s['grasping_events_count'] for s in summaries)
        avg_grasping_frequency = sum(s['grasping_frequency'] for s in summaries) / len(summaries)
        
        # Enhanced analysis with 2-category grasp type insights
        grasp_type_insights = {
            'methodology': {
                'grasping': 'High pressure contact with multiple fingers close to thumb, tight grip',
                'holding': 'Object manipulation with moderate grip strength, sustained contact',
                'detection_criteria': 'Multi-factor analysis: finger distance + curvature + hand openness',
                'categories': 'Simplified 2-category system for cleaner analysis'
            },
            'orientation_optimization': {
                'portrait_mode': 'Relaxed thresholds for closer camera perspective',
                'landscape_mode': 'Standard thresholds for distant camera perspective'
            }
        }
        
        enhanced_summary = {
            'analysis_info': {
                'total_videos_analyzed': len(summaries),
                'total_duration_seconds': float(total_duration),
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'algorithm_version': 'Enhanced 2-Category Grasping Detection v2.1'
            },
            'grasping_analysis': {
                'total_grasping_events': int(total_grasping_events),
                'average_grasping_frequency': float(avg_grasping_frequency),
                'grasping_events_per_video': [int(s['grasping_events_count']) for s in summaries],
                'grasp_categories': ['Grasping (Strong)', 'Holding (Moderate)'],
                'grasp_type_breakdown': 'Available in detailed CSV data'
            },
            'methodology': grasp_type_insights,
            'video_summaries': clean_summaries
        }
        
        json_path = os.path.join(output_dir, 'enhanced_summary.json')
        with open(json_path, 'w') as f:
            json.dump(enhanced_summary, f, indent=2)
        
        print(f"Enhanced summary with 2-category grasp analysis saved: {json_path}")

    def create_consolidated_dataset(self, summaries, output_dir):
        """Create master dataset combining all 20 videos for LLM analysis"""
        print("🔄 Creating consolidated master dataset...")
        
        # Combine all frame-level data
        all_frame_data = []
        video_summary_stats = []
        
        for summary in summaries:
            # Frame-level data
            frame_df = pd.DataFrame(summary['frame_data'])
            frame_df['video_id'] = summary['video_label']
            frame_df['video_duration'] = summary['duration']
            frame_df['video_orientation'] = summary.get('orientation', 'unknown')
            all_frame_data.append(frame_df)
            
            # Video-level summary stats
            video_stats = {
                'video_id': summary['video_label'],
                'duration_seconds': summary['duration'],
                'total_frames': summary['total_frames'],
                'fps': summary['fps'],
                'orientation': summary.get('orientation', 'unknown'),
                'avg_velocity': summary['avg_velocity'],
                'max_velocity': summary['max_velocity'],
                'path_length': summary['total_path_length'],
                'avg_hand_openness': summary['avg_hand_openness'],
                'grasping_events_count': summary['grasping_events_count'],
                'grasping_frequency': summary['grasping_frequency'],
                'most_common_gesture': summary['most_common_gesture'][0],
                'gesture_diversity': len(summary['gesture_counts']),
                'gesture_transitions': self.count_gesture_transitions(summary['frame_data']),
                'movement_efficiency': self.calculate_movement_efficiency(summary['frame_data']),
                'hand_stability_index': self.calculate_stability_index(summary['frame_data'])
            }
            video_summary_stats.append(video_stats)
        
        # Create consolidated DataFrames
        master_frame_data = pd.concat(all_frame_data, ignore_index=True)
        master_summary_data = pd.DataFrame(video_summary_stats)
        
        # Save consolidated datasets
        master_frame_path = os.path.join(output_dir, 'master_frame_dataset.csv')
        master_summary_path = os.path.join(output_dir, 'master_video_summary.csv')
        
        master_frame_data.to_csv(master_frame_path, index=False)
        master_summary_data.to_csv(master_summary_path, index=False)
        
        print(f"✅ Master frame dataset saved: {master_frame_path}")
        print(f"✅ Master summary dataset saved: {master_summary_path}")
        
        return master_frame_data, master_summary_data

    def count_gesture_transitions(self, frame_data):
        """Count gesture transitions in frame data"""
        if not frame_data or len(frame_data) < 2:
            return 0
        
        transitions = 0
        prev_gesture = frame_data[0].get('gesture', '')
        
        for frame in frame_data[1:]:
            current_gesture = frame.get('gesture', '')
            if current_gesture != prev_gesture:
                transitions += 1
            prev_gesture = current_gesture
            
        return transitions

    def calculate_movement_efficiency(self, frame_data):
        """Calculate movement efficiency (straight line distance / actual path)"""
        if not frame_data or len(frame_data) < 2:
            return 0
        
        # Get start and end positions
        start_x, start_y = frame_data[0]['center_x'], frame_data[0]['center_y']
        end_x, end_y = frame_data[-1]['center_x'], frame_data[-1]['center_y']
        
        # Calculate straight line distance
        straight_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Calculate actual path length
        actual_path = 0
        for i in range(1, len(frame_data)):
            dx = frame_data[i]['center_x'] - frame_data[i-1]['center_x']
            dy = frame_data[i]['center_y'] - frame_data[i-1]['center_y']
            actual_path += np.sqrt(dx**2 + dy**2)
        
        if actual_path == 0:
            return 0
        
        return straight_distance / actual_path

    def calculate_stability_index(self, frame_data):
        """Calculate hand stability based on velocity variation"""
        if not frame_data or len(frame_data) < 3:
            return 0
        
        velocities = [frame.get('velocity', 0) for frame in frame_data]
        
        if len(velocities) < 2:
            return 0
        
        velocity_std = np.std(velocities)
        velocity_mean = np.mean(velocities)
        
        if velocity_mean == 0:
            return 1.0
        
        # Higher stability = lower coefficient of variation
        stability = 1.0 / (1.0 + velocity_std / velocity_mean)
        return min(1.0, stability)

    def create_statistical_comparison_analysis(self, summaries, output_dir):
        """Create statistical comparison analysis between videos"""
        print("📊 Creating statistical comparison analysis...")
        
        # Extract metrics for statistical analysis
        metrics = {
            'video_ids': [s['video_label'] for s in summaries],
            'grasping_frequencies': [s['grasping_frequency'] for s in summaries],
            'avg_velocities': [s['avg_velocity'] for s in summaries],
            'max_velocities': [s['max_velocity'] for s in summaries],
            'hand_openness': [s['avg_hand_openness'] for s in summaries],
            'durations': [s['duration'] for s in summaries],
            'grasping_counts': [s['grasping_events_count'] for s in summaries]
        }
        
        # Statistical tests and comparisons
        statistical_results = {
            'descriptive_stats': {},
            'correlations': {},
            'object_rankings': {},
            'statistical_tests': {},
            'effect_sizes': {}
        }
        
        # Descriptive statistics
        for metric, values in metrics.items():
            if metric != 'video_ids':
                statistical_results['descriptive_stats'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                }
        
        # Correlations between metrics
        numeric_metrics = ['grasping_frequencies', 'avg_velocities', 'max_velocities', 'hand_openness', 'durations']
        for i, metric1 in enumerate(numeric_metrics):
            for metric2 in numeric_metrics[i+1:]:
                correlation, p_value = stats.pearsonr(metrics[metric1], metrics[metric2])
                statistical_results['correlations'][f'{metric1}_vs_{metric2}'] = {
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'significance': 'significant' if p_value < 0.05 else 'not_significant'
                }
        
        # Object difficulty ranking based on multiple metrics
        object_scores = []
        for i, video_id in enumerate(metrics['video_ids']):
            # Normalized composite score (higher = more engaging/easier)
            score = (
                (metrics['grasping_frequencies'][i] / max(metrics['grasping_frequencies'])) * 0.4 +
                (metrics['avg_velocities'][i] / max(metrics['avg_velocities'])) * 0.3 +
                (metrics['hand_openness'][i] / max(metrics['hand_openness'])) * 0.3
            )
            object_scores.append((video_id, score))
        
        # Sort by engagement score
        object_scores.sort(key=lambda x: x[1], reverse=True)
        statistical_results['object_rankings'] = {
            'engagement_ranking': [{'object': obj, 'score': float(score), 'rank': rank+1} 
                                 for rank, (obj, score) in enumerate(object_scores)]
        }
        
        # Statistical tests (if we have enough data)
        if len(summaries) >= 3:
            # Test for differences in grasping frequency between objects
            grasping_freq_groups = [metrics['grasping_frequencies']]
            statistical_results['statistical_tests']['grasping_frequency_normality'] = {
                'shapiro_wilk_p': float(stats.shapiro(metrics['grasping_frequencies'])[1])
            }
        
        # Save statistical analysis
        stats_path = os.path.join(output_dir, 'statistical_analysis.json')
        with open(stats_path, 'w') as f:
            json.dump(statistical_results, f, indent=2)
        
        print(f"✅ Statistical analysis saved: {stats_path}")
        return statistical_results

    def create_llm_analysis_report(self, summaries, statistical_results, output_dir):
        """Create comprehensive markdown report optimized for LLM analysis"""
        print("📝 Creating LLM-optimized analysis report...")
        
        report_path = os.path.join(output_dir, 'llm_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Hand Movement Analysis Report\n")
            f.write("## Dataset Overview: 20-Video Scientific Analysis\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            total_duration = sum(s['duration'] for s in summaries)
            total_grasping = sum(s['grasping_events_count'] for s in summaries)
            avg_frequency = sum(s['grasping_frequency'] for s in summaries) / len(summaries)
            
            f.write(f"- **Total Videos Analyzed**: {len(summaries)}\n")
            f.write(f"- **Total Analysis Duration**: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)\n")
            f.write(f"- **Total Grasping Events Detected**: {total_grasping:,}\n")
            f.write(f"- **Average Grasping Frequency**: {avg_frequency:.2f} events/second\n")
            f.write(f"- **Analysis Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Key Findings
            f.write("## Key Scientific Findings\n\n")
            
            # Object rankings
            rankings = statistical_results['object_rankings']['engagement_ranking']
            f.write("### Object Engagement Rankings\n\n")
            f.write("| Rank | Object | Engagement Score | Classification |\n")
            f.write("|------|--------|------------------|----------------|\n")
            
            for ranking in rankings:
                classification = "High" if ranking['score'] > 0.7 else "Medium" if ranking['score'] > 0.4 else "Low"
                f.write(f"| {ranking['rank']} | {ranking['object'].title()} | {ranking['score']:.3f} | {classification} |\n")
            
            f.write("\n")
            
            # Statistical insights
            f.write("### Statistical Insights\n\n")
            desc_stats = statistical_results['descriptive_stats']
            
            f.write(f"- **Grasping Frequency**: {desc_stats['grasping_frequencies']['mean']:.2f} ± {desc_stats['grasping_frequencies']['std']:.2f} events/s\n")
            f.write(f"- **Hand Movement Velocity**: {desc_stats['avg_velocities']['mean']:.2f} ± {desc_stats['avg_velocities']['std']:.2f} px/s\n")
            f.write(f"- **Hand Openness**: {desc_stats['hand_openness']['mean']:.3f} ± {desc_stats['hand_openness']['std']:.3f}\n")
            f.write(f"- **Coefficient of Variation (Grasping)**: {desc_stats['grasping_frequencies']['cv']:.3f}\n\n")
            
            # Correlations
            f.write("### Significant Correlations\n\n")
            for corr_name, corr_data in statistical_results['correlations'].items():
                if corr_data['significance'] == 'significant':
                    metric1, metric2 = corr_name.replace('_vs_', ' vs ').replace('_', ' ').title().split(' Vs ')
                    f.write(f"- **{metric1} vs {metric2}**: r = {corr_data['correlation']:.3f}, p = {corr_data['p_value']:.4f}\n")
            f.write("\n")
            
            # Individual video performance
            f.write("## Individual Video Analysis\n\n")
            f.write("| Video | Duration(s) | Grasping Events | Frequency(events/s) | Avg Velocity | Most Common Gesture |\n")
            f.write("|-------|-------------|-----------------|---------------------|--------------|---------------------|\n")
            
            for summary in sorted(summaries, key=lambda x: x['grasping_frequency'], reverse=True):
                f.write(f"| {summary['video_label'].title()} | {summary['duration']:.1f} | {summary['grasping_events_count']} | {summary['grasping_frequency']:.2f} | {summary['avg_velocity']:.2f} | {summary['most_common_gesture'][0]} |\n")
            
            f.write("\n")
            
            # Methodology
            f.write("## Methodology\n\n")
            f.write("### Data Collection\n")
            f.write("- **Hand Tracking**: MediaPipe Hands solution\n")
            f.write("- **Gesture Classification**: 3-category system (Grasping, Holding, Other)\n")
            f.write("- **Movement Analysis**: Frame-by-frame velocity and trajectory tracking\n")
            f.write("- **Statistical Analysis**: Pearson correlations, descriptive statistics\n\n")
            
            f.write("### Metrics Defined\n")
            f.write("- **Grasping Events**: Tight grip contact with high pressure\n")
            f.write("- **Holding Events**: Object manipulation with moderate grip\n")
            f.write("- **Movement Efficiency**: Straight-line distance / actual path ratio\n")
            f.write("- **Hand Stability**: Inverse of velocity coefficient of variation\n")
            f.write("- **Engagement Score**: Composite metric (40% grasping freq + 30% velocity + 30% openness)\n\n")
            
            # Research implications
            f.write("## Research Implications\n\n")
            
            highest_engagement = max(summaries, key=lambda x: x['grasping_frequency'])
            lowest_engagement = min(summaries, key=lambda x: x['grasping_frequency'])
            engagement_ratio = highest_engagement['grasping_frequency'] / lowest_engagement['grasping_frequency']
            
            f.write(f"### Object-Specific Findings\n")
            f.write(f"- **Most Engaging Object**: {highest_engagement['video_label'].title()} ({highest_engagement['grasping_frequency']:.1f} events/s)\n")
            f.write(f"- **Least Engaging Object**: {lowest_engagement['video_label'].title()} ({lowest_engagement['grasping_frequency']:.1f} events/s)\n")
            f.write(f"- **Engagement Ratio**: {engagement_ratio:.1f}x difference between most and least engaging objects\n\n")
            
            f.write("### Clinical/Research Applications\n")
            f.write("- Object selection for motor assessment protocols\n")
            f.write("- Standardized metrics for hand function evaluation\n")
            f.write("- Baseline data for comparative studies\n")
            f.write("- Technology-assisted rehabilitation planning\n\n")
            
            # Data files reference
            f.write("## Data Files\n\n")
            f.write("- `master_frame_dataset.csv`: Frame-by-frame data for all videos\n")
            f.write("- `master_video_summary.csv`: Summary statistics per video\n")
            f.write("- `statistical_analysis.json`: Detailed statistical results\n")
            f.write("- `enhanced_summary.json`: Complete analysis metadata\n")
            f.write("- Visualization files: `1_main_analysis.png`, `2_grasping_analysis.png`, etc.\n")
        
        print(f"✅ LLM analysis report saved: {report_path}")
        return report_path

    def create_plot_data_for_llm(self, summaries, output_dir):
        """Convert plot visualizations to text-based data for LLM analysis"""
        print("📊 Creating text-based plot data for LLM...")
        
        plot_data = {
            'trajectory_analysis': {},
            'grasping_frequency_comparison': {},
            'temporal_patterns': {},
            'statistical_distributions': {}
        }
        
        # Extract trajectory data
        for summary in summaries:
            video_label = summary['video_label']
            frame_data = summary['frame_data']
            
            if frame_data:
                # Trajectory summary
                x_positions = [frame['center_x'] for frame in frame_data]
                y_positions = [frame['center_y'] for frame in frame_data]
                
                plot_data['trajectory_analysis'][video_label] = {
                    'start_position': [x_positions[0], y_positions[0]],
                    'end_position': [x_positions[-1], y_positions[-1]],
                    'position_range_x': [min(x_positions), max(x_positions)],
                    'position_range_y': [min(y_positions), max(y_positions)],
                    'trajectory_length': len(frame_data),
                    'movement_area': (max(x_positions) - min(x_positions)) * (max(y_positions) - min(y_positions))
                }
        
        # Grasping frequency data
        grasping_freq_data = [(s['video_label'], s['grasping_frequency']) for s in summaries]
        grasping_freq_data.sort(key=lambda x: x[1], reverse=True)
        
        plot_data['grasping_frequency_comparison'] = {
            'ranked_frequencies': grasping_freq_data,
            'frequency_statistics': {
                'mean': np.mean([freq for _, freq in grasping_freq_data]),
                'std': np.std([freq for _, freq in grasping_freq_data]),
                'range': [min(freq for _, freq in grasping_freq_data), max(freq for _, freq in grasping_freq_data)]
            }
        }
        
        # Temporal patterns (simplified time-binned analysis)
        for summary in summaries:
            video_label = summary['video_label']
            frame_data = summary['frame_data']
            
            if frame_data and len(frame_data) > 10:
                # Create 5 time bins
                total_frames = len(frame_data)
                bin_size = total_frames // 5
                
                time_bins = []
                for i in range(5):
                    start_idx = i * bin_size
                    end_idx = min((i + 1) * bin_size, total_frames)
                    bin_frames = frame_data[start_idx:end_idx]
                    
                    grasping_rate = sum(1 for frame in bin_frames if frame.get('is_grasping', False)) / len(bin_frames)
                    time_bins.append(grasping_rate)
                
                plot_data['temporal_patterns'][video_label] = {
                    'time_bins': time_bins,
                    'peak_activity_bin': time_bins.index(max(time_bins)) + 1,
                    'activity_trend': 'increasing' if time_bins[-1] > time_bins[0] else 'decreasing'
                }
        
        # Save plot data
        plot_data_path = os.path.join(output_dir, 'plot_data_for_llm.json')
        with open(plot_data_path, 'w') as f:
            json.dump(plot_data, f, indent=2)
        
        print(f"✅ Plot data for LLM saved: {plot_data_path}")
        return plot_data

    def create_llm_optimized_analysis(self, summaries, output_dir):
        """Master function to create all LLM-optimized analysis files"""
        print("\n🤖 Creating LLM-Optimized Analysis Package...")
        print("="*60)
        
        # Create master analysis directory
        master_dir = os.path.join(output_dir, 'llm_analysis')
        os.makedirs(master_dir, exist_ok=True)
        
        # 1. Consolidated dataset
        master_frame_data, master_summary_data = self.create_consolidated_dataset(summaries, master_dir)
        
        # 2. Statistical comparison analysis
        statistical_results = self.create_statistical_comparison_analysis(summaries, master_dir)
        
        # 3. LLM-optimized markdown report
        report_path = self.create_llm_analysis_report(summaries, statistical_results, master_dir)
        
        # 4. Plot data in text format
        plot_data = self.create_plot_data_for_llm(summaries, master_dir)
        
        # 5. Create key findings summary for quick LLM reference
        key_findings = self.extract_key_findings_for_llm(summaries, statistical_results, master_dir)
        
        print(f"\n✅ LLM-Optimized Analysis Package Complete!")
        print(f"📁 Location: {master_dir}")
        print("📋 Files created:")
        print("   • master_frame_dataset.csv - All frame data")
        print("   • master_video_summary.csv - Video summaries") 
        print("   • statistical_analysis.json - Statistical results")
        print("   • llm_analysis_report.md - Comprehensive markdown report")
        print("   • plot_data_for_llm.json - Text-based plot data")
        print("   • key_findings_summary.json - Quick reference findings")
        
        return master_dir

    def extract_key_findings_for_llm(self, summaries, statistical_results, output_dir):
        """Extract key findings in structured format for LLM quick reference"""
        
        # Calculate key metrics
        total_videos = len(summaries)
        total_duration = sum(s['duration'] for s in summaries)
        total_grasping = sum(s['grasping_events_count'] for s in summaries)
        
        # Object performance analysis
        object_performance = []
        for summary in summaries:
            performance = {
                'object': summary['video_label'],
                'grasping_frequency': summary['grasping_frequency'],
                'engagement_level': 'high' if summary['grasping_frequency'] > 15 else 'medium' if summary['grasping_frequency'] > 8 else 'low',
                'dominant_gesture': summary['most_common_gesture'][0],
                'movement_characteristics': {
                    'velocity': 'high' if summary['avg_velocity'] > 3 else 'medium' if summary['avg_velocity'] > 1.5 else 'low',
                    'hand_openness': 'open' if summary['avg_hand_openness'] > 0.7 else 'moderate' if summary['avg_hand_openness'] > 0.4 else 'closed'
                }
            }
            object_performance.append(performance)
        
        # Sort by engagement
        object_performance.sort(key=lambda x: x['grasping_frequency'], reverse=True)
        
        key_findings = {
            'study_overview': {
                'total_videos': total_videos,
                'total_duration_minutes': round(total_duration / 60, 1),
                'total_grasping_events': total_grasping,
                'analysis_scope': '20-video comprehensive hand movement study'
            },
            'performance_rankings': {
                'most_engaging': object_performance[0]['object'],
                'least_engaging': object_performance[-1]['object'],
                'engagement_ratio': round(object_performance[0]['grasping_frequency'] / object_performance[-1]['grasping_frequency'], 1)
            },
            'object_characteristics': object_performance,
            'statistical_significance': {
                'significant_correlations': [
                    corr_name for corr_name, corr_data in statistical_results['correlations'].items()
                    if corr_data['significance'] == 'significant'
                ],
                'grasping_frequency_cv': statistical_results['descriptive_stats']['grasping_frequencies']['cv']
            },
            'research_implications': [
                f"Object difficulty varies by {object_performance[0]['grasping_frequency'] / object_performance[-1]['grasping_frequency']:.1f}x across objects",
                f"Most objects show {object_performance[0]['dominant_gesture'].lower()} as dominant interaction pattern",
                f"Hand movement patterns demonstrate clear object-specific preferences",
                f"Statistical analysis based on {total_grasping:,} individual grasping events"
            ]
        }
        
        # Save key findings
        findings_path = os.path.join(output_dir, 'key_findings_summary.json')
        with open(findings_path, 'w') as f:
            json.dump(key_findings, f, indent=2)
        
        print(f"✅ Key findings summary saved: {findings_path}")
        return key_findings

def main():
    parser = argparse.ArgumentParser(description='Enhanced Hand Movement Analysis with Survey Integration')
    parser.add_argument('video_path', nargs='?', help='Path to single video file or folder')
    parser.add_argument('--path', help='Path to folder containing multiple videos')
    parser.add_argument('--batch', help='Path to folder for batch processing')
    parser.add_argument('--show', action='store_true', help='Show real-time analysis')
    
    args = parser.parse_args()
    
    # Determine processing mode
    target_path = args.video_path or args.path or args.batch
    
    if not target_path:
        print("❌ Please provide a video file or folder path!")
        print("\nUsage examples:")
        print("  Single video: python3 advanced_hand_analyzer.py ID1/toy.mov")
        print("  Folder batch: python3 advanced_hand_analyzer.py --batch ID1/")
        print("  Folder batch: python3 advanced_hand_analyzer.py ID1/")
        print("  Current dir:  python3 advanced_hand_analyzer.py --batch ./")
        parser.print_help()
        return
    
    analyzer = HandMovementAnalyzer()
    
    try:
        # Check if target is a folder or file
        if os.path.isdir(target_path):
            # Process folder with object-specific labels
            print(f"Processing folder: {target_path}")
            analyzer.process_folder(target_path, show_video=args.show)
        elif os.path.isfile(target_path):
            # Process single video file
            print(f"Processing single video: {target_path}")
            
            # Extract video label from filename (for consistency)
            video_filename = os.path.basename(target_path)
            video_label = os.path.splitext(video_filename)[0]
            
            summary = analyzer.process_video(target_path, show_video=args.show, video_label=video_label)
            
            if summary:
                # Create single-video output directory
                output_dir = os.path.join('analysis_results', video_label)
                os.makedirs(output_dir, exist_ok=True)
                print(f"Output directory: {output_dir}")
                
                # Create DataFrame from summary data
                df = pd.DataFrame(summary['frame_data'])
                df['video_label'] = summary['video_label']
                
                # Save comprehensive data
                csv_path = os.path.join(output_dir, 'detailed_analysis.csv')
                df.to_csv(csv_path, index=False)
                print(f"Detailed data saved: {csv_path}")
                
                # Create plots for single video
                analyzer.create_main_analysis_plots(df, [summary], output_dir)
                analyzer.create_grasping_analysis_plots(df, [summary], output_dir)
                analyzer.create_temporal_analysis_plots(df, [summary], output_dir)
                analyzer.create_summary_table_plot([summary], output_dir)
                print(f"Enhanced analysis plots created in 4 separate figures!")
                
                # Save enhanced summary
                analyzer.save_enhanced_summary([summary], output_dir)
                
                # Create analysis report
                analyzer.create_analysis_report([summary], output_dir)
                
                print(f"\nSingle video analysis complete! Results saved to: {output_dir}")
            else:
                print("Failed to process the video!")
        else:
            # Path doesn't exist or is neither file nor directory
            print(f"❌ Error: Path '{target_path}' does not exist or is not accessible!")
            print("\nPlease check:")
            print("  • File/folder path is correct")
            print("  • File has video extension (.mov, .mp4, .avi, etc.)")
            print("  • You have read permissions")
                
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 