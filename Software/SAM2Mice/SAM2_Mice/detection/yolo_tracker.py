import json
import os
from typing import List
import cv2
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from collections import deque



class MouseTracker:
    def __init__(self, model_path, num_mice=5, history_size=30, max_missing_frames=50):
        # Tracking settings
        self.num_mice = num_mice  # Fixed number of mice (5)
        self.history_size = history_size  # Frames to keep in history
        self.max_missing_frames = max_missing_frames  # Max consecutive frames before considering a track lost
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Tracking state variables
        self.active_tracks = {}  # Current active tracks {id: last_position}
        self.track_history = {}  # History of positions {id: deque of positions}
        self.box_history = {}  # History of bounding boxes {id: deque of boxes}
        self.consecutive_missing_frames = {}  # Count of consecutive frames a mouse ID has been missing
        self.initialized = False  # Whether we've assigned initial IDs
        
        # Video writer
        self.video_writer = None
        
        # Generate a distinct color for each mouse ID
        self.id_colors = self.generate_distinct_colors(num_mice)
    
    def generate_distinct_colors(self, n):
        """Generate n distinct colors for visualization"""
        colors = []
        for i in range(n):
            # Use HSV color space to generate evenly distributed colors
            hue = i * 255 / n
            # Convert to BGR for OpenCV
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            # Convert to integer tuple
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU (Intersection over Union) between two boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate both boxes' areas
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def calculate_overlap_percentage(self, box1, box2):
        """Calculate what percentage of box1 is contained within box2"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate box1 area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        # Calculate percentage of box1 contained in box2
        if box1_area == 0:
            return 0.0
        
        return intersection_area / float(box1_area)
    
    def filter_overlapping_boxes(self, boxes, overlap_threshold=0.9):
        """Filter out boxes that are mostly contained within other boxes"""
        if len(boxes) <= 1:
            return boxes
        
        # Sort boxes by area (largest first) to prioritize larger detections
        box_areas = [(i, (b[2]-b[0])*(b[3]-b[1])) for i, b in enumerate(boxes)]
        box_areas.sort(key=lambda x: x[1], reverse=True)
        
        keep_indices = []
        for i, (idx1, _) in enumerate(box_areas):
            keep = True
            for j, (idx2, _) in enumerate(box_areas):
                # Skip self-comparison
                if i == j:
                    continue
                
                # Calculate what percentage of the smaller box is inside the larger box
                overlap = self.calculate_overlap_percentage(boxes[idx1], boxes[idx2])
                
                # If this box is mostly contained within another box, don't keep it
                if overlap > overlap_threshold:
                    keep = False
                    break
                    
            if keep:
                keep_indices.append(idx1)
                
        return [boxes[i] for i in keep_indices]
    
    def calculate_similarity_score(self, track_id, box_idx, boxes, centers):
        """Calculate similarity score based on both distance and IoU"""
        # Get current track position
        track_pos = self.active_tracks[track_id]
        center = centers[box_idx]
        
        # Calculate Euclidean distance
        dist = np.sqrt((track_pos[0] - center[0])**2 + (track_pos[1] - center[1])**2)
        # Normalize distance (lower is better)
        max_dist = 1000  # Set a reasonable maximum distance
        norm_dist = max(0, 1 - (dist / max_dist))
        
        # Calculate IoU if there's history
        iou_score = 0
        if track_id in self.box_history and self.box_history[track_id]:
            last_box = self.box_history[track_id][-1]
            current_box = boxes[box_idx]
            iou_score = self.calculate_iou(last_box, current_box)
        
        # Weight factors for distance and IoU
        dist_weight = 0.8
        iou_weight = 0.2
        
        # Combined score (higher is better)
        score = (dist_weight * norm_dist) + (iou_weight * iou_score)
        
        return score
    
    def initialize_tracks(self, boxes):
        """Initialize tracks with IDs from 0 to num_mice-1"""
        if len(boxes) > self.num_mice:
            # Keep only the first num_mice boxes if we have more detections
            boxes = boxes[:self.num_mice]
        
        for i, box in enumerate(boxes):
            track_id = i  # Use indices 0 to num_mice-1 as track IDs
            
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            self.active_tracks[track_id] = (center_x, center_y)
            self.track_history[track_id] = deque(maxlen=self.history_size)
            self.track_history[track_id].append((center_x, center_y))
            self.consecutive_missing_frames[track_id] = 0
            
            # Initialize box history
            self.box_history[track_id] = deque(maxlen=self.history_size)
            self.box_history[track_id].append(box)
        
        # Create empty tracks for any missing mice
        for i in range(len(boxes), self.num_mice):
            track_id = i
            self.active_tracks[track_id] = None  # Mark as missing initially
            self.track_history[track_id] = deque(maxlen=self.history_size)
            self.box_history[track_id] = deque(maxlen=self.history_size)
            self.consecutive_missing_frames[track_id] = self.max_missing_frames  # Mark as fully missing
        
        self.initialized = True
    
    def assign_ids(self, boxes, frame_shape):
        """Assign IDs to detected boxes based on position, history, and IoU"""
        # Initialize tracks if this is the first frame
        if not self.initialized:
            self.initialize_tracks(boxes)
            return [(boxes[i], i) for i in range(min(len(boxes), self.num_mice))]
        
        if not boxes:
            # Increment consecutive missing frames counter for all active tracks
            for track_id in self.active_tracks.keys():
                if self.active_tracks[track_id] is not None:  # Only for tracks that were active
                    self.consecutive_missing_frames[track_id] += 1
            return []
        
        # Calculate centers of current boxes
        centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]
        
        # Calculate similarity scores between current detections and active tracks
        similarity_scores = {}
        for track_id in self.active_tracks.keys():
            # Skip tracks that don't have a position yet
            if self.active_tracks[track_id] is None:
                continue
                
            for box_idx in range(len(boxes)):
                score = self.calculate_similarity_score(track_id, box_idx, boxes, centers)
                similarity_scores[(track_id, box_idx)] = score
        
        # Sort similarity scores (higher is better)
        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Assign boxes to tracks greedily based on scores
        assigned_track_ids = set()
        assigned_box_indices = set()
        assigned_boxes = []
        
        for (track_id, box_idx), score in sorted_scores:
            # If either this track or this box is already assigned, skip
            if track_id in assigned_track_ids or box_idx in assigned_box_indices:
                continue
                
            assigned_track_ids.add(track_id)
            assigned_box_indices.add(box_idx)
            
            # Update track position
            box = boxes[box_idx]
            center_x, center_y = centers[box_idx]
            self.active_tracks[track_id] = (center_x, center_y)
            
            # Update history
            self.track_history[track_id].append((center_x, center_y))
            self.box_history[track_id].append(box)
            
            # Reset consecutive missing frames counter
            self.consecutive_missing_frames[track_id] = 0
            
            assigned_boxes.append((box, track_id))
        
        # For any unassigned boxes, assign them to the tracks that have been missing the longest
        unassigned_boxes = [boxes[i] for i in range(len(boxes)) if i not in assigned_box_indices]
        unassigned_tracks = [i for i in range(self.num_mice) if i not in assigned_track_ids]
        
        # Sort unassigned tracks by consecutive missing frames (highest first)
        unassigned_tracks.sort(key=lambda track_id: self.consecutive_missing_frames.get(track_id, 0), reverse=True)
        
        for i, track_id in enumerate(unassigned_tracks):
            if i >= len(unassigned_boxes):
                break
                
            box = unassigned_boxes[i]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            self.active_tracks[track_id] = (center_x, center_y)
            self.track_history[track_id].append((center_x, center_y))
            self.box_history[track_id].append(box)
            self.consecutive_missing_frames[track_id] = 0
            
            assigned_boxes.append((box, track_id))
        
        # Update consecutive missing frames counter for tracks that weren't assigned
        for track_id in range(self.num_mice):
            if track_id not in assigned_track_ids:
                self.consecutive_missing_frames[track_id] += 1
        
        return assigned_boxes
    
    def predict(self, img, classes=[], conf=0.5):
        """Run YOLO prediction on the image"""
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)
        
        return results
    
    def extract_boxes_from_results(self, results):
        """Extract bounding boxes from YOLO results"""
        all_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                all_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        return all_boxes
    
    def process_frame(self, frame, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        """Process a single frame to detect and track mice"""
        # Get original detection results
        results = self.predict(frame, classes, conf=conf)
        
        # Extract boxes from results
        all_boxes = self.extract_boxes_from_results(results)
        
        # Filter overlapping boxes
        filtered_boxes = self.filter_overlapping_boxes(all_boxes)
        
        # Assign IDs to boxes
        img_height, img_width = frame.shape[:2]
        assigned_boxes = self.assign_ids(filtered_boxes, (img_width, img_height))
        
        # Create output image with annotations
        output_img = self.draw_annotations(frame, assigned_boxes, rectangle_thickness, text_thickness)
        
        return output_img, results, assigned_boxes
    
    def draw_annotations(self, img, assigned_boxes, rectangle_thickness=2, text_thickness=1):
        """Draw boxes, IDs, and trajectories on the image"""
        output_img = img.copy()
        for box, track_id in assigned_boxes:
            x1, y1, x2, y2 = box
            
            # Get color for this ID
            color = self.id_colors[track_id]
            
            # Draw bounding box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, rectangle_thickness)
            
            # Draw ID
            cv2.putText(output_img, f"Mouse #{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, text_thickness)
            
            # Draw trajectory (optional)
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = list(self.track_history[track_id])
                for i in range(1, len(points)):
                    if points[i-1] is not None and points[i] is not None:
                        cv2.line(output_img, 
                                (int(points[i-1][0]), int(points[i-1][1])),
                                (int(points[i][0]), int(points[i][1])), 
                                color, 1)
        
        # Display the count of active tracks
        active_count = sum(1 for id in range(self.num_mice) 
                        if id in self.active_tracks and self.active_tracks[id] is not None)
        cv2.putText(output_img, f"Detected mice: {active_count}/{self.num_mice}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output_img
    
    def init_video_writer(self, output_path, frame_width, frame_height, fps=30):
        """Initialize video writer for saving output"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    def process_video(self, input_video_path, output_video_path, classes=[], conf=0.5, display=False):
        """Process entire video, track mice, and save output"""
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {input_video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        self.init_video_writer(output_video_path, width, height, fps)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame, _, _ = self.process_frame(frame, classes, conf)
            
            # Write to output video
            self.video_writer.write(processed_frame)
            
            # Display progress
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # Display frame if requested
            if display:
                cv2.imshow('Mouse Tracking', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Clean up
        cap.release()
        self.video_writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"Video processing complete. Output saved to {output_video_path}")
        return True
    
    def close(self):
        """Clean up resources"""
        if self.video_writer is not None:
            self.video_writer.release()
