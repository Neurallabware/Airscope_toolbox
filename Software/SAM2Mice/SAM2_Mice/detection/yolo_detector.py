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


def sample_points_from_masks(masks: np.ndarray, num_points: int = 10) -> List[np.ndarray]:
    """
    Sample positive points from masks for prompting SAM2, ensuring points are distributed
    to cover the entire mask area.
    
    Args:
        masks (np.ndarray): Binary masks with shape (n, h, w)
        num_points (int, optional): Number of points to sample from each mask. Defaults to 10.
    
    Returns:
        List[np.ndarray]: List of sampled points from each mask with shape (num_points, 2)
    """
    sampled_points_list = []
    
    for mask in masks:
        # Find coordinates of all positive pixels in the mask
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) > 0:
            points = np.stack([x_indices, y_indices], axis=1)
            
            # If there are fewer positive pixels than requested points, use all of them
            if len(points) <= num_points:
                sampled_points_list.append(points)
                continue
                
            # Use K-means clustering to identify representative points across the mask
            from sklearn.cluster import KMeans
            
            # Apply K-means to find cluster centers distributed across the mask
            kmeans = KMeans(n_clusters=num_points, n_init=1, random_state=0)
            kmeans.fit(points)
            
            # For each cluster, find the point closest to its center
            sampled_points = []
            for center in kmeans.cluster_centers_:
                # Find the point in the original data closest to this center
                distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
                closest_point_idx = np.argmin(distances)
                sampled_points.append(points[closest_point_idx])
            
            sampled_points = np.array(sampled_points)
            sampled_points_list.append(sampled_points)
        else:
            # If mask is empty, return an empty array with correct shape
            sampled_points_list.append(np.zeros((0, 2), dtype=np.float32))
    
    return sampled_points_list


class YOLODetector:
    def __init__(self, model_path, conf=0.5, classes=None, device=None):
        """
        Initialize the YOLO detector class.

        Args:
            model_path: Path to the YOLO model weights
            conf: Confidence threshold for detections
            classes: List of class indices to filter for
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.id_to_name = {0: "mouse"}

    def predict(self, img, conf=None, classes=None):
        """
        Run YOLO model prediction on an image.

        Args:
            img: Image to run prediction on (file path or numpy array)
            conf: Optional confidence threshold (overrides default)
            classes: Optional list of class indices (overrides default)

        Returns:
            The results from the YOLO model
        """
        conf_threshold = conf if conf is not None else self.conf
        class_filter = classes if classes is not None else self.classes

        if class_filter is not None:
            results = self.model.predict(img, classes=class_filter, conf=conf_threshold, device=self.device)
        else:
            results = self.model.predict(img, conf=conf_threshold, device=self.device)

        return results

    def get_boxes(self, results):
        """
        Extract bounding boxes from YOLO results, xyxy.
        """
        return results[0].boxes.xyxy.cpu().numpy()

    def get_classes(self, results):
        """
        Extract class IDs from YOLO results.
        """
        return results[0].boxes.cls.cpu().numpy()

    def get_confidence(self, results):
        """
        Extract confidence scores from YOLO results.
        """
        return results[0].boxes.conf.cpu().numpy()
    
    def infernence_image(self, img, results, save_path=None):
        """
        Visualize detection results on the image.

        Args:
            img: Original image (numpy array)
            results: Results from YOLO prediction
            save_path: Path to save the annotated image

        Returns:
            list os boxes
        """
        # Get boxes, convert to supervision format
        boxes = self.get_boxes(results)
        confidences = self.get_confidence(results)
        object_ids = np.arange(len(boxes), dtype=np.int32) + 1

        # Create detections object
        detections = sv.Detections(
            xyxy=boxes,
            class_id=object_ids,
            confidence=confidences
        )
        # Annotate
        box_annotator = sv.BoxAnnotator()
        annotated_img = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_img = label_annotator.annotate(annotated_img, detections=detections,
                                                labels=[f"mouse{i}" for i in object_ids])
        annotated_img = np.array(annotated_img)

        if save_path:
            cv2.imwrite(save_path, annotated_img)

        return boxes

    # def infernence_image(self, img, results, save_path=None):
    #     """
    #     Visualize detection results on the image.

    #     Args:
    #         img: Original image (numpy array)
    #         results: Results from YOLO prediction
    #         save_path: Path to save the annotated image

    #     Returns:
    #         reordered list of boxes
    #     """
    #     # 1) Get boxes & confidences
    #     boxes = self.get_boxes(results)           # shape: (N, 4)
    #     confidences = self.get_confidence(results) # shape: (N,)
        
    #     # 2) define your desired permutation:
    #     #    here: [1,2,0,3,4] maps old indices 0→1, 1→2, 2→0, 3→3, 4→4
    #     perm = [1, 2, 0] + list(range(3, len(boxes)))
        
    #     # 3) apply it
    #     boxes       = boxes[perm]
    #     confidences = confidences[perm]
        
    #     # 4) regenerate object IDs and labels in the new order
    #     object_ids = np.arange(len(boxes), dtype=np.int32) + 1
    #     labels     = [f"mouse{oid}" for oid in object_ids]

    #     # 5) build your supervision detections
    #     detections = sv.Detections(
    #         xyxy=boxes,
    #         class_id=object_ids,
    #         confidence=confidences
    #     )

    #     # 6) draw boxes + labels
    #     box_annotator   = sv.BoxAnnotator()
    #     annotated_img   = box_annotator.annotate(scene=img.copy(), detections=detections)

    #     label_annotator = sv.LabelAnnotator()
    #     annotated_img   = label_annotator.annotate(annotated_img, detections=detections, labels=labels)
    #     annotated_img   = np.array(annotated_img)

    #     # 7) optionally save
    #     if save_path:
    #         cv2.imwrite(save_path, annotated_img)

    #     return boxes
    

    def infernence_image_with_SAM2(self, img, results, sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt",
                                   sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml", save_path=None):
        """
        Visualize detection results on the image.

        Args:
            img: Original image (numpy array)
            results: Results from YOLO prediction
            save_path: Path to save the annotated image

        Returns:
            box: n*4 xyxy, masks: n*h*w
        """
        # Enable optimizations if CUDA is available
        if torch.cuda.is_available():
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        boxes = self.get_boxes(results)
        object_ids = np.arange(len(boxes), dtype=np.int32) + 1

        sam2_image_model = build_sam2(sam2_model_cfg, sam2_checkpoint)
        image_predictor = SAM2ImagePredictor(sam2_image_model)

        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(img)

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            masks = (masks > 0.0)
        
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks,  # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32))
        
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections,
                                                labels=[f"mouse{i}" for i in object_ids])
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        
        if save_path:
            cv2.imwrite(save_path, annotated_frame)
            
        return sv.mask_to_xyxy(masks), masks
    
    
    def infernence_video(self, input_video_path, output_video_path, 
                         box_save_path="boxes_per_frame.json"):

        rectangle_thickness=2
        text_thickness=1

        cap = cv2.VideoCapture(input_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # fps = 10

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps,
                                (frame_width, frame_height))

        all_boxes = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.predict(frame, conf=0.5)

            for result in results:
                for box in result.boxes:
                    cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), rectangle_thickness)
                    cv2.putText(frame, f"{result.names[int(box.cls[0])]}",
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), text_thickness)
            writer.write(frame)


            frame_boxes = []

            for result in results:
                for box in result.boxes:
                    box_info = {
                        "frame": frame_idx,
                        "class_id": int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                        "xyxy": [float(coord) for coord in box.xyxy[0]]
                    }
                    frame_boxes.append(box_info)

            all_boxes.append(frame_boxes)
            frame_idx += 1

        cap.release()
        writer.release()

        if box_save_path:

            with open(box_save_path, "w") as f:
                json.dump(all_boxes, f)






if __name__ == "__main__":

    ####### YOLO detector #######

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize YOLO detector
    yolo_model_path = "checkpoints_detection/yolo_v11_l.pt"
    detector = YOLODetector(yolo_model_path, conf=0.5, device=DEVICE)


    # demo1_path = "notebooks_SAM2-MICE/images/demo1.jpg"
    # img1 = cv2.imread(demo1_path)
    # results = detector.predict(img=img1, conf=0.5)
    # detector.infernence_image(img1, results, save_path="notebooks_SAM2-MICE/images/demo1_detected.jpg")


    demo1_path = "notebooks_SAM2-MICE/videos/open_field_five_mouse_frames/00002.jpg"
    img1 = cv2.imread(demo1_path)
    results = detector.predict(img=img1, conf=0.5)
    detector.infernence_image(img1, results, save_path="notebooks_SAM2-MICE/images/demo3_detected.jpg")


    # demo2_path = "notebooks_SAM2-MICE/images/demo1.jpg"
    # SAM2_CHECK_POINT = "./checkpoints/sam2_base_five_mouse_finetuned.pt"
    # SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    # img2 = cv2.imread(demo2_path)
    # results = detector.predict(img=img2, conf=0.5)
    # detector.infernence_image_with_SAM2(img2, results, sam2_checkpoint=SAM2_CHECK_POINT,
    #                                sam2_model_cfg=SAM2_MODEL_CONFIG,
    #                                save_path="notebooks_SAM2-MICE/images/demo1_detected_and_seged.jpg")
    

    # VIDEO_PATH = "notebooks_SAM2-MICE/videos/open_field_five_mouse.mp4"
    # OUTPUT_VIDEO_PATH = "notebooks_SAM2-MICE/videos/open_field_five_mouse_yolo_detect.mp4"

    # detector.infernence_video(VIDEO_PATH, OUTPUT_VIDEO_PATH, 
    #                             box_save_path="")
    

    #  ####### YOLO RE-ID tracker #######

    # tracker = MouseTracker(yolo_model_path, num_mice=5, max_missing_frames=50)
    
    # OUTPUT_VIDEO_PATH_REID = "notebooks_SAM2-MICE/videos/open_field_five_mouse_yolo_track.mp4"
    
    # # Run processing
    # tracker.process_video(VIDEO_PATH, OUTPUT_VIDEO_PATH_REID, conf=0.5, display=False)
    
    # # Clean up
    # tracker.close()


    # VIDEO_PATH = "/mnt/nas01/LAR/pico/Analysis/tracking/segmentation/test_sam2_mice/habitat_2/approach.mp4"
    # OUTPUT_VIDEO_PATH = "/mnt/nas01/LAR/pico/Analysis/tracking/segmentation/test_sam2_mice/habitat_2/approach_yolo.mp4"
    # detector.infernence_video(VIDEO_PATH, OUTPUT_VIDEO_PATH, 
    #                             box_save_path="")



    # demo1_path = "/mnt/nas01/LAR/pico/Experiments/habitat_ultra_new/SAM_behavior_seg/approach1/raw/00000.jpg"
    # img1 = cv2.imread(demo1_path)
    # results = detector.predict(img=img1, conf=0.5)
    # detector.infernence_image(img1, results, save_path="notebooks_SAM2-MICE/images/demo1_detected.jpg")
    



    


    



    



















    
