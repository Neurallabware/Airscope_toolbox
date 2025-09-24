import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from SAM2_Mice.detection import YOLODetector, sample_points_from_masks
from SAM2_Mice.utils import CommonUtils
from SAM2_Mice.utils import MaskDictionaryModel, ObjectInfo, create_video_from_images, VideoFrameExtractor
import json
import copy



def calculate_overlap_percentage(box1, box2):
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


def filter_overlapping_boxes(boxes, overlap_threshold=0.9):
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
            overlap = calculate_overlap_percentage(boxes[idx1], boxes[idx2])
            
            # If this box is mostly contained within another box, don't keep it
            if overlap > overlap_threshold:
                keep = False
                break
                
        if keep:
            keep_indices.append(idx1)
            
    return [boxes[i] for i in keep_indices]


def auto_tracking_with_sam2(
    video_path,
    frames_dir,
    output_dir="",
    sam2_checkpoint="./checkpoints/sam2_base_five_mouse_finetuned.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
    detection_ckpt_path="checkpoints_detection/yolo_v11_l.pt",
    prompt_type="mask",
    frame_step=30,
    frame_rate=30,
    detection_conf=0.5,
    iou_threshold=0.3,
    extract_frames=True,
    object_label="mouse"
):
    """
    Segment objects in a video using SAM2 and YOLO.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save output files
        sam2_checkpoint (str): Path to the SAM2 model checkpoint
        model_cfg (str): Path to the SAM2 model configuration
        detection_ckpt_path (str): Path to the YOLO detection model checkpoint
        prompt_type (str): Type of prompt for SAM2 video predictor ("mask", "box", or "point")
        frame_step (int): Step size for processing frames
        frame_rate (int): Frame rate for the output video
        detection_conf (float): Confidence threshold for object detection
        iou_threshold (float): IOU threshold for mask matching
        extract_frames (bool): Whether to extract frames from the video
        object_label (str): Label to assign to detected objects
    
    Returns:
        str: Path to the output video
    """
    # Setup CUDA and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # Turn on tfloat32 for better performance on Ampere or newer GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    if video_path:
        if not frames_dir:
            frames_dir = os.path.join(os.path.dirname(video_path),
                                            f"{os.path.splitext(os.path.basename(video_path))[0]}_frames")
            if not os.path.exists(frames_dir):
                    raise ValueError("Please make sure frames have been extracted into the default folder!")
        cap = cv2.VideoCapture(video_path)
        if len([file for file in os.listdir(frames_dir) if file.endswith(".jpg")]) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                raise ValueError("Please make sure extracted frame number is the same as video !")
        cap.release()
    elif not os.path.exists(frames_dir):
            raise ValueError("Please make sure the video frame folder exists!")

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(video_path), "segmentation_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results save dir {output_dir}")

    CommonUtils.creat_dirs(frames_dir)
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)
    CommonUtils.creat_dirs(result_dir)
    
    output_video_path = os.path.join(output_dir, "output.mp4")
    
    # Extract frames if requested
    if extract_frames:
        VideoFrameExtractor.extract_frames(video_path, frames_dir)
    
    # Initialize models
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    # Initialize detector
    detector = YOLODetector(detection_ckpt_path, conf=detection_conf)
    
    # Scan all frame names
    frame_names = [
        p for p in os.listdir(frames_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # Initialize video predictor state
    inference_state = video_predictor.init_state(
        video_path=frames_dir, 
        offload_video_to_cpu=True, 
        async_loading_frames=True
    )
    
    sam2_masks = MaskDictionaryModel()
    objects_count = 0
    
    print(f"Total frames: {len(frame_names)}")
    for start_frame_idx in range(0, len(frame_names), frame_step):
        print(f"Processing frame {start_frame_idx}")
        
        img_path = os.path.join(frames_dir, frame_names[start_frame_idx])
        image = Image.open(img_path)
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(promote_type=prompt_type, mask_name=f"mask_{image_base_name}.npy")
        
        img = cv2.imread(img_path)
        results = detector.predict(img=img, conf=detection_conf)
        
        # Process detection results
        input_boxes = results[0].boxes.xyxy.cpu().numpy()
        input_boxes = filter_overlapping_boxes(input_boxes)
        input_boxes = np.array(input_boxes)
        
        OBJECTS = []
        for _ in range(len(input_boxes)):
            OBJECTS.append(object_label)
        
        if input_boxes.shape[0] != 0:
            # Predict masks using SAM2
            image_predictor.set_image(np.array(image.convert("RGB")))
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            # Convert mask shape to (n, H, W)
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)
            
            # Add annotations based on prompt type
            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(device),
                    box_list=torch.tensor(input_boxes),
                    label_list=OBJECTS
                )
            else:
                raise NotImplementedError("SAM2 video predictor only supports mask prompts")
            
            # Update masks
            objects_count = mask_dict.update_masks_new(
                sam2_masks,
                matching_strategy="hybrid",
                iou_threshold=iou_threshold,
                objects_count=objects_count
            )
            print(f"Current object count: {objects_count}")
        else:
            print(f"No object detected in frame {frame_names[start_frame_idx]}, skipping merge")
            mask_dict = sam2_masks
        
        if len(mask_dict.labels) == 0:
            mask_dict.save_empty_mask_and_json(
                mask_data_dir,
                json_data_dir,
                image_name_list=frame_names[start_frame_idx:start_frame_idx+frame_step]
            )
            print(f"No object detected in frame {start_frame_idx}, skipping")
            continue
        else:
            video_predictor.reset_state(inference_state)
            
            # Add masks to video predictor
            for object_id, object_info in mask_dict.labels.items():
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )
            
            # for object_id, object_info in mask_dict.labels.items():
            #     mask_np = object_info.mask.cpu().numpy()
            #     mask_up = np.expand_dims(mask_np, axis=0)

            #     points = sample_points_from_masks(masks=mask_up, num_points=10)[0]
            #     labels = np.ones((points.shape[0]), dtype=np.int32)
                
            #     frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            #             inference_state=inference_state,
            #             frame_idx=start_frame_idx,
            #             obj_id=object_id,
            #             points=points,
            #             labels=labels,
            #         )

            # Propagate masks through video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                inference_state,
                max_frame_num_to_track=frame_step,
                start_frame_idx=start_frame_idx
            ):
                frame_masks = MaskDictionaryModel()
                
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)
                    object_info = ObjectInfo(
                        instance_id=out_obj_id,
                        mask=out_mask[0],
                        class_name=mask_dict.get_target_class_name(out_obj_id)
                    )
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]
                
                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)
            
            print(f"Processed {len(video_segments)} video segments")
            
            # Save tracking masks and JSON files
            for frame_idx, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                for obj_id, obj_info in mask.items():
                    mask_img[obj_info.mask == True] = obj_id
                
                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)
                
                json_data = frame_masks_info.to_dict()
                json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)
    
    # Draw results and save video
    CommonUtils.draw_masks_and_box_with_supervision(frames_dir, mask_data_dir, json_data_dir, result_dir)
    create_video_from_images(result_dir, output_video_path, frame_rate=frame_rate)
    

if __name__ == "__main__":


    video_path = "notebooks_SAM2-MICE/videos/open_field_three_mice_not_continous.mp4"
    video_dir = "notebooks_SAM2-MICE/videos/open_field_three_mice_not_continous/frames"
    VideoFrameExtractor.extract_frames(video_path, video_dir)
    # 'output_dir' is the directory to save the annotated frames
    output_dir = "notebooks_SAM2-MICE/videos/open_field_three_mice_not_continous/seg"
    iou_threshold = 0.3


    # video_path = "/mnt/nas01/LAR/pico/Analysis/tracking/segmentation/test_sam2_mice/habitat_2/approach.mp4"
    # video_dir = "notebooks_SAM2-MICE/videos/habitat_not_continouse/frames"
    # output_dir = "notebooks_SAM2-MICE/videos/habitat_not_continouse/seg4"
    # output_video_path = "notebooks_SAM2-MICE/videos/habitat_not_continouse/seg4/output.mp4"
    # iou_threshold = 0.3


    auto_tracking_with_sam2(
        video_path=video_path,
        frames_dir=video_dir,
        output_dir=output_dir,
        sam2_checkpoint="./checkpoints/sam2_base_five_mouse_finetuned.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
        detection_ckpt_path="checkpoints_detection/yolo_v11_l.pt",
        prompt_type="mask",
        frame_step=30,
        frame_rate=30,
        detection_conf=0.5,
        iou_threshold=iou_threshold,
        
        extract_frames=True,
        
        object_label="mouse"
    )
