import gzip
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2
import shutil
import supervision as sv


def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    """Create an MP4 video from sorted image files in a folder."""
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")



class VideoSegmentManager:
    """
    Manages video segmentation masks with in-memory packbits compression.

    Internally every mask is stored as a flat uint8 packbits array (identical to
    the on-disk format), so RAM usage is ~8x lower than storing bool/float32 arrays.
    Masks are decompressed on-the-fly only when needed for rendering or saving.
    The on-disk pickle format (save/load) is unchanged.
    """

    def __init__(self):
        """Initialize empty segment and frame-path storage."""
        # Each entry is None  OR  {obj_id: uint8 packbits array}
        self.video_segments = []
        self.frame_paths = []
        # (H, W) of the masks; filled on first non-None segment
        self._mask_shape: tuple = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pack(self, mask) -> np.ndarray:
        """Compress a single (1,H,W) or (H,W) bool/float mask to packbits uint8."""
        mask = np.squeeze(mask)          # → (H, W)
        if self._mask_shape is None:
            self._mask_shape = mask.shape
        return np.packbits(mask.astype(bool), axis=None)

    def _unpack(self, bits: np.ndarray) -> np.ndarray:
        """Decompress packbits → (1, H, W) int8, matching the original format."""
        flat = np.unpackbits(bits).astype(np.int8)
        h, w = self._mask_shape
        return flat[: h * w].reshape(1, h, w)

    def _pack_segment(self, segment):
        """Convert a raw {obj_id: mask_array} dict to packed storage."""
        if segment is None:
            return None
        return {obj_id: self._pack(mask) for obj_id, mask in segment.items()}

    def _decode_segment(self, packed):
        """Decompress a packed segment dict back to {obj_id: (1,H,W) int8}."""
        if packed is None:
            return None
        return {obj_id: self._unpack(bits) for obj_id, bits in packed.items()}

    # ------------------------------------------------------------------
    # Public API  (unchanged signatures)
    # ------------------------------------------------------------------

    def add_segment(self, segment):
        """Append one frame's object masks after packbits compression."""
        self.video_segments.append(self._pack_segment(segment))

    def add_frame_path(self, path):
        """Append the source image path associated with one segment frame."""
        self.frame_paths.append(path)

    def extend_segments(self, segments):
        """Append multiple frame segments after packbits compression."""
        for seg in segments:
            self.video_segments.append(self._pack_segment(seg))

    def extend_frame_paths(self, paths):
        """Append multiple source frame paths."""
        self.frame_paths.extend(paths)

    def get_segments(self):
        """Return fully decompressed segments (same format as before)."""
        return [self._decode_segment(s) for s in self.video_segments]

    def get_frame_paths(self):
        """Return the stored source frame paths."""
        return self.frame_paths

    def clear(self):
        """Remove all stored frame paths, masks, and cached mask shape."""
        self.video_segments.clear()
        self.frame_paths.clear()
        self._mask_shape = None
        print("All video segments and frame paths have been cleared.")

    def get_len(self):
        """Return the synchronized number of stored frames and segments."""
        if len(self.video_segments) != len(self.frame_paths):
            raise ValueError(
                f"Segment count ({len(self.video_segments)}) != "
                f"frame path count ({len(self.frame_paths)})"
            )
        return len(self.video_segments)

    # ------------------------------------------------------------------
    # Disk I/O  (on-disk format unchanged)
    # ------------------------------------------------------------------

    def save_video_segments(self, file_name):
        """Save to a gzip-compressed pickle; disk format is identical to before."""
        processed_segments = []
        shape = self._mask_shape  # already known from add/extend

        for packed in self.video_segments:
            if packed is None:
                processed_segments.append(None)
                continue
            # Already packbits — write directly without re-allocating the full array
            processed_segments.append(packed)

        with gzip.open(file_name, 'wb') as f:
            pickle.dump(processed_segments, f)
        print(f"{file_name} saved successfully!")
        return shape

    def load_video_segments(self, file_name, shape):
        """Load from a gzip-compressed pickle; returns decompressed segments."""
        with gzip.open(file_name, 'rb') as f:
            processed_segments = pickle.load(f)

        self._mask_shape = shape
        # Keep data in packed form in memory
        self.video_segments = processed_segments  # list of None | {obj_id: uint8}
        print(f"Successfully loaded {file_name}!")
        # Return decompressed view for callers that iterate immediately
        return [self._decode_segment(s) for s in self.video_segments]

    @staticmethod
    def generate_mask(mask, obj_id=None, random_color=False):
        """
        Generate a mask image.

        Args:
            mask (numpy.ndarray): Binary mask.
            obj_id (int, optional): Object ID for colormap selection. Defaults to None.
            random_color (bool, optional): Whether to use a random color. Defaults to False.

        Returns:
            numpy.ndarray: Mask image.
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image
    
    
    def generate_masked_video_and_image(self, output_video_path, fps, temp_folder, save_masks=True):
        """
        Generate a video from masked frames and optionally save the masked images.

        Args:
            output_video_path (str): Path to save the output video.
            fps (int): Frames per second for the output video.
            temp_folder (str): Path to save temporary masked images.
            save_masks (bool, optional): Whether to save the masked images. Defaults to True.
        """
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        masked_image_paths = []

        # Determine frame_size before spawning threads to avoid a shared-variable race condition.
        first_frame = cv2.imread(self.frame_paths[0])
        frame_size = (first_frame.shape[1], first_frame.shape[0])  # (width, height)

        def process_frame(out_frame_idx, frame_path):
            """Render and save one masked frame for later video assembly."""
            masked_img_path = os.path.join(temp_folder, f'{out_frame_idx:05d}.jpg')

            packed = self.video_segments[out_frame_idx]
            if packed is None:
                shutil.copy(frame_path, masked_img_path)
                return masked_img_path

            segment = self._decode_segment(packed)
            img = Image.open(frame_path)
            for out_obj_id, out_mask in segment.items():
                mask_image = self.generate_mask(out_mask, obj_id=out_obj_id)
                masked_img_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
                masked_img = Image.alpha_composite(img.convert('RGBA'), masked_img_pil.convert('RGBA'))
                img = masked_img.convert('RGB')

            masked_img = img.convert('RGB')
            masked_img.save(masked_img_path)
            return masked_img_path

        print("Generating masks in parallel...")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_frame, idx, path) for idx, path in enumerate(self.frame_paths)]
            for future in tqdm(futures, total=len(self.frame_paths), desc="Mask Generation"):
                masked_image_paths.append(future.result())

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        print("Writing frames to video...")
        for img_path in tqdm(masked_image_paths, desc="Video Writing"):
            frame = cv2.imread(img_path)
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()

        # Optionally, clean up the temporary folder after video is created
        if not save_masks:
            for path in masked_image_paths:
                os.remove(path)
            shutil.rmtree(temp_folder)

    
    def generate_masked_video_supervision(self, output_video_path, fps, mask_save_folder):
        """Render mask, box, and label overlays with supervision and save a video."""

        os.makedirs(mask_save_folder, exist_ok=True)

        for frame_idx in tqdm(range(len(self.video_segments))):
            packed = self.video_segments[frame_idx]

            if packed is None:
                shutil.copy(self.frame_paths[frame_idx], os.path.join(mask_save_folder, f"{frame_idx:05d}.jpg"))
                continue

            segments = self._decode_segment(packed)
            img = cv2.imread(self.frame_paths[frame_idx])

            object_ids = []
            masks = []
            for obj_id, mask in segments.items():
                mask = np.squeeze(mask).astype(bool)
                if mask.any():
                    object_ids.append(obj_id)
                    masks.append(mask)

            if not masks:
                shutil.copy(self.frame_paths[frame_idx], os.path.join(mask_save_folder, f"{frame_idx:05d}.jpg"))
                continue

            masks = np.stack(masks, axis=0)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks,  # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections,
                                                    labels=[f"mouse{i}" for i in object_ids])
            cv2.imwrite(os.path.join(mask_save_folder, f"{frame_idx:05d}.jpg"), annotated_frame)

        create_video_from_images(mask_save_folder, output_video_path, frame_rate=fps)


import json
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MaskDictionaryModel:
    mask_name:str = ""
    mask_height: int = 1080
    mask_width:int = 1920
    promote_type:str = "mask"
    labels:dict = field(default_factory=dict)

    def add_new_frame_annotation(self, mask_list, box_list, label_list, background_value = 0):
        """Build per-object annotations for a frame from masks, boxes, and labels."""
        mask_img = torch.zeros(mask_list.shape[-2:])
        anno_2d = {}
        for idx, (mask, box, label) in enumerate(zip(mask_list, box_list, label_list)):
            final_index = background_value + idx + 1

            if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            # mask = mask
            mask_img[mask == True] = final_index
            # print("label", label)
            name = label
            box = box # .numpy().tolist()
            new_annotation = ObjectInfo(instance_id = final_index, mask = mask, class_name = name, x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3])
            anno_2d[final_index] = new_annotation

        # np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))
        self.mask_height = mask_img.shape[0]
        self.mask_width = mask_img.shape[1]
        self.labels = anno_2d

    def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
        """Update current object IDs by matching masks against previous annotations."""
        updated_masks = {}

        for seg_obj_id, seg_mask in self.labels.items():  # tracking_masks
            flag = 0 
            new_mask_copy = ObjectInfo()
            if seg_mask.mask.sum() == 0:
                continue
            
            for object_id, object_info in tracking_annotation_dict.labels.items():  # grounded_sam masks
                iou = self.calculate_iou(seg_mask.mask, object_info.mask)  # tensor, numpy
                # print("iou", iou)
                if iou > iou_threshold:
                    flag = object_info.instance_id
                    new_mask_copy.mask = seg_mask.mask
                    new_mask_copy.instance_id = object_info.instance_id
                    new_mask_copy.class_name = seg_mask.class_name
                    break
                
            if not flag:
                objects_count += 1
                flag = objects_count
                new_mask_copy.instance_id = objects_count
                new_mask_copy.mask = seg_mask.mask
                new_mask_copy.class_name = seg_mask.class_name
            updated_masks[flag] = new_mask_copy
        self.labels = updated_masks
        return objects_count
    
    def update_masks_new(self, tracking_annotation_dict, matching_strategy="mask_iou", iou_threshold=0.8, center_weight=0.5, objects_count=0):
        """
        Update masks while keeping instance IDs consistent.
        
        Args:
        - tracking_annotation_dict: Previous tracking annotation dictionary.
        - matching_strategy: Strategy used to match masks. Options:
            - "mask_iou": Use mask IoU only.
            - "box_iou": Use bounding box IoU only.
            - "hybrid": Combine mask IoU and bounding box IoU.
            - "center_dist": Combine bounding box IoU and center-point distance.
        - iou_threshold: IoU threshold above which masks are treated as the same object.
        - center_weight: Weight of the center-point distance in the "center_dist" strategy.
        - objects_count: Initial object count.
        
        Returns:
        - Updated object count.
        """
        updated_masks = {}

        for seg_obj_id, seg_mask in self.labels.items():
            flag = 0 
            new_mask_copy = ObjectInfo()
            if seg_mask.mask.sum() == 0:
                continue
            
            # Compute the bounding box if the current mask does not have one.
            if seg_mask.x1 == 0 and seg_mask.x2 == 0 and seg_mask.y1 == 0 and seg_mask.y2 == 0:
                seg_mask.update_box()
            
            # Compute the center point of the current mask.
            current_center_x = (seg_mask.x1 + seg_mask.x2) / 2
            current_center_y = (seg_mask.y1 + seg_mask.y2) / 2
            
            best_score = -1
            best_object_id = None
            
            for object_id, object_info in tracking_annotation_dict.labels.items():
                # Compute the bounding box if the tracked object does not have one.
                if object_info.x1 == 0 and object_info.x2 == 0 and object_info.y1 == 0 and object_info.y2 == 0:
                    object_info.update_box()
                
                # Compute the score according to the selected strategy.
                score = 0
                
                if matching_strategy == "mask_iou":
                    # Option 1: Use mask IoU only.
                    score = self.calculate_iou(seg_mask.mask, object_info.mask)
                    
                elif matching_strategy == "box_iou":
                    # Option 2: Use bounding box IoU only.
                    score = self.calculate_box_iou(seg_mask, object_info)
                    
                elif matching_strategy == "hybrid":
                    # Option 3: Combine mask IoU and bounding box IoU.
                    mask_iou = self.calculate_iou(seg_mask.mask, object_info.mask)
                    box_iou = self.calculate_box_iou(seg_mask, object_info)
                    score = (mask_iou + box_iou) / 2
                    
                elif matching_strategy == "center_dist":
                    # Option 4: Combine bounding box IoU with center-point distance.
                    box_iou = self.calculate_box_iou(seg_mask, object_info)
                    
                    # Compute the center point.
                    obj_center_x = (object_info.x1 + object_info.x2) / 2
                    obj_center_y = (object_info.y1 + object_info.y2) / 2
                    
                    # Compute normalized center-point distance. Lower is better.
                    max_dim = max(self.mask_width, self.mask_height)
                    center_dist = torch.sqrt(((current_center_x - obj_center_x) ** 2 + 
                                            (current_center_y - obj_center_y) ** 2)) / max_dim
                    
                    # Convert to a similarity score in the 0-1 range. Higher is better.
                    center_similarity = 1 - min(center_dist, 1)
                    
                    # Weighted combination.
                    score = (1 - center_weight) * box_iou + center_weight * center_similarity
                
                # Update the best match.
                if score > best_score and score > iou_threshold:
                    # best_score = score
                    best_object_id = object_info.instance_id
            
            # Use the existing ID if a match is found.
            if best_object_id is not None:
                flag = best_object_id
                new_mask_copy.mask = seg_mask.mask
                new_mask_copy.instance_id = best_object_id
                new_mask_copy.class_name = seg_mask.class_name
                new_mask_copy.x1 = seg_mask.x1
                new_mask_copy.y1 = seg_mask.y1
                new_mask_copy.x2 = seg_mask.x2
                new_mask_copy.y2 = seg_mask.y2
            else:
                # Otherwise assign a new ID.
                objects_count += 1
                flag = objects_count
                new_mask_copy.instance_id = objects_count
                new_mask_copy.mask = seg_mask.mask
                new_mask_copy.class_name = seg_mask.class_name
                new_mask_copy.x1 = seg_mask.x1
                new_mask_copy.y1 = seg_mask.y1
                new_mask_copy.x2 = seg_mask.x2
                new_mask_copy.y2 = seg_mask.y2
                
            updated_masks[flag] = new_mask_copy
        
        self.labels = updated_masks
        return objects_count

    @staticmethod
    def calculate_box_iou(mask1_info, mask2_info):
        """
        Calculate the IoU between two bounding boxes.
        """
        box1 = [mask1_info.x1, mask1_info.y1, mask1_info.x2, mask1_info.y2]
        box2 = [mask2_info.x1, mask2_info.y1, mask2_info.x2, mask2_info.y2]
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0

    def get_target_class_name(self, instance_id):
        """Return the class name for an instance ID."""
        return self.labels[instance_id].class_name

    def get_target_logit(self, instance_id):
        """Return the stored confidence/logit for an instance ID."""
        return self.labels[instance_id].logit
        
    @staticmethod
    def calculate_iou(mask1, mask2):
        """Calculate mask IoU between two torch tensors."""
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
        return (intersection / union).item() if union > 0 else 0.0

    def save_empty_mask_and_json(self, mask_data_dir, json_data_dir, image_name_list=None):
        """Write empty mask arrays and matching JSON metadata files."""
        mask_img = torch.zeros((self.mask_height, self.mask_width))
        if image_name_list:
            for image_base_name in image_name_list:
                image_base_name = image_base_name.split(".")[0]+".npy"
                mask_name = "mask_"+image_base_name
                np.save(os.path.join(mask_data_dir, mask_name), mask_img.numpy().astype(np.uint16))

                json_data_path = os.path.join(json_data_dir, mask_name.replace(".npy", ".json"))
                print("save_empty_mask_and_json", json_data_path)
                self.to_json(json_data_path)
        else:
            np.save(os.path.join(mask_data_dir, self.mask_name), mask_img.numpy().astype(np.uint16))
            json_data_path = os.path.join(json_data_dir, self.mask_name.replace(".npy", ".json"))
            print("save_empty_mask_and_json", json_data_path)
            self.to_json(json_data_path)


    def to_dict(self):
        """Serialize this mask dictionary to plain Python objects."""
        return {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }
    
    def to_json(self, json_file):
        """Write this mask dictionary to a JSON file."""
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def from_json(self, json_file):
        """Load mask dictionary metadata from a JSON file."""
        with open(json_file, "r") as f:
            data = json.load(f)
            self.mask_name = data["mask_name"]
            self.mask_height = data["mask_height"]
            self.mask_width = data["mask_width"]
            self.promote_type = data["promote_type"]
            self.labels = {int(k): ObjectInfo(**v) for k, v in data["labels"].items()}
        return self


@dataclass
class ObjectInfo:
    instance_id:int = 0
    mask: Optional[torch.Tensor] = None
    class_name:str = ""
    x1:int = 0
    y1:int = 0
    x2:int = 0
    y2:int = 0
    logit:float = 0.0

    def get_mask(self):
        """Return the stored object mask tensor."""
        return self.mask
    
    def get_id(self):
        """Return the object instance ID."""
        return self.instance_id

    def update_box(self):
        """Update bounding-box coordinates from the nonzero mask region."""
        # Find the indices of all nonzero values.
        nonzero_indices = torch.nonzero(self.mask)
        
        # Return an empty bounding box if there are no nonzero values.
        if nonzero_indices.size(0) == 0:
            # print("nonzero_indices", nonzero_indices)
            return []
        
        # Compute the minimum and maximum indices.
        y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
        y_max, x_max = torch.max(nonzero_indices, dim=0)[0]
        
        # Create the bounding box [x_min, y_min, x_max, y_max].
        bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]        
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
    
    def to_dict(self):
        """Serialize object metadata to plain Python objects."""
        return {
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "logit": self.logit
        }
