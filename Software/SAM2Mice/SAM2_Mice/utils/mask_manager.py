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
    A class to manage video segments, including saving, loading, and visualizing masks.
    """

    def __init__(self):
        self.video_segments = []
        self.frame_paths = []

    def add_segment(self, segment):
        """Add a segment to the list of video segments."""
        self.video_segments.append(segment)

    def add_frame_path(self, path):
        """Add a frame path to the list of frame paths."""
        self.frame_paths.append(path)

    def extend_segments(self, segments):
        """Extend the list of video segments."""
        self.video_segments.extend(segments)

    def extend_frame_paths(self, paths):
        """Extend the list of frame paths."""
        self.frame_paths.extend(paths)

    def get_segments(self):
        """Get the list of video segments."""
        return self.video_segments

    def get_frame_paths(self):
        """Get the list of frame paths."""
        return self.frame_paths

    def clear(self):
        """Clear all video segments and frame paths."""
        self.video_segments.clear()
        self.frame_paths.clear()
        print("All video segments and frame paths have been cleared.")
    
    def get_len(self):
        
        assert len(self.video_segments) == len(self.frame_paths)
        
        return len(self.video_segments)


    def save_video_segments(self, file_name):
        """
        Save video segments to a compressed pickle file.

        Args:
            file_name (str): Path to save the compressed pickle file.

        Returns:
            tuple: Shape of the mask (height, width).
        """
        processed_segments = []
        shape = None

        for segment in self.video_segments:
            if segment is None:
                processed_segments.append(None)
                continue

            processed_segment = {}
            for obj_id, mask in segment.items():
                mask = np.squeeze(mask, axis=0)
                if shape is None and mask is not None:
                    shape = mask.shape
                mask = np.where(mask > 0, 1, 0).astype(np.int8)
                # Convert mask to binary and then to bytes
                binary_mask = np.packbits(mask, axis=None)
                processed_segment[obj_id] = binary_mask
            processed_segments.append(processed_segment)

        # Compress with gzip
        with gzip.open(file_name, 'wb') as f:
            pickle.dump(processed_segments, f)
        print(f"{file_name} saved successfully!")
        return shape

    def load_video_segments(self, file_name, shape):
        """
        Load video segments from a compressed pickle file.

        Args:
            file_name (str): Path to the compressed pickle file.
            shape (tuple): Shape of the mask (height, width).

        Returns:
            list: List of video segments.
        """
        with gzip.open(file_name, 'rb') as f:
            processed_segments = pickle.load(f)

        self.video_segments = []
        for segment in processed_segments:
            if segment is None:
                self.video_segments.append(None)
                continue

            decompressed_segment = {}
            for obj_id, binary_mask in segment.items():
                # Convert bytes back to mask
                mask = np.unpackbits(binary_mask).astype(np.int8)
                mask = mask.reshape(shape)
                mask = np.expand_dims(mask, axis=0)  # h*w to 1*h*w
                decompressed_segment[obj_id] = mask
            self.video_segments.append(decompressed_segment)

        print(f"Successfully loaded {file_name}!")
        return self.video_segments

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
        frame_size = None

        def process_frame(out_frame_idx, frame_path):
            nonlocal frame_size
            masked_img_path = os.path.join(temp_folder, f'{out_frame_idx:05d}.jpg')

            if self.video_segments[out_frame_idx] is None:
                shutil.copy(frame_path, masked_img_path)
                if frame_size is None:
                    img = cv2.imread(frame_path)
                    frame_size = (img.shape[1], img.shape[0])  # Width, Height
                return masked_img_path

            img = Image.open(frame_path)
            for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                mask_image = self.generate_mask(out_mask, obj_id=out_obj_id)
                masked_img_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
                masked_img = Image.alpha_composite(img.convert('RGBA'), masked_img_pil.convert('RGBA'))
                img = masked_img.convert('RGB')

            masked_img = img.convert('RGB')
            masked_img.save(masked_img_path)

            if frame_size is None:
                frame_size = masked_img.size  # (width, height)
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

        os.makedirs(mask_save_folder, exist_ok=True)

        for frame_idx in tqdm(range(len(self.video_segments))):
            segments = self.video_segments[frame_idx]

            if segments is None:
                shutil.copy(self.frame_paths[frame_idx], os.path.join(mask_save_folder, f"{frame_idx:05d}.jpg"))
                continue

            img = cv2.imread(self.frame_paths[frame_idx])

            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks,  # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections,
                                                    labels=[f"mouse{i}" for i in object_ids])
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(mask_save_folder, f"{frame_idx:05d}.jpg"), annotated_frame)

        create_video_from_images(mask_save_folder, output_video_path, frame_rate=fps)


import json
import torch
from dataclasses import dataclass, field


@dataclass
class MaskDictionaryModel:
    mask_name:str = ""
    mask_height: int = 1080
    mask_width:int = 1920
    promote_type:str = "mask"
    labels:dict = field(default_factory=dict)

    def add_new_frame_annotation(self, mask_list, box_list, label_list, background_value = 0):
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
        更新掩码并保持实例ID的一致性
        
        参数:
        - tracking_annotation_dict: 先前的跟踪注释字典
        - matching_strategy: 用于匹配掩码的策略，可选：
            - "mask_iou": 仅使用掩码的IoU
            - "box_iou": 仅使用边界框的IoU
            - "hybrid": 结合掩码IoU和边界框IoU
            - "center_dist": 结合边界框IoU和中心点距离
        - iou_threshold: IoU阈值，高于此值认为是同一对象
        - center_weight: 在"center_dist"策略中，中心点距离的权重
        - objects_count: 初始对象计数
        
        返回:
        - 更新后的对象计数
        """
        updated_masks = {}

        for seg_obj_id, seg_mask in self.labels.items():
            flag = 0 
            new_mask_copy = ObjectInfo()
            if seg_mask.mask.sum() == 0:
                continue
            
            # 如果当前掩码没有边界框信息，计算它
            if seg_mask.x1 == 0 and seg_mask.x2 == 0 and seg_mask.y1 == 0 and seg_mask.y2 == 0:
                seg_mask.update_box()
            
            # 计算当前掩码的中心点
            current_center_x = (seg_mask.x1 + seg_mask.x2) / 2
            current_center_y = (seg_mask.y1 + seg_mask.y2) / 2
            
            best_score = -1
            best_object_id = None
            
            for object_id, object_info in tracking_annotation_dict.labels.items():
                # 如果追踪对象没有边界框信息，计算它
                if object_info.x1 == 0 and object_info.x2 == 0 and object_info.y1 == 0 and object_info.y2 == 0:
                    object_info.update_box()
                
                # 根据选择的策略计算得分
                score = 0
                
                if matching_strategy == "mask_iou":
                    # 选项1: 仅使用掩码IoU
                    score = self.calculate_iou(seg_mask.mask, object_info.mask)
                    
                elif matching_strategy == "box_iou":
                    # 选项2: 仅使用边界框IoU
                    score = self.calculate_box_iou(seg_mask, object_info)
                    
                elif matching_strategy == "hybrid":
                    # 选项3: 掩码IoU和边界框IoU的组合
                    mask_iou = self.calculate_iou(seg_mask.mask, object_info.mask)
                    box_iou = self.calculate_box_iou(seg_mask, object_info)
                    score = (mask_iou + box_iou) / 2
                    
                elif matching_strategy == "center_dist":
                    # 选项4: 边界框IoU结合中心点距离
                    box_iou = self.calculate_box_iou(seg_mask, object_info)
                    
                    # 计算中心点
                    obj_center_x = (object_info.x1 + object_info.x2) / 2
                    obj_center_y = (object_info.y1 + object_info.y2) / 2
                    
                    # 计算归一化中心点距离 (越小越好)
                    max_dim = max(self.mask_width, self.mask_height)
                    center_dist = torch.sqrt(((current_center_x - obj_center_x) ** 2 + 
                                            (current_center_y - obj_center_y) ** 2)) / max_dim
                    
                    # 转换为相似度分数 (0-1范围，越大越好)
                    center_similarity = 1 - min(center_dist, 1)
                    
                    # 加权组合
                    score = (1 - center_weight) * box_iou + center_weight * center_similarity
                
                # 更新最佳匹配
                if score > best_score and score > iou_threshold:
                    # best_score = score
                    best_object_id = object_info.instance_id
            
            # 如果找到匹配，使用现有ID
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
                # 否则分配新ID
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
        计算两个边界框的IoU
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
        return self.labels[instance_id].class_name

    def get_target_logit(self, instance_id):
        return self.labels[instance_id].logit
        
    @staticmethod
    def calculate_iou(mask1, mask2):
        # Convert masks to float tensors for calculations
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)
            
        # Calculate intersection and union
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
            
        # Calculate IoU
        iou = intersection / union
        return iou

    def save_empty_mask_and_json(self, mask_data_dir, json_data_dir, image_name_list=None):
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
        return {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }
    
    def to_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def from_json(self, json_file):
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
    mask: any = None
    class_name:str = ""
    x1:int = 0
    y1:int = 0
    x2:int = 0
    y2:int = 0
    logit:float = 0.0

    def get_mask(self):
        return self.mask
    
    def get_id(self):
        return self.instance_id

    def update_box(self):
        # 找到所有非零值的索引
        nonzero_indices = torch.nonzero(self.mask)
        
        # 如果没有非零值，返回一个空的边界框
        if nonzero_indices.size(0) == 0:
            # print("nonzero_indices", nonzero_indices)
            return []
        
        # 计算最小和最大索引
        y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
        y_max, x_max = torch.max(nonzero_indices, dim=0)[0]
        
        # 创建边界框 [x_min, y_min, x_max, y_max]
        bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]        
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
    
    def to_dict(self):
        return {
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "logit": self.logit
        }