import os
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from labelme import utils
from ultralytics import YOLO
from SAM2_Mice.detection import YOLODetector, sample_points_from_masks
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from SAM2_Mice.utils.frame_extractor import VideoFrameExtractor
from SAM2_Mice.utils.mask_manager import VideoSegmentManager


class VideoSegmentationInference:
    """Class for performing video segmentation inference with SAM2."""

    def __init__(self, model_cfg, checkpoint_path, vos_optimized=False):
        """Initialize the video segmentation inference.

        Args:
            model_cfg (str): Path to the model configuration file.
            checkpoint_path (str): Path to the model checkpoint.
            vos_optimized (bool, optional): Whether to use VOS optimization. Defaults to False.
        """
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, vos_optimized=vos_optimized)
        self.inference_state = None
        self.segment_manager = VideoSegmentManager()
        self.manual_prompts = {}

    def reset(self):
        """Reset the inference state and other attributes."""
        # Clean up
        if self.predictor:
            self.predictor.reset_state(self.inference_state)

        # Clear the segment manager and manual prompts
        self.segment_manager.clear()
        self.manual_prompts.clear()

        self.inference_state = None


    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        """Display a mask on a matplotlib axis.

        Args:
            mask (numpy.ndarray): Binary mask.
            ax (matplotlib.axes.Axes): Matplotlib axis to display the mask on.
            obj_id (int, optional): Object ID for colormap selection. Defaults to None.
            random_color (bool, optional): Whether to use a random color. Defaults to False.
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        """Display positive and negative points on a matplotlib axis.

        Args:
            coords (numpy.ndarray): Array of point coordinates.
            labels (numpy.ndarray): Array of point labels (1 for positive, 0 for negative).
            ax (matplotlib.axes.Axes): Matplotlib axis to display the points on.
            marker_size (int, optional): Size of the marker. Defaults to 200.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
    
    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def extract_frames_before_seg(self, video_path=None, frames_dir=None,):

        # Handle inputs
        if video_path and not frames_dir:
            # Extract frames from video
            frames_dir = os.path.join(os.path.dirname(video_path),
                                      f"{os.path.splitext(os.path.basename(video_path))[0]}")
            VideoFrameExtractor.extract_frames(video_path, frames_dir)

        if not frames_dir or not os.path.exists(frames_dir):
            raise ValueError("Either video_path or a valid frames_dir must be provided")


    def run(self, video_path=None, frames_dir=None, prompt_source="detection", 
            detection_ckpt_path="", prompt_type="points", save_dir=None, fps=10):
        
        """Run video segmentation inference.

        Args:
            video_path (str, optional): Path to the video file. If provided, frames will be extracted.
            frames_dir (str, optional): Directory containing video frames and annotations.
            prompt_type (str, optional): Type of prompts to use, either "points" or "mask". Defaults to "points".
            save_dir (str, optional): Directory to save results. Defaults to a subdirectory of frames_dir.
            fps (int, optional): Frames per second for output video. Defaults to 10.

        Returns:
            tuple: (video_segments, output_video_path)
        """
        assert prompt_source in ["detection", "manual"], "SAM2-MICE base only support prompt from automatical detection or external manual prompts"

        if video_path:
            if frames_dir is None:
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

        if not save_dir:
            save_dir = os.path.join(os.path.dirname(video_path), "segmentation_results")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results save dir {save_dir}")

        # Load frame paths
        frame_names = [
            p for p in os.listdir(frames_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        frame_paths = [os.path.join(frames_dir, name) for name in frame_names]

        # Initialize segment manager with frame paths
        self.segment_manager.extend_frame_paths(frame_paths)

        if prompt_source=="detection":
            
            return self._run_with_automatical_prompts_from_detection(frames_dir=frames_dir, frame_paths=frame_paths, 
                                                                     detection_ckpt_path=detection_ckpt_path, prompt_type=prompt_type, 
                                                                     save_dir=save_dir, fps=fps)

        elif prompt_source=="manual":

            # Get prompts based on prompt type
            if prompt_type.lower() == "point":
                return self._run_with_point_prompts(frames_dir, frame_paths, save_dir, fps)
            elif prompt_type.lower() == "mask":
                return self._run_with_mask_prompts(frames_dir, frame_paths, save_dir, fps)
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}. Use 'point' or 'mask'")

    def _run_with_point_prompts(self, frames_dir, frame_paths, save_dir, fps):
        """Run inference using point prompts."""
        # Gather point prompts from JSON files

        for file_name in os.listdir(frames_dir):
            if file_name.endswith(".json"):
                # Extract frame number (labelme annotations format like 00000.json)
                frame_number = int(file_name.split(".")[0])
                with open(os.path.join(frames_dir, file_name), 'r') as f:
                    jsonx = json.load(f)
                    for item in jsonx['shapes']:
                        ann_obj_id = item["group_id"]  # Corresponds to ann_obj_id
                        points = np.array(item["points"], dtype=np.float32)
                        labels = np.ones(len(points), np.int32)

                        if frame_number not in list(self.manual_prompts.keys()):
                            self.manual_prompts[frame_number] = {}

                        self.manual_prompts[frame_number][ann_obj_id] = (points, labels)

        if not self.manual_prompts:
            raise ValueError("No point prompts found in JSON files")
        
        sorted_keys = sorted(self.manual_prompts.keys())
        self.manual_prompts = {key: self.manual_prompts[key] for key in sorted_keys}

        print(f"Found point prompts in frames: {list(self.manual_prompts.keys())}")

        # Initialize state
        self.inference_state = self.predictor.init_state(video_path=frames_dir)

        # Visualize prompts
        os.makedirs(os.path.join(save_dir, "prompts"), exist_ok=True)
        for frame_index, prompt_frame in self.manual_prompts.items():
            for ann_obj_id, (points, labels) in prompt_frame.items():
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                    inference_state=self.inference_state,
                    frame_idx=frame_index,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )

            plt.figure(figsize=(12, 8))
            plt.title(f"Frame {frame_index} with Point Prompts")
            plt.imshow(Image.open(os.path.join(frames_dir, f"{frame_index:05d}.jpg")))
            for i, out_obj_id in enumerate(out_obj_ids):
                self.show_points(*prompt_frame[out_obj_id], plt.gca())
                self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
            img_save_path = os.path.join(save_dir, "prompts", f"frame{frame_index:05d}.jpg")
            plt.savefig(img_save_path)
            plt.close()

        # Run propagation and collect results
        video_segments = []
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments.append({
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            })

        # Handle potential frame count mismatch
        frame_interval = len(frame_paths)
        if len(video_segments) < frame_interval:
            print(f"Padding {frame_interval - len(video_segments)} missing frames with None")
            padded_segments = [None] * (frame_interval - len(video_segments))
            padded_segments.extend(video_segments)
            video_segments = padded_segments

        # Update segment manager with results
        self.segment_manager.extend_segments(video_segments)

        # Generate output video
        output_video_path = os.path.join(save_dir, "segmented_video.mp4")
        temp_folder = os.path.join(save_dir, "frames")
        self.segment_manager.generate_masked_video_and_image(output_video_path, fps, temp_folder)

        # Save segments for later use
        pickle_save_path = os.path.join(save_dir, "segment_masks.pickle")
        shape = self.segment_manager.save_video_segments(pickle_save_path)

        # Clean up
        self.predictor.reset_state(self.inference_state)

        return video_segments, output_video_path

    def _run_with_mask_prompts(self, frames_dir, frame_paths, save_dir, fps):
        """Run inference using mask prompts."""
        # Gather mask prompts from JSON files

        for file_name in os.listdir(frames_dir):
            if file_name.endswith(".json"):
                # Extract frame number
                frame_number = int(file_name.split(".")[0])

                if frame_number not in list(self.manual_prompts.keys()):
                    self.manual_prompts[frame_number] = {}

                data = json.load(open(os.path.join(frames_dir, file_name)))
                img = utils.image.img_b64_to_arr(data['imageData'])

                label_name_to_value = {"_background_": 0}
                for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                    label_name = shape["label"]
                    if label_name not in label_name_to_value:
                        label_name_to_value[label_name] = len(label_name_to_value)

                lbl, _ = utils.shapes_to_label(
                    img.shape, data["shapes"], label_name_to_value
                )

                # Visualize mask prompt
                os.makedirs(os.path.join(save_dir, "prompt_masks"), exist_ok=True)
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(img)
                axs[0].axis("off")
                axs[0].set_title("Original Image")

                axs[1].imshow((lbl * 255).astype(np.uint8), cmap="viridis")
                axs[1].axis("off")
                axs[1].set_title("Mask Prompt")
                plt.savefig(os.path.join(save_dir, "prompt_masks", f"mask_prompt_{frame_number:05d}.jpg"))
                plt.close()

                # Extract object masks
                ids = np.unique(lbl)
                for i in ids:
                    if i > 0:  # Skip background
                        self.manual_prompts[frame_number][i] = np.array(lbl == i)

        if not self.manual_prompts:
            raise ValueError("No mask prompts found in JSON files")
        
        sorted_keys = sorted(self.manual_prompts.keys())
        self.manual_prompts = {key: self.manual_prompts[key] for key in sorted_keys}

        print(f"Found mask prompts in frames: {list(self.manual_prompts.keys())}")

        # Initialize state
        self.inference_state = self.predictor.init_state(video_path=frames_dir)

        # Visualize initial masks
        os.makedirs(os.path.join(save_dir, "initial_masks"), exist_ok=True)
        for frame_index, prompt_frame in self.manual_prompts.items():
            for ann_obj_id, mask in prompt_frame.items():
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_index,
                    obj_id=ann_obj_id,
                    mask=mask,
                )

            plt.figure(figsize=(12, 8))
            plt.title(f"Frame {frame_index} with Mask Prompts")
            plt.imshow(Image.open(os.path.join(frames_dir, f"{frame_index:05d}.jpg")))
            for i, out_obj_id in enumerate(out_obj_ids):
                self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
            img_save_path = os.path.join(save_dir, "initial_masks", f"initial_mask_{frame_index:05d}.jpg")
            plt.savefig(img_save_path)
            plt.close()

        # Run propagation and collect results
        video_segments = []
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments.append({
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            })

        # Handle potential frame count mismatch
        frame_interval = len(frame_paths)
        if len(video_segments) < frame_interval:
            print(f"Padding {frame_interval - len(video_segments)} missing frames with None")
            padded_segments = [None] * (frame_interval - len(video_segments))
            padded_segments.extend(video_segments)
            video_segments = padded_segments

        # Update segment manager with results
        self.segment_manager.extend_segments(video_segments)

        # Generate output video
        output_video_path = os.path.join(save_dir, "segmented_video.mp4")
        temp_folder = os.path.join(save_dir, "masks")
        self.segment_manager.generate_masked_video_and_image(output_video_path, fps, temp_folder)

        # Save segments for later use
        pickle_save_path = os.path.join(save_dir, "segment_masks.pickle")
        shape = self.segment_manager.save_video_segments(pickle_save_path)

        # Clean up
        self.predictor.reset_state(self.inference_state)

        return video_segments, output_video_path
    
    def _run_with_automatical_prompts_from_detection(self, frames_dir, frame_paths, detection_ckpt_path="", 
                                                     prompt_type="points", save_dir="", fps=10, ann_frame_idx=0):  # the frame index we interact with
        

        # init video predictor state
        # init sam image predictor and video predictor model

        sam2_image_model = build_sam2(self.model_cfg, self.checkpoint_path)
        image_predictor = SAM2ImagePredictor(sam2_image_model)

        self.inference_state = self.predictor.init_state(video_path=frames_dir)

        # prompt grounding dino to get the box coordinates on specific frame
        img_path = frame_paths[ann_frame_idx]

        img = cv2.imread(img_path)
        detector = YOLODetector(detection_ckpt_path, conf=0.5)
        results = detector.predict(img=img, conf=0.5)
        boxes = results[0].boxes

        OBJECTS = []
        # Get boxes in xyxy format
        input_boxes = boxes.xyxy.cpu().numpy()
        for i, box in enumerate(input_boxes):
            OBJECTS.append(i + 1)  # Start from 1
        print(input_boxes)

        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(img)

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        assert prompt_type in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if prompt_type == "point":
            # sample the positive points from mask for each objects
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        # Using box prompt
        elif prompt_type == "box":
            for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        # Using mask prompt is a more straightforward way
        elif prompt_type == "mask":
            for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )
        else:
            raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")
        
        # Run propagation and collect results
        video_segments = []
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments.append({
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            })

        
        # Handle potential frame count mismatch
        frame_interval = len(frame_paths)
        if len(video_segments) < frame_interval:
            print(f"Padding {frame_interval - len(video_segments)} missing frames with None")
            padded_segments = [None] * (frame_interval - len(video_segments))
            padded_segments.extend(video_segments)
            video_segments = padded_segments

        # Update segment manager with results
        self.segment_manager.extend_segments(video_segments)

        # Generate output video
        output_video_path = os.path.join(save_dir, "segmented_video.mp4")
        temp_folder = os.path.join(save_dir, "masks")
        # self.segment_manager.generate_masked_video_and_image(output_video_path, fps, temp_folder)
        self.segment_manager.generate_masked_video_supervision(output_video_path=output_video_path, fps=fps, mask_save_folder=temp_folder)

        # Save segments for later use
        pickle_save_path = os.path.join(save_dir, "segment_masks.pickle")
        shape = self.segment_manager.save_video_segments(pickle_save_path)

        # Clean up
        self.predictor.reset_state(self.inference_state)

        return video_segments, output_video_path
    


    def load_segments(self, pickle_path, shape, frame_paths=None):
        """Load previously saved video segments.

        Args:
            pickle_path (str): Path to the compressed pickle file.
            shape (tuple): Shape of the mask (height, width).
            frame_paths (list, optional): List of frame paths. Defaults to None.

        Returns:
            list: List of video segments.
        """
        segments = self.segment_manager.load_video_segments(pickle_path, shape)

        if frame_paths:
            self.segment_manager.extend_frame_paths(frame_paths)

        return segments


if __name__ == "__main__":

    # Enable optimizations if CUDA is available
    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Configuration
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    checkpoint_path = "./checkpoints/sam2_base_five_mouse_finetuned.pt"

    # Initialize inference
    inference = VideoSegmentationInference(model_cfg, checkpoint_path)

    # Option 1: Run from video file with point prompts
    
    # video_path = "path/to/video.mp4"
    # frames_dir = "/mnt/nas01/LAR/pico/Analysis/tracking/segmentation/test_sam2_mice/habitat_1/raw"

    # # inference.extract_frames_before_seg(video_path=video_path)

    # segments, video_output = inference.run(
    #     video_path=None,
    #     frames_dir= frames_dir,
    #     prompt_source = "manual",
    #     detection_ckpt_path="",
    #     prompt_type="points",
    #     save_dir="/mnt/nas01/LAR/pico/Analysis/tracking/segmentation/test_sam2_mice/habitat_1/seg"
    # )

    # inference.reset()

    # # Option 2: Run from existing frames with mask prompts
    # frames_dir = "path/to/frames"
    # segments, video_output = inference.run(
    #     frames_dir=frames_dir,
    #     prompt_type="mask",
    #     save_dir="output/mask_results",
    #     fps=15
    # )

    # print(f"Segmentation complete. Output video saved at: {video_output}")


    # Option 3: Run prompt-free YOLO detection

    video_path = "notebooks_SAM2-MICE/videos/open_field_five_mouse.mp4"
    detection_ckpt_path = "checkpoints_detection/yolo_v11_l.pt"
    frames_dir = None

    inference.extract_frames_before_seg(video_path=video_path)

    segments, video_output = inference.run(
        video_path=video_path,
        frames_dir= frames_dir,
        prompt_source = "detection",
        detection_ckpt_path=detection_ckpt_path,
        prompt_type="point",
        save_dir=None
    )

    inference.reset()





