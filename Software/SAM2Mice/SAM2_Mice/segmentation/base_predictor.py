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
        """Draw an xyxy prompt box on a matplotlib axis."""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def extract_frames_before_seg(self, video_path=None, frames_dir=None,
                                   annotate=False, port=7860, share=False):
        """Extract frames and optionally launch the interactive annotator.

        Args:
            video_path: Path to the source video.
            frames_dir: Directory to save / read frames from.
            annotate: If True, launch the Gradio annotation UI after extraction.
                      Access via http://localhost:<port> (use SSH tunnel on remote servers).
            port: Port for the Gradio server (default 7860).
            share: If True, create a public Gradio share link.
        """
        if video_path and not frames_dir:
            frames_dir = os.path.join(os.path.dirname(video_path),
                                      f"{os.path.splitext(os.path.basename(video_path))[0]}")
            VideoFrameExtractor.extract_frames(video_path, frames_dir)
        elif video_path and frames_dir:
            VideoFrameExtractor.extract_frames(video_path, frames_dir)

        if annotate and frames_dir:
            from SAM2_Mice.utils.annotator import launch_annotator
            launch_annotator(frames_dir, port=port, share=share)

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
        if prompt_source not in ["detection", "manual"]:
            raise ValueError(f"prompt_source must be 'detection' or 'manual', got '{prompt_source}'")

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

        prompt_type = prompt_type.lower()
        if prompt_type == "points":
            prompt_type = "point"

        if prompt_source=="detection":
            
            return self._run_with_automatical_prompts_from_detection(frames_dir=frames_dir, frame_paths=frame_paths, 
                                                                     detection_ckpt_path=detection_ckpt_path, prompt_type=prompt_type, 
                                                                     save_dir=save_dir, fps=fps)

        elif prompt_source=="manual":

            # Get prompts based on prompt type
            if prompt_type == "point":
                return self._run_with_point_prompts(frames_dir, frame_paths, save_dir, fps)
            elif prompt_type == "box":
                return self._run_with_box_prompts(frames_dir, frame_paths, save_dir, fps)
            elif prompt_type == "mask":
                return self._run_with_mask_prompts(frames_dir, frame_paths, save_dir, fps)
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}. Use 'point', 'box', or 'mask'")

    def _run_with_point_prompts(self, frames_dir, frame_paths, save_dir, fps):
        """Run inference using point prompts.

        Reads standard labelme JSON and uses polygon vertices as positive SAM2
        prompt points. The annotator stores Point-mode clicks as
        ``shape_type == "polygon"`` with ``flags.sam2mice_prompt == "point"``.
        Regular polygon vertices remain supported for backward compatibility.
        """
        for file_name in os.listdir(frames_dir):
            if not file_name.endswith(".json"):
                continue
            stem = os.path.splitext(file_name)[0]
            if not stem.isdigit():
                continue
            frame_number = int(stem)
            with open(os.path.join(frames_dir, file_name), 'r') as f:
                jsonx = json.load(f)

            # Group point prompts by obj_id. Old polygon-vertex prompts remain supported.
            pts_by_obj = {}
            for item in jsonx.get('shapes', []):
                shape_type = item.get("shape_type")
                ann_obj_id = item.get("group_id")
                pts = item.get("points", [])
                if ann_obj_id is None:
                    continue
                if shape_type == "point" and len(pts) >= 1:
                    pts_by_obj.setdefault(int(ann_obj_id), []).append(pts[0])
                elif shape_type == "polygon" and len(pts) >= 1:
                    pts_by_obj.setdefault(int(ann_obj_id), []).extend(pts)

            if not pts_by_obj:
                continue
            self.manual_prompts.setdefault(frame_number, {})
            for ann_obj_id, pts in pts_by_obj.items():
                points = np.array(pts, dtype=np.float32)
                labels = np.ones(len(points), np.int32)
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

    def _run_with_box_prompts(self, frames_dir, frame_paths, save_dir, fps):
        """Run inference using box prompts from the labelme JSON.

        Consumes shapes whose ``shape_type == "rectangle"`` (two-corner points).
        If multiple boxes share an obj_id within a frame, only the last one
        wins (SAM2 video predictor accepts one box per obj_id per frame).
        """
        for file_name in os.listdir(frames_dir):
            if not file_name.endswith(".json"):
                continue
            stem = os.path.splitext(file_name)[0]
            if not stem.isdigit():
                continue
            frame_number = int(stem)
            with open(os.path.join(frames_dir, file_name), "r") as f:
                jsonx = json.load(f)

            box_by_obj = {}
            for item in jsonx.get("shapes", []):
                if item.get("shape_type") != "rectangle":
                    continue
                ann_obj_id = item.get("group_id")
                pts = item.get("points", [])
                if ann_obj_id is None or len(pts) < 2:
                    continue
                (x1, y1), (x2, y2) = pts[0], pts[1]
                x1, x2 = sorted([float(x1), float(x2)])
                y1, y2 = sorted([float(y1), float(y2)])
                box_by_obj[int(ann_obj_id)] = np.array([x1, y1, x2, y2], dtype=np.float32)

            if not box_by_obj:
                continue
            self.manual_prompts.setdefault(frame_number, {})
            for ann_obj_id, box in box_by_obj.items():
                self.manual_prompts[frame_number][ann_obj_id] = box

        if not self.manual_prompts:
            raise ValueError("No box prompts found in JSON files")

        sorted_keys = sorted(self.manual_prompts.keys())
        self.manual_prompts = {k: self.manual_prompts[k] for k in sorted_keys}
        print(f"Found box prompts in frames: {list(self.manual_prompts.keys())}")

        self.inference_state = self.predictor.init_state(video_path=frames_dir)

        os.makedirs(os.path.join(save_dir, "prompts"), exist_ok=True)
        for frame_index, prompt_frame in self.manual_prompts.items():
            out_obj_ids, out_mask_logits = None, None
            for ann_obj_id, box in prompt_frame.items():
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_index,
                    obj_id=ann_obj_id,
                    box=box,
                )

            plt.figure(figsize=(12, 8))
            plt.title(f"Frame {frame_index} with Box Prompts")
            plt.imshow(Image.open(os.path.join(frames_dir, f"{frame_index:05d}.jpg")))
            ax = plt.gca()
            for ann_obj_id, box in prompt_frame.items():
                self.show_box(box, ax)
            if out_obj_ids is not None:
                for i, out_obj_id in enumerate(out_obj_ids):
                    self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), ax, obj_id=out_obj_id)
            plt.savefig(os.path.join(save_dir, "prompts", f"frame{frame_index:05d}.jpg"))
            plt.close()

        video_segments = []
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments.append({
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            })

        frame_interval = len(frame_paths)
        if len(video_segments) < frame_interval:
            print(f"Padding {frame_interval - len(video_segments)} missing frames with None")
            padded_segments = [None] * (frame_interval - len(video_segments))
            padded_segments.extend(video_segments)
            video_segments = padded_segments

        self.segment_manager.extend_segments(video_segments)

        output_video_path = os.path.join(save_dir, "segmented_video.mp4")
        temp_folder = os.path.join(save_dir, "frames")
        self.segment_manager.generate_masked_video_and_image(output_video_path, fps, temp_folder)

        pickle_save_path = os.path.join(save_dir, "segment_masks.pickle")
        shape = self.segment_manager.save_video_segments(pickle_save_path)

        self.predictor.reset_state(self.inference_state)

        return video_segments, output_video_path

    def _run_with_mask_prompts(self, frames_dir, frame_paths, save_dir, fps):
        """Run inference using mask prompts from labelme JSON.

        Rasterizes ``polygon`` and ``rectangle`` shapes via
        ``labelme.utils.shapes_to_label`` and feeds them to SAM2 as mask prompts.
        ``group_id`` is used as the SAM2 object id.
        """
        for file_name in os.listdir(frames_dir):
            if not file_name.endswith(".json"):
                continue
            stem = os.path.splitext(file_name)[0]
            if not stem.isdigit():
                continue
            frame_number = int(stem)

            if frame_number not in list(self.manual_prompts.keys()):
                self.manual_prompts[frame_number] = {}

            with open(os.path.join(frames_dir, file_name), "r") as _f:
                data = json.load(_f)
            img = utils.image.img_b64_to_arr(data['imageData'])

            # Build a label-name -> integer map keyed by group_id so the
            # rasterized label image uses the mouse id directly.
            label_name_to_value = {"_background_": 0}
            shapes = []
            for shape in data.get("shapes", []):
                if shape.get("shape_type") not in ("polygon", "rectangle"):
                    continue
                if (shape.get("flags") or {}).get("sam2mice_prompt") == "point":
                    continue
                oid = shape.get("group_id")
                if oid is None:
                    continue
                key = f"mouse{int(oid)}"
                label_name_to_value[key] = int(oid)
                s = dict(shape)
                s["label"] = key
                shapes.append(s)

            if not shapes:
                continue

            lbl, _ = utils.shapes_to_label(img.shape, shapes, label_name_to_value)

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

            ids = np.unique(lbl)
            for i in ids:
                if i > 0:
                    self.manual_prompts[frame_number][int(i)] = np.array(lbl == i)

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
        """Run video segmentation with prompts generated automatically by YOLO."""
        prompt_type = prompt_type.lower()
        if prompt_type == "points":
            prompt_type = "point"
        if prompt_type not in ["point", "box", "mask"]:
            raise ValueError(f"prompt_type must be 'point', 'box', or 'mask', got '{prompt_type}'")

        # Detect boxes on the annotation frame and use them as SAM2 prompts.
        img_path = frame_paths[ann_frame_idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read annotation frame: {img_path}")

        detector = YOLODetector(detection_ckpt_path, conf=0.5)
        results = detector.predict(img=img, conf=0.5)
        boxes = results[0].boxes

        input_boxes = boxes.xyxy.cpu().numpy()
        if len(input_boxes) == 0:
            raise ValueError(f"No YOLO detections found on frame {ann_frame_idx}: {img_path}")
        object_ids = np.arange(1, len(input_boxes) + 1, dtype=np.int32)
        print(f"Detected {len(input_boxes)} objects on frame {ann_frame_idx}")
        print(input_boxes)

        self.inference_state = self.predictor.init_state(video_path=frames_dir)

        masks = None
        point_prompts = {}
        if prompt_type in ["point", "mask"]:
            sam2_image_model = build_sam2(self.model_cfg, self.checkpoint_path)
            image_predictor = SAM2ImagePredictor(sam2_image_model)
            image_predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if prompt_type == "point":
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            for object_id, points in zip(object_ids, all_sample_points):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                point_prompts[int(object_id)] = (points, labels)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=int(object_id),
                    points=points,
                    labels=labels,
                )
        elif prompt_type == "box":
            for object_id, box in zip(object_ids, input_boxes):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=int(object_id),
                    box=box,
                )
        elif prompt_type == "mask":
            for object_id, mask in zip(object_ids, masks):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=int(object_id),
                    mask=mask,
                )

        prompt_vis_dir = os.path.join(save_dir, "prompts")
        os.makedirs(prompt_vis_dir, exist_ok=True)
        prompt_vis_path = os.path.join(prompt_vis_dir, f"frame{ann_frame_idx:05d}_{prompt_type}_prompt.jpg")
        plt.figure(figsize=(12, 8))
        plt.title(f"Detection prompts on frame {ann_frame_idx} ({prompt_type})")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax = plt.gca()
        for box in input_boxes:
            self.show_box(box, ax)
        if prompt_type == "point":
            for object_id, (points, labels) in point_prompts.items():
                self.show_points(points, labels, ax)
        for i, out_obj_id in enumerate(out_obj_ids):
            self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), ax, obj_id=out_obj_id)
        plt.axis("off")
        plt.savefig(prompt_vis_path, bbox_inches="tight", pad_inches=0)
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
