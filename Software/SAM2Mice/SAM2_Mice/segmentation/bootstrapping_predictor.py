import os
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import datetime
from concurrent.futures import ThreadPoolExecutor
from labelme import utils
from ultralytics import YOLO

from SAM2_Mice.segmentation.base_predictor import VideoSegmentationInference, VideoFrameExtractor, VideoSegmentManager
from SAM2_Mice.detection import YOLODetector, sample_points_from_masks
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class BootstrappingVideoSegmentationInference(VideoSegmentationInference):
    """
    A class for performing video segmentation inference with bootstrapping capabilities,
    allowing for the transmission of masks between different batches of frames.
    """

    def __init__(self, model_cfg, checkpoint_path, vos_optimized=False):
        """Initialize the bootstrapping video segmentation inference.

        Args:
            model_cfg (str): Path to the model configuration file.
            checkpoint_path (str): Path to the model checkpoint.
            vos_optimized (bool, optional): Whether to use VOS optimization. Defaults to False.
        """
        super().__init__(model_cfg, checkpoint_path, vos_optimized)
        self.boots_segment = {}  # Store bootstrapped segments for cross-batch transmission

    def extract_bootstrapping_frames(self, video_path, batch_size=1000, batch_save_dir=""):
        """
        Extract frames from a video file into batches for bootstrapping.

        Args:
            video_path (str): Path to the video file.
            batch_size (int, optional): Number of frames per batch. Defaults to 1000.
            output_dir (str, optional): Directory to save extracted frames. Defaults to "".

        Returns:
            list: List of directories containing extracted frames.
        """
        if not batch_save_dir:
            # Get the base directories
            base_path = os.path.dirname(video_path)
            file_name = os.path.basename(video_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            batch_save_dir = os.path.join(base_path, file_name_without_ext)

        # return VideoFrameExtractor.extract_bootstrapping_frames(video_path, batch_size, output_dir)
        return VideoFrameExtractor.extract_bootstrapping_frames_multithreaded(video_path, batch_size, batch_save_dir)

    def run_bootstrapping(self, video_path,
                          frame_interval=1000,
                          extract_frame=False,
                          
                          prompt_source="detection", 
                          
                          detection_frame_idx=0,
                          detection_ckpt_path="", 
                          prompt_type="points", 

                          batch_limit=None,

                          save_dir=None,
                          fps=10):
        """
        Run video segmentation inference with bootstrapping across batches.

        Args:
            video_path (str): Path to the video file.
            frame_interval (int, optional): Number of frames per batch. Defaults to 1000.
            extract_frame (bool, optional): Whether to extract frames. Defaults to False.
            use_point_prompts (bool, optional): Whether to use point prompts. Defaults to True.
            batch_limit (int, optional): Limit the number of batches to process. Defaults to None.
            detection_ckpt_path (str, optional): Path to YOLO detection checkpoint. Defaults to "".
            prompt_type (str, optional): Type of prompts to use. Defaults to "point".
            save_dir (str, optional): Directory to save results. Defaults to None.
            fps (int, optional): Frames per second for output video. Defaults to 30.

        Returns:
            tuple: (video_segments, output_video_path)
        """
        # Enable optimizations if CUDA is available
        if torch.cuda.is_available():
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Get the base directories
        base_path = os.path.dirname(video_path)
        file_name = os.path.basename(video_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        batch_save_dir = os.path.join(base_path, file_name_without_ext)

        # Create output directory with timestamp
        if save_dir is None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            save_dir = os.path.join(batch_save_dir, f"results_{current_time}")

        os.makedirs(save_dir, exist_ok=True)
        mask_save_dir = os.path.join(save_dir, "saved_masks")
        pickle_save_path = os.path.join(save_dir, "segment_masks.pickle")
        output_video_path = os.path.join(save_dir, "segmented_video.mp4")
        print(f"Output directory: {save_dir}")

        # Extract frames into batches or get paths to existing frame batches
        if extract_frame:
            batched_folders = self.extract_bootstrapping_frames(video_path, frame_interval, batch_save_dir)
        else:
            batched_folders = [os.path.join(batch_save_dir, name) for name in os.listdir(batch_save_dir)
                               if os.path.isdir(os.path.join(batch_save_dir, name)) and "batch_" in name]
            # Sort the batched folders
            batched_folders.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

        # Apply batch limit if specified
        if batch_limit is not None:
            batched_folders = batched_folders[:batch_limit]
            print(f"Limiting processing to {batch_limit} batches.")

        print("Bootstrapping SAM2 inference beginning, processing folders:")
        for batched_folder in batched_folders:
            print(batched_folder)
        
        num_folders = len(batched_folders)

        no_mice_flag = True  # Flag to track if mice have been detected
        detect_batch_to_skip = 0  if prompt_source=="manual" else  detection_frame_idx // frame_interval - 1         # Index to skip for YOLO

        # Process each batch with bootstrapping
        for i, batch_folder in enumerate(batched_folders):
            print(f"\nProcessing batch {i + 1}/{len(batched_folders)}: {os.path.basename(batch_folder)}")

            frame_names = sorted(
                            [p for p in os.listdir(batch_folder) if p.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))],
                            key=lambda p: int(os.path.splitext(p)[0]))

            if i == 0 or no_mice_flag:

                if prompt_source=="detection":

                    if not (detection_ckpt_path and os.path.exists(detection_ckpt_path)):
                        raise FileNotFoundError(f"Please ensure detection ckpt path exist: {detection_ckpt_path}")
                    
                    if i<= detect_batch_to_skip:
                        
                        print(f"[{i+1}/{num_folders}] skipping batch {i+1} as assigned detection_frame begin index is {detection_frame_idx}.")
                        batched_segments = [None] * frame_interval
                        batch_frame_paths = [os.path.join(batch_folder, name) for name in frame_names[:-1]]
                        boots_segment = None
                    
                    else:
                        
                        print(f"Running automatical detection using yolov11 in frame {detection_frame_idx}.")
                        detection_frame_idx_for_curremt_batch = detection_frame_idx-(detect_batch_to_skip+1)*frame_interval
                        batched_segments, boots_segment, batch_frame_paths = self._run_with_automatical_prompts_from_detection(frames_dir=batch_folder, 
                                                                                detection_ckpt_path=detection_ckpt_path, prompt_type=prompt_type, 
                                                                                save_dir=save_dir, 
                                                                                ann_frame_idx=detection_frame_idx_for_curremt_batch,
                                                                                boots_id=i + 1, boots_predictor=True)
                        no_mice_flag = False

                elif prompt_source=="manual":

                    if prompt_type.lower() == "point":
                        prompts = self._load_point_prompts_from_json(batch_folder)
                    elif prompt_type.lower() == "mask":
                        prompts = self._load_mask_prompts_from_json(batch_folder, save_dir)
                    else:
                        raise ValueError(f"Unsupported prompt type: {prompt_type}. Use 'point' or 'mask'")
                    
                    if prompts:
                        batched_segments, boots_segment, batch_frame_paths = self.bootstrapping_video_predictor(
                            video_dir=batch_folder,
                            prompts_exist=True,
                            prompt_type=prompt_type,
                            prompts=prompts,
                            base_dir=save_dir,
                            boots_id=i + 1,
                            boots_predictor=True)
                        if len(batched_segments) < frame_interval:
                            print(
                                f"Mice appears in frame {frame_interval - len(batched_segments) + 1}, padding None as segment mask.")
                            batched_segments = [None] * (
                                    frame_interval - len(batched_segments)) + batched_segments

                        no_mice_flag = False

                    else:
                        print(f"There is no mice found in batch{i + 1}, processing the next one!")

                        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
                        batch_frame_paths = [os.path.join(batch_folder, name) for name in frame_names[:-1]]

                        empty_segments = [None for i in range(frame_interval)]
                        batched_segments = empty_segments
                        boots_segment = None
                
                else:
                    raise NotImplementedError("bootstrapping SAM2-MICE only support prompt from automatical detection or external manual prompts")
                
                self.segment_manager.extend_segments(batched_segments)
                self.segment_manager.extend_frame_paths(batch_frame_paths)
                self.boots_segment = boots_segment 


            # Subsequent batches
            else:

                if prompt_type.lower() == "point":
                        prompts = self._load_point_prompts_from_json(batch_folder)
                elif prompt_type.lower() == "mask":
                    prompts = self._load_mask_prompts_from_json(batch_folder, save_dir)
                else:
                    raise ValueError(f"Unsupported prompt type: {prompt_type}. Use 'point' or 'mask'")

                if prompts:  # Manual prompts exist
                    print(f"Using manual prompts for batch {i + 1}")
                    prompts_exist = True
                    
                # Use bootstrapped mask from previous batch
                elif self.boots_segment:
                    print(f"Using bootstrapping mask from previous batch for batch {i + 1}")
                    frame_0_prompts = {0: {}}
                    for obj_id, mask in self.boots_segment.items():
                        # Ensure mask has correct shape
                        if isinstance(mask, np.ndarray):
                            if mask.ndim == 3:  # If 1×H×W shape
                                mask = np.squeeze(mask, axis=0)
                            mask = mask.astype(bool)
                            frame_0_prompts[0][obj_id] = mask
                    prompts_exist = False
                    prompts = frame_0_prompts

                    # for frame_index, prompt in frame_0_prompts.items():
                    #     print(f"frame index: {frame_index}")
                    #     for ann_obj_id, mask in prompt.items():
                    #         print(f"ann_obj_id: {ann_obj_id}")
                    #         print(f"masks: {mask}")
                    

                if i == len(batched_folders) - 1:
                    # for the last predictor, set boots_predictor=False

                    batched_segments, batch_frame_paths = self.bootstrapping_video_predictor(video_dir=batch_folder,
                                                                                                prompts=prompts,
                                                                                                prompts_exist=prompts_exist,
                                                                                                prompt_type=prompt_type,
                                                                                                base_dir=save_dir,
                                                                                                boots_id=i + 1,
                                                                                                boots_predictor=False)
                    boots_segment = None                                                                            
                else:
                    # for not the last predictor, set boots_predictor=True
                    batched_segments, boots_segment, batch_frame_paths = self.bootstrapping_video_predictor(
                        video_dir=batch_folder,
                        prompts=prompts,
                        prompts_exist=prompts_exist,
                        prompt_type=prompt_type,
                        base_dir=save_dir,
                        boots_id=i + 1,
                        boots_predictor=True)
                    
                self.segment_manager.extend_segments(batched_segments)
                self.segment_manager.extend_frame_paths(batch_frame_paths)
                self.boots_segment = boots_segment

            # Clear CUDA cache after each batch
            torch.cuda.empty_cache()

        print(f"Total processed segments: {self.segment_manager.get_len()}")


        # Generate output video
        os.makedirs(mask_save_dir, exist_ok=True)
        self.segment_manager.generate_masked_video_and_image(output_video_path, fps, mask_save_dir)

        # Save segments for later use
        pickle_save_path = os.path.join(save_dir, "segment_masks.pickle")
        shape = self.segment_manager.save_video_segments(pickle_save_path)


    def bootstrapping_video_predictor(self, video_dir="./dataset/frames2",
                                  base_dir="",
                                  boots_id=0,
                                  prompts={},
                                  prompts_exist=True,
                                  prompt_type="point",
                                  boots_predictor=True):
        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        batched_frame_paths = [os.path.join(video_dir, frame_name) for frame_name in frame_names]

        inference_state = self.predictor.init_state(video_path=video_dir)

        if prompts_exist:
            # using the prompt in point format
            if prompt_type.lower() =="point":
                # without running, lets show the point
                for frame_index, prompt_frame in prompts.items():
                    for ann_obj_id, (points, labels) in prompt_frame.items():
                        ann_frame_idx = frame_index
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            points=points,
                            labels=labels,
                        )

                    plt.figure(figsize=(12, 8))
                    plt.title(f"frame {frame_index}")
                    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_index])))
                    for i, out_obj_id in enumerate(out_obj_ids):
                        self.show_points(*prompt_frame[out_obj_id], plt.gca())
                        self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
                    save_path = os.path.join(base_dir, f"predictor_{boots_id}_frame{frame_index}.jpg")
                    plt.savefig(save_path)
                    plt.close()

            elif prompt_type.lower() == "mask": # using the prompt in mask format

                for frame_index, prompt_frame in prompts.items():
                    ann_frame_idx = frame_index  # the frame index we interact with
                    for ann_obj_id, mask in prompt_frame.items():
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            mask=mask,
                        )

                    plt.figure(figsize=(12, 8))
                    plt.title(f"frame {frame_index}")
                    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_index])))
                    for i, out_obj_id in enumerate(out_obj_ids):
                        self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
                    tmp = os.path.join(base_dir, f"mask_frame{frame_index}.jpg")
                    plt.savefig(tmp)
                    plt.close()


        # use bootstrapping mask
        else:

            for frame_index, prompt_frame in prompts.items():
                ann_frame_idx = frame_index  # the frame index we interact with
                for ann_obj_id, mask in prompt_frame.items():
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        mask=mask,
                    )

                plt.figure(figsize=(12, 8))
                plt.title(f"frame {frame_index}")
                plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_index])))
                for i, out_obj_id in enumerate(out_obj_ids):
                    self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
                save_path = os.path.join(base_dir, f"predictor_{boots_id}_mask_frame{frame_index}.jpg")
                plt.savefig(save_path)
                plt.close()

        # run propagation throughout the video and collect the results in a list
        batched_video_segments = []  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            batched_video_segments.append({
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)

            })

        # very important, empty unused cuda cathe, prevent cuda out of memory
        self.predictor.reset_state(inference_state)
        del inference_state
        torch.cuda.empty_cache()

        # if bootstrapping, return 1000 segments and the last 1 segments for bootstrapping
        if boots_predictor:
            return batched_video_segments[:-1], batched_video_segments[-1], batched_frame_paths[:-1]
        else:
            return batched_video_segments, batched_frame_paths



    def _load_point_prompts_from_json(self, json_path):
        """Load point prompts from JSON files.

        Args:
            json_path (str): Path to the directory containing JSON files.

        Returns:
            dict: Dictionary of point prompts.
        """
        prompts = {}
        for file_name in os.listdir(json_path):
            if file_name.endswith(".json"):
                # Extract frame number (labelme annotations format like 00000.json)
                frame_number = int(file_name.split(".")[0])
                with open(os.path.join(json_path, file_name), 'r') as f:
                    jsonx = json.load(f)
                    for item in jsonx['shapes']:
                        if 'group_id' in item:
                            ann_obj_id = item["group_id"]  # Corresponds to ann_obj_id
                        else:
                            # If no group_id, use a generated one based on label
                            ann_obj_id = hash(item["label"]) % 100  # Simple hash to get a consistent ID

                        points = np.array(item["points"], dtype=np.float32)
                        labels = np.ones(len(points), np.int32)

                        if frame_number not in list(prompts.keys()):
                            prompts[frame_number] = {}
                        prompts[frame_number][ann_obj_id] = (points, labels)

        if prompts:
            print(f"Found point prompts in frames: {list(prompts.keys())}")

        return prompts

    def _load_mask_prompts_from_json(self, json_path, save_dir):
        """Load mask prompts from JSON files.

        Args:
            json_path (str): Path to the directory containing JSON files.
            save_dir (str): Directory to save visualizations.

        Returns:
            dict: Dictionary of mask prompts.
        """
        prompts = {}
        for file_name in os.listdir(json_path):
            if file_name.endswith(".json"):
                frame_number = int(file_name.split(".")[0])

                if frame_number not in list(prompts.keys()):
                    prompts[frame_number] = {}

                data = json.load(open(os.path.join(json_path, file_name)))
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
                        prompts[frame_number][i] = np.array(lbl == i)

        if prompts:
            print(f"Found mask prompts in frames: {list(prompts.keys())}")

        return prompts


    def _run_with_automatical_prompts_from_detection(self, frames_dir, detection_ckpt_path="",
                                                     prompt_type="point", save_dir="", ann_frame_idx=0, boots_id=0, boots_predictor=True):
        """
        Modified version of the parent method to return segments without adding to manager.
        This is needed for batch processing.
        """
        # Initialize SAM image predictor and video predictor
        sam2_image_model = build_sam2(self.model_cfg, self.checkpoint_path)
        image_predictor = SAM2ImagePredictor(sam2_image_model)

        inference_state = self.predictor.init_state(video_path=frames_dir)

        frame_names = [
            p for p in os.listdir(frames_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        frame_paths = [os.path.join(frames_dir, frame_name) for frame_name in frame_names]

        # Get the detection frame
        img_path = frame_paths[ann_frame_idx]
        img = cv2.imread(img_path)

        # Run YOLO detection
        detector = YOLODetector(detection_ckpt_path, conf=0.5)
        results = detector.predict(img=img, conf=0.5)
        boxes = results[0].boxes

        OBJECTS = []
        # Get boxes in xyxy format
        input_boxes = boxes.xyxy.cpu().numpy()
        for i, box in enumerate(input_boxes):
            OBJECTS.append(i + 1)  # Start from 1
        print(f"Detected {len(input_boxes)} objects")

        if len(input_boxes) == 0:
            print("No objects detected in frame")
            return [], []

        # Get masks from SAM2 image predictor
        image_predictor.set_image(img)
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Convert mask shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Process according to prompt type
        if prompt_type == "point":
            # Sample points from masks
            all_sample_points = sample_points_from_masks(masks=masks, num_points=5)

            for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )

                plt.figure(figsize=(12, 8))
                plt.title(f"frame {ann_frame_idx}")
                plt.imshow(Image.open(frame_paths[ann_frame_idx]))
                self.show_points(points, labels, plt.gca())
                for i, out_obj_id in enumerate(out_obj_ids):
                    self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
                    self.show_box(input_boxes[out_obj_id-1], plt.gca())
                save_path = os.path.join(save_dir, f"predictor_{boots_id}_yolo_{ann_frame_idx}.jpg")
                plt.savefig(save_path)
                plt.close()

                
        elif prompt_type == "box":
            for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )

                plt.figure(figsize=(12, 8))
                plt.title(f"frame {ann_frame_idx}")
                plt.imshow(Image.open(frame_paths[ann_frame_idx]))
                for i, out_obj_id in enumerate(out_obj_ids):
                    self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
                    self.show_box(input_boxes[out_obj_id], plt.gca())
                save_path = os.path.join(save_dir, f"predictor_{boots_id}_yolo_{ann_frame_idx}.jpg")
                plt.savefig(save_path)
                plt.close()

        elif prompt_type == "mask":
            for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )

                plt.figure(figsize=(12, 8))
                plt.title(f"frame {ann_frame_idx}")
                plt.imshow(Image.open(frame_paths[ann_frame_idx]))
                for i, out_obj_id in enumerate(out_obj_ids):
                    self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
                    self.show_box(input_boxes[out_obj_id], plt.gca())
                save_path = os.path.join(save_dir, f"predictor_{boots_id}_yolo_{ann_frame_idx}.jpg")
                plt.savefig(save_path)
                plt.close()
        else:
            raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

        # Run propagation and collect results
        batched_video_segments = []
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            batched_video_segments.append({
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            })

        # Handle potential frame count mismatch
        frame_interval = len(frame_paths)
        if len(batched_video_segments) < frame_interval:
            print(f"Padding {frame_interval - len(batched_video_segments)} missing frames with None")
            batched_video_segments = [None] * (frame_interval - len(batched_video_segments)) + batched_video_segments

        # Clean up
        self.predictor.reset_state(inference_state)

        # if bootstrapping, return 1000 segments and the last 1 segments for bootstrapping
        if boots_predictor:
            return batched_video_segments[:-1], batched_video_segments[-1], frame_paths[:-1]
        else:
            return batched_video_segments, frame_paths


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
    detection_ckpt_path = "./checkpoints_detection/yolo_v11_l.pt"

    # # Initialize bootstrapping inference
    inference = BootstrappingVideoSegmentationInference(model_cfg, checkpoint_path=checkpoint_path)

    # VIDEO_PATH = "/mnt/nas00/LAR/dataset/sam2_train_data/PICO_experiment_video/1006_five_mouse.mp4"
    VIDEO_PATH = "/mnt/nas00/lk/pico/Experiments/multimice/20250103multimice5/behavior/behavior.mp4"

    # inference.extract_bootstrapping_frames(video_path=VIDEO_PATH, batch_size=1000, batch_save_dir="")


    # inference.run_bootstrapping(video_path=VIDEO_PATH,
    #                       frame_interval=1000,
    #                       extract_frame=False,
                          
    #                       prompt_source="manual", 
                          
    #                       detection_frame_idx=0,
    #                       detection_ckpt_path="", 
    #                       prompt_type="point", 

    #                       batch_limit=5,

    #                       save_dir=None,
    #                       fps=10)
    

    inference.run_bootstrapping(video_path=VIDEO_PATH,
                          frame_interval=1000,
                          extract_frame=False,
                          
                          prompt_source="detection", 
                          
                          detection_frame_idx=2275,
                          detection_ckpt_path=detection_ckpt_path, 
                          prompt_type="point", 

                          batch_limit=4,

                          save_dir=None,
                          fps=10)

