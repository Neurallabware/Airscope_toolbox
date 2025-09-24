#!/usr/bin/env python3
"""
Unified entrypoint to run SAM2-MICE workflows:
- base: single-run video segmentation using SAM2 video predictor with manual or YOLO prompts
- boots: bootstrapping segmentation across frame batches
- track: auto-tracking pipeline producing masks/json per frame and a rendered video

Examples:
  Base (YOLO prompts, points):
    python build_sam2_mice.py base \
      --video-path notebooks_SAM2-MICE/videos/open_field_five_mouse.mp4 \
      --model-cfg sam2/sam2_hiera_b+.yaml \
      --checkpoint checkpoints/sam2_base_five_mouse_finetuned.pt \
      --detection-ckpt-path checkpoints_detection/yolo_v11_l.pt \
      --prompt-source detection --prompt-type point

  Bootstrapping (YOLO prompts starting from a frame index):
    python build_sam2_mice.py boots \
      --video-path /path/to/long_video.mp4 \
      --frame-interval 1000 \
      --detection-frame-idx 2500 \
      --model-cfg sam2/sam2_hiera_b+.yaml \
      --checkpoint checkpoints/sam2_base_five_mouse_finetuned.pt \
      --detection-ckpt-path checkpoints_detection/yolo_v11_l.pt

  Auto-tracking (mask prompts):
    python build_sam2_mice.py track \
      --video-path /path/to/video.mp4 \
      --frames-dir /path/to/frames \
      --output-dir /path/to/output \
      --model-cfg sam2/sam2_hiera_b+.yaml \
      --checkpoint checkpoints/sam2_base_five_mouse_finetuned.pt \
      --detection-ckpt-path checkpoints_detection/yolo_v11_l.pt \
      --frame-step 30 --frame-rate 30 --iou-threshold 0.3
"""

import argparse
import os
import sys


from SAM2_Mice.segmentation import VideoSegmentationInference, BootstrappingVideoSegmentationInference, auto_tracking_with_sam2

def run_base(args: argparse.Namespace) -> None:
    inference = VideoSegmentationInference(
        model_cfg=args.model_cfg,
        checkpoint_path=args.checkpoint,
        vos_optimized=args.vos_optimized,
    )

    # Extract frames first if requested
    if args.extract_frames and args.video_path and not args.frames_dir:
        inference.extract_frames_before_seg(video_path=args.video_path)

    video_segments, output_video = inference.run(
        video_path=args.video_path,
        frames_dir=args.frames_dir,
        prompt_source=args.prompt_source,
        detection_ckpt_path=args.detection_ckpt_path,
        prompt_type=args.prompt_type,
        save_dir=args.save_dir,
        fps=args.fps,
    )

    print(f"Base segmentation done. Output video: {output_video}")
    inference.reset()


def run_boots(args: argparse.Namespace) -> None:
    boots = BootstrappingVideoSegmentationInference(
        model_cfg=args.model_cfg,
        checkpoint_path=args.checkpoint,
        vos_optimized=args.vos_optimized,
    )

    boots.run_bootstrapping(
        video_path=args.video_path,
        frame_interval=args.frame_interval,
        extract_frame=args.extract_frames,
        prompt_source=args.prompt_source,
        detection_frame_idx=args.detection_frame_idx,
        detection_ckpt_path=args.detection_ckpt_path,
        prompt_type=args.prompt_type,
        batch_limit=args.batch_limit,
        save_dir=args.save_dir,
        fps=args.fps,
    )


def run_track(args: argparse.Namespace) -> None:
    output_video = auto_tracking_with_sam2(
        video_path=args.video_path,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        sam2_checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        detection_ckpt_path=args.detection_ckpt_path,
        prompt_type=args.prompt_type,
        frame_step=args.frame_step,
        frame_rate=args.frame_rate,
        detection_conf=args.detection_conf,
        iou_threshold=args.iou_threshold,
        extract_frames=args.extract_frames,
        object_label=args.object_label,
    )
    print(f"Auto-tracking done. Output video: {output_video}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM2-MICE runner: base segmentation, bootstrapping, or auto-tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common defaults
    default_model_cfg = "sam2/sam2_hiera_b+.yaml"
    default_checkpoint = "checkpoints/sam2_base_five_mouse_finetuned.pt"
    default_det_ckpt = "checkpoints_detection/yolo_v11_l.pt"

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Base
    p_base = subparsers.add_parser("base", help="Run base SAM2 video segmentation")
    p_base.add_argument("--video-path", type=str, default=None, help="Path to input video")
    p_base.add_argument("--frames-dir", type=str, default=None, help="Path to extracted frames directory")
    p_base.add_argument("--model-cfg", type=str, default=default_model_cfg, help="SAM2 model config path")
    p_base.add_argument("--checkpoint", type=str, default=default_checkpoint, help="SAM2 checkpoint path")
    p_base.add_argument("--detection-ckpt-path", type=str, default=default_det_ckpt, help="YOLO checkpoint path (for detection prompts)")
    p_base.add_argument("--prompt-source", type=str, choices=["detection", "manual"], default="detection")
    p_base.add_argument("--prompt-type", type=str, choices=["point", "box", "mask"], default="point")
    p_base.add_argument("--save-dir", type=str, default=None, help="Output directory; defaults next to video")
    p_base.add_argument("--fps", type=int, default=10)
    p_base.add_argument("--vos-optimized", action="store_true", help="Enable VOS optimization in predictor")
    p_base.add_argument("--extract-frames", action="store_true", help="Extract frames if frames-dir not provided")
    p_base.set_defaults(func=run_base)

    # Bootstrapping
    p_boots = subparsers.add_parser("boots", help="Run bootstrapping SAM2 segmentation on long videos")
    p_boots.add_argument("--video-path", type=str, required=True, help="Path to input video")
    p_boots.add_argument("--frame-interval", type=int, default=1000, help="Frames per batch")
    p_boots.add_argument("--model-cfg", type=str, default=default_model_cfg)
    p_boots.add_argument("--checkpoint", type=str, default=default_checkpoint)
    p_boots.add_argument("--detection-ckpt-path", type=str, default=default_det_ckpt)
    p_boots.add_argument("--prompt-source", type=str, choices=["detection", "manual"], default="detection")
    p_boots.add_argument("--prompt-type", type=str, choices=["point", "box", "mask"], default="point")
    p_boots.add_argument("--detection-frame-idx", type=int, default=0, help="Global frame index to seed detection in the first segment with mice")
    p_boots.add_argument("--batch-limit", type=int, default=None, help="Optionally limit number of batches processed")
    p_boots.add_argument("--save-dir", type=str, default=None)
    p_boots.add_argument("--fps", type=int, default=10)
    p_boots.add_argument("--vos-optimized", action="store_true")
    p_boots.add_argument("--extract-frames", action="store_true", help="Extract frames into batches if needed")
    p_boots.set_defaults(func=run_boots)

    # Auto-tracking
    p_track = subparsers.add_parser("track", help="Run auto-tracking pipeline")
    p_track.add_argument("--video-path", type=str, required=True)
    p_track.add_argument("--frames-dir", type=str, default=None, help="Directory for frames (can be auto-extracted)")
    p_track.add_argument("--output-dir", type=str, default=None, help="Directory to write masks/json/video")
    p_track.add_argument("--model-cfg", type=str, default=default_model_cfg)
    p_track.add_argument("--checkpoint", type=str, default=default_checkpoint)
    p_track.add_argument("--detection-ckpt-path", type=str, default=default_det_ckpt)
    p_track.add_argument("--prompt-type", type=str, choices=["mask"], default="mask", help="Auto-tracking currently supports mask prompts")
    p_track.add_argument("--frame-step", type=int, default=30, help="Process n frames per iteration")
    p_track.add_argument("--frame-rate", type=int, default=30, help="Output video frame rate")
    p_track.add_argument("--detection-conf", type=float, default=0.5)
    p_track.add_argument("--iou-threshold", type=float, default=0.3)
    p_track.add_argument("--object-label", type=str, default="mouse")
    p_track.add_argument("--extract-frames", action="store_true")
    p_track.set_defaults(func=run_track)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Basic path checks for common files when using defaults
    if not os.path.exists(args.model_cfg):
        print(f"Warning: model config not found at {args.model_cfg}. Update --model-cfg if needed.")
    if not os.path.exists(args.checkpoint):
        print(f"Warning: checkpoint not found at {args.checkpoint}. Update --checkpoint if needed.")
    if hasattr(args, "detection_ckpt_path") and args.detection_ckpt_path and not os.path.exists(args.detection_ckpt_path):
        print(f"Warning: detection checkpoint not found at {args.detection_ckpt_path}. Update --detection-ckpt-path if needed.")

    return args.func(args)


if __name__ == "__main__":
    main()
