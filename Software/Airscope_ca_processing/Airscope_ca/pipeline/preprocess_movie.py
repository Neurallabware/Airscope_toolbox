import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
from ultralytics import YOLO

from Airscope_ca.preprocessing import adjust_intensity_image, get_vessel_mask, visualize_img_and_mask

_CKPT_DIR = Path(__file__).resolve().parents[2] / 'ckpt'
from Airscope_ca.preprocessing.YOLO_ROI_detection import YOLO_center_detect
from Airscope_ca.utils.io_videos import build_remap_maps, normalize_frames_parallel, open_zarr_array, preprocess_frames_parallel, save_video_to_zarr
from Airscope_ca.Visualization import save_video


def preprocess_movie(cfg, video, paths, logger):
    """Prepare the motion-corrected calcium movie for neural extraction.

    This stage applies the non-neural image conditioning needed before
    background rejection: intensity-field correction, optical-distortion
    correction, automatic crop refinement, optional upsampling, normalization,
    and vessel-mask extraction. The preprocessed movie is saved to Zarr for the
    background-removal stage, while vessel images are written to disk for later
    ROI filtering and visualization.

    When ``video`` is ``None``, the stage is being resumed from a previous run.
    In that mode the saved vessel image and vessel mask are loaded from
    ``paths.out_path`` without reprocessing frames.

    Args:
        cfg: Composed pipeline configuration. Uses ``preprocessing``, ``rmbg``,
            motion frame rate, debug-video settings, and AVI quality settings.
        video: Motion-corrected movie frames, or ``None`` to load prior vessel
            products from disk.
        paths: Pipeline path bundle for Zarr, vessel-mask, and debug outputs.
        logger: Pipeline logger for correction, crop, and output metadata.

    Returns:
        Tuple ``(vessel_img, vessel_mask)`` used by final ROI filtering.
    """
    if video is None:
        logger.info('MC processing jumped. loading preprocessed results\n')
        vessel_img = cv2.imread(
            os.path.join(paths.out_path, 'vessel_image.tif'), cv2.IMREAD_GRAYSCALE
        ).astype(np.float16)
        vessel_mask = cv2.imread(
            os.path.join(paths.out_path, 'vessel_mask.tif'), cv2.IMREAD_GRAYSCALE
        ).astype(np.float32)
        vessel_mask = vessel_mask / 255
        return vessel_img, vessel_mask

    pre_cfg = cfg.preprocessing
    rmbg_cfg = cfg.rmbg
    fr = cfg.motion.fr
    crop_parameter = list(pre_cfg.crop_parameter)
    intensity_corr_flag = pre_cfg.intensity_corr_flag
    preprocess_workers = pre_cfg.preprocess_workers
    up_sample = rmbg_cfg.up_sample
    up_sample_flag = up_sample > 1
    rmbg_chunk_size = rmbg_cfg.rmbg_chunk_size
    preprocess_chunk_size = pre_cfg.preprocess_chunk_size
    zarr_chunk_size = pre_cfg.zarr_chunk_size
    save_debug_video = cfg.save_debug_video
    avi_quality = cfg.avi_quality

    logger.info('=======>do upsampling<=======\n')
    if intensity_corr_flag:
        weight_map, I_change = adjust_intensity_image(video[0])
    else:
        weight_map = np.ones_like(video[0], dtype=np.float32)
        I_change = 1

    Correction_Map = np.load(_CKPT_DIR / 'CorrectionMap.npz')
    error_XX_new = Correction_Map['error_XX_new']
    error_YY_new = Correction_Map['error_YY_new']
    map_x, map_y = build_remap_maps(video[0].shape, error_XX_new, error_YY_new)

    if save_debug_video:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(error_XX_new)
        ax[1].imshow(error_XX_new)
        plt.close(fig)

    logger.info('=======>field distortion correction and intensity uniformity<=======\n')
    logger.info(f"Former crop_parameter is {pre_cfg.crop_parameter}")

    model = YOLO(str(_CKPT_DIR / 'yolo_v8s.pt'))
    crop_parameter_init = crop_parameter.copy()

    img_change_frame = video[0].astype(np.float64) * weight_map
    img_change_frame = cv2.remap(img_change_frame, map_x, map_y, cv2.INTER_CUBIC)
    img_change_frame = cv2.normalize(img_change_frame, None, 0, 255, cv2.NORM_MINMAX)
    img_change_frame = img_change_frame.astype(np.uint8)

    crop_parameter = YOLO_center_detect(model, img_change_frame, crop_parameter_init, output_dir=paths.out_path)
    logger.info(f"Corrected crop_parameter is as follows: {crop_parameter}")
    del model

    logger.info(f'=======>preprocess frames with {preprocess_workers} workers<=======\n')
    video_preprocessed, max_v = preprocess_frames_parallel(
        video,
        weight_map.astype(np.float32, copy=False),
        map_x,
        map_y,
        crop_parameter,
        up_sample if up_sample_flag else 1,
        preprocess_workers,
        task_chunk_size=preprocess_chunk_size
    )

    del video

    video_preprocessed = normalize_frames_parallel(video_preprocessed, max_v, preprocess_workers,
                                                   task_chunk_size=preprocess_chunk_size)

    logger.info(f'=======>save preprocessed video to Zarr: {paths.preprocess_zarr_path}<=======\n')
    save_video_to_zarr(video_preprocessed, paths.preprocess_zarr_path, zarr_chunk_size, dtype=np.uint8)

    if save_debug_video:
        save_video(video_preprocessed, fr, paths.out_path + '/preprocessed.avi', quality=avi_quality)

    logger.info('=======>get vessel mask<=======\n')
    norm_img = np.array(video_preprocessed[0]).astype(np.float32)
    norm_img = norm_img / norm_img.max()
    vessel_img, vessel_mask = get_vessel_mask(norm_img, str(_CKPT_DIR / 'vessel_model.pt'))

    visualize_img_and_mask(vessel_img, vessel_mask, save_path=os.path.join(paths.out_path, 'vessel_mask.png'))
    plt.close()

    tifffile.imwrite(os.path.join(paths.out_path, 'vessel_mask.tif'), ((vessel_mask > 0) * 255).astype(np.uint8))
    tifffile.imwrite(os.path.join(paths.out_path, 'vessel_image.tif'), (vessel_img * 255).astype(np.uint8))

    logger.info(f'video_preprocessed: ndarray {video_preprocessed.shape}, {video_preprocessed.dtype}')
    logger.info(f'vessel_mask: dtype {vessel_mask.dtype}, min {vessel_mask.min()}, max {vessel_mask.max()}')

    del video_preprocessed
    return vessel_img, vessel_mask
