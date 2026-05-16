import os
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
from scipy.io import savemat
from skimage import io

from Airscope_ca.segmentation import convert_to_sparse, save_sparse_frames_to_mat
from Airscope_ca.preprocessing import visualize_img_and_mask
from Airscope_ca.Visualization import com, plot_cm, plot_trace, filter_masks_by_roundness


def export_calcium_results(cfg, vessel_img, vessel_mask, A, C, cm, d1, d2, paths, logger):
    """Filter extracted ROIs and export final calcium-analysis products.

    This final stage converts raw segmentation output into user-facing calcium
    results. It saves center-of-mass plots, removes ROIs overlapping cleaned and
    dilated vessel masks, filters masks by shape quality, recomputes centers for
    retained ROIs, writes filtered masks/traces/centers, and generates summary
    images plus trace plots for review.

    Args:
        cfg: Composed pipeline configuration. Included for a consistent stage
            signature; filtering constants are currently defined in this module.
        vessel_img: Vessel image produced by :func:`preprocess_movie`.
        vessel_mask: Vessel mask produced by :func:`preprocess_movie`.
        A: ROI mask stack from :func:`extract_neural_signals`.
        C: Calcium trace matrix from :func:`extract_neural_signals`.
        cm: ROI center coordinates from :func:`extract_neural_signals`.
        d1: Frame width used by center plotting and vessel-mask resizing.
        d2: Frame height used by center plotting and vessel-mask resizing.
        paths: Pipeline path bundle containing output and segmentation folders.
        logger: Pipeline logger for export progress.
    """
    logger.info('=======>visualization<=======\n')

    plot_cm(cm, d1, d2,
            save_path=os.path.join(paths.seg_out, 'cm.png'),
            save_svg_path=os.path.join(paths.seg_out, 'cm.svg'))
    plt.pause(5)
    plt.close()

    # --- vessel filtering ---
    invalid_idx = []
    vessel_mask_resize = cv2.resize(vessel_mask, (d1, d2), interpolation=cv2.INTER_NEAREST)

    min_size = 300
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_mask_resize.astype(np.uint8))
    filtered_mask = np.zeros_like(vessel_mask_resize)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == label] = 1
    vessel_mask_resize = filtered_mask

    tifffile.imwrite(os.path.join(paths.out_path, 'vessel_mask_clean.tif'),
                     ((vessel_mask_resize > 0) * 255).astype(np.uint8))

    vessel_mask_resize = cv2.dilate(vessel_mask_resize.astype(np.float32),
                                    np.ones((7, 7), np.uint8), iterations=1)
    tifffile.imwrite(os.path.join(paths.out_path, 'vessel_mask_dilate.tif'),
                     ((vessel_mask_resize > 0) * 255).astype(np.uint8))

    for i in range(len(cm)):
        if vessel_mask_resize[int(cm[i][1]), int(cm[i][0])] > 0:  # cm is (x, y) not (h, w)
            invalid_idx.append(i)
    del cm

    A_filtered = np.delete(A, invalid_idx, axis=0)
    C_filtered = np.delete(C, invalid_idx, axis=0)

    # --- shape filtering ---
    invalid_idx2 = filter_masks_by_roundness(A_filtered, max_axis_ratio=3, min_occupancy=0.5)
    A_filtered = np.delete(A_filtered, invalid_idx2, axis=0)
    C_filtered = np.delete(C_filtered, invalid_idx2, axis=0)

    mask_sum = np.sum(A_filtered, axis=0).astype('uint8')
    mask_sum = mask_sum * int(255 / np.max(mask_sum))
    io.imsave(paths.seg_out + '/SEG_SUM_filtered.png', mask_sum)

    # --- filtered center of mass ---
    A_2d_filtered = A_filtered.reshape(A_filtered.shape[0], -1).T
    cm_chunk = 1000
    cm_chunk_num = math.ceil(A_2d_filtered.shape[1] / cm_chunk)
    cm_filtered = None
    for i in tqdm(range(cm_chunk_num)):
        cm_tmp = com(A=A_2d_filtered[:, i * cm_chunk:(i + 1) * cm_chunk].astype(np.float32), d1=d1, d2=d2)
        cm_filtered = cm_tmp if i == 0 else np.concatenate((cm_filtered, cm_tmp), axis=0)

    plot_cm(cm_filtered, d1, d2,
            save_path=os.path.join(paths.seg_out, 'cm_wo_vessel.png'),
            save_svg_path=os.path.join(paths.seg_out, 'cm_wo_vessel.svg'))
    plt.pause(5)
    plt.close()

    visualize_img_and_mask(
        vessel_img.astype(np.float32),
        np.sum(A, axis=0).astype(np.float32),
        np.sum(A_filtered, axis=0).astype(np.float32),
        save_path=os.path.join(paths.seg_out, 'neuron_mask.png'),
        save_svg_path=os.path.join(paths.seg_out, 'neuron_mask.svg'),
    )
    plt.pause(5)
    plt.close()

    # --- save filtered results ---
    sparse_frames = convert_to_sparse(A_filtered)
    save_sparse_frames_to_mat(sparse_frames, paths.seg_out + '/seg_results_filtered.mat')
    savemat(paths.seg_out + '/infer_results_filtered.mat', {'C': C_filtered})
    savemat(paths.seg_out + '/cm_filtered.mat', {'cm': cm_filtered})

    logger.info('Plotting Traces.....................\n')
    plot_trace(
        paths.seg_out + '/seg_results_filtered.mat',
        paths.seg_out + '/infer_results_filtered.mat',
        paths.seg_out + '/Neuron_trace/',
        frame_len=1000,
        neuron_step=100,
    )
