import os
import math

import numpy as np
from tqdm import tqdm
from scipy.io import savemat, loadmat
from skimage import io

from Airscope_ca.segmentation import (
    neuron_segmentation,
    convert_to_sparse,
    load_sparse_frames_from_mat,
    save_sparse_frames_to_mat,
)
from Airscope_ca.utils.io_videos import open_zarr_array
from Airscope_ca.Visualization import com


def extract_neural_signals(cfg, paths, logger):
    """Segment neural ROIs and extract calcium traces from the enhanced movie.

    The background-removed movie is loaded from Zarr, tiled into spatial
    patches, and passed to the neuron-segmentation backend. Patch-local masks
    are expanded back into full-frame coordinates, concatenated across patches,
    and paired with their extracted temporal traces. The stage also computes
    center-of-mass coordinates for every ROI and persists masks, traces, center
    coordinates, and a summed mask preview.

    Args:
        cfg: Composed pipeline configuration. Uses ``segmentation`` parameters
            such as patch size, pixel size, area thresholds, and confidence
            thresholds.
        paths: Pipeline path bundle containing the background-removed input
            Zarr and segmentation output directory.
        logger: Pipeline logger for patch progress and output array metadata.

    Returns:
        Tuple ``(A, C, cm, d1, d2)`` where ``A`` is the full-frame ROI mask
        stack, ``C`` is the calcium trace matrix, ``cm`` contains ROI center
        coordinates, and ``d1``/``d2`` are the frame width and height used by
        downstream visualization.
    """
    seg_cfg = cfg.segmentation
    patch_size = seg_cfg.patch_size

    logger.info('=======>segmentation<=======\n')
    logger.info(f'=======>load RMBG video from Zarr: {paths.rmbg_zarr_path}<=======\n')

    neuron_video = np.asarray(open_zarr_array(paths.rmbg_zarr_path, mode='r'))
    _, d1, d2 = neuron_video.shape

    begin_to_concat = False
    A, C = None, None

    for i in range(0, d1, patch_size):
        for j in range(0, d2, patch_size):
            logger.info(f'=======>segment patch {i},{j}<=======\n')
            patch = neuron_video[:, i:i + patch_size, j:j + patch_size]
            tmp_seg_out = os.path.join(paths.seg_out, f'patch_{i}_{j}')
            os.makedirs(tmp_seg_out, exist_ok=True)

            A_tmp, C_tmp = neuron_segmentation(
                patch,
                tmp_seg_out,
                pixel_size=seg_cfg.pixel_size,
                minArea=seg_cfg.minArea,
                avgArea=seg_cfg.avgArea,
                thresh_pmap=seg_cfg.thresh_pmap,
                thresh_mask=seg_cfg.thresh_mask,
                thresh_COM0=seg_cfg.thresh_COM0,
                thresh_COM=seg_cfg.thresh_COM,
                cons=seg_cfg.cons,
            )

            if isinstance(A_tmp, np.ndarray):
                A_tmp2 = np.zeros([A_tmp.shape[0], d1, d2], dtype=bool)
                A_tmp2[:, i:i + patch_size, j:j + patch_size] = A_tmp
                if not begin_to_concat:
                    A, C = A_tmp2, C_tmp
                    begin_to_concat = True
                else:
                    A = np.concatenate((A, A_tmp2), axis=0)
                    C = np.concatenate((C, C_tmp), axis=0)

    logger.info(f"A.shape: {A.shape}")
    logger.info(f"A.dtype: {A.dtype}")
    logger.info(f"C.shape: {C.shape}")
    logger.info(f"C.dtype: {C.dtype}")

    logger.info('=======>calculate center of these neurons<=======\n')
    d1 = A.shape[2]  # x, width
    d2 = A.shape[1]  # y, height
    A_2d = A.reshape(A.shape[0], -1).T
    cm_chunk = 1000
    cm_chunk_num = math.ceil(A_2d.shape[1] / cm_chunk)
    cm = None
    for i in tqdm(range(cm_chunk_num)):
        cm_tmp = com(A=A_2d[:, i * cm_chunk:(i + 1) * cm_chunk].astype(np.float32), d1=d1, d2=d2)
        cm = cm_tmp if i == 0 else np.concatenate((cm, cm_tmp), axis=0)

    del A_2d
    sparse_frames = convert_to_sparse(A)
    assert A.shape[0] == C.shape[0]
    save_sparse_frames_to_mat(sparse_frames, paths.seg_out + '/seg_results.mat')
    savemat(paths.seg_out + '/infer_results.mat', {'C': C})
    savemat(paths.seg_out + '/cm.mat', {'cm': cm})

    mask_sum = np.sum(A, axis=0).astype('uint8')
    mask_sum = mask_sum * int(255 / np.max(mask_sum))
    io.imsave(paths.seg_out + '/SEG_SUM.png', mask_sum)

    return A, C, cm, d1, d2


def load_extracted_signals(cfg, paths, logger):
    """Load previously extracted ROI masks, traces, and center coordinates.

    This resume helper skips segmentation and reconstructs the same in-memory
    objects returned by :func:`extract_neural_signals` from files written in
    ``paths.seg_out``.

    Args:
        cfg: Composed pipeline configuration. Included for a consistent stage
            signature; currently not read directly.
        paths: Pipeline path bundle containing the segmentation output directory.
        logger: Pipeline logger for resume-stage messages.

    Returns:
        Tuple ``(A, C, cm, d1, d2)`` matching :func:`extract_neural_signals`.
    """
    logger.info('=======>loading segmentation results<=======\n')
    A = np.array(load_sparse_frames_from_mat(filename=paths.seg_out + '/seg_results.mat'))
    d1 = A.shape[2]  # x, width
    d2 = A.shape[1]  # y, height
    C = loadmat(paths.seg_out + '/infer_results.mat')['C']
    cm = loadmat(paths.seg_out + '/cm.mat')['cm']
    return A, C, cm, d1, d2
