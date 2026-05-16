import os
import math
import shutil
import tempfile

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

import Airscope_ca.caiman as caiman
from Airscope_ca.caiman import normcorre_function
from Airscope_ca.suite2p_registration import registration_wrapper
from Airscope_ca.suite2p_registration.defaults import default_settings
from Airscope_ca.utils.io_videos import save_video_to_zarr
from Airscope_ca.Visualization import save_video

MC_METHODS = ['caiman', 'suite2p']


def correct_motion(cfg, video, paths, logger):
    """Correct frame-to-frame motion in the raw calcium movie.

    The configured motion-correction backend is selected from ``cfg.motion`` and
    applied chunk-by-chunk to keep memory use bounded. Corrected frames are
    returned for the next stage and also written to ``paths.mc_zarr_path`` as a
    Zarr array. When debug output is enabled, an AVI preview is exported in the
    pipeline output directory.

    Args:
        cfg: Composed pipeline configuration. Uses ``motion`` settings,
            ``save_debug_video``, and ``avi_quality``.
        video: Raw movie frames produced by :func:`load_raw_movie`.
        paths: Pipeline path bundle containing motion-correction output paths.
        logger: Pipeline logger for backend, chunk, and output messages.

    Returns:
        List of motion-corrected frames in processing order.

    Raises:
        NotImplementedError: If ``cfg.motion.method`` is not one of
        ``MC_METHODS``.
    """
    mc_cfg = cfg.motion
    method = str(mc_cfg.get('method', 'caiman'))
    mc_chunk_size = mc_cfg.mc_chunk_size
    zarr_chunk_size = mc_cfg.zarr_chunk_size
    save_debug_video = cfg.save_debug_video
    avi_quality = cfg.avi_quality

    logger.info('=======>motion correction<=======\n')
    _remove_chunk_dirs(paths.mc_out, logger)

    if method == 'caiman':
        mc_video = _correct_motion_caiman(mc_cfg, video, paths, logger, mc_chunk_size, save_debug_video)
    elif method == 'suite2p':
        mc_video = _correct_motion_suite2p(mc_cfg, video, paths, logger, mc_chunk_size, save_debug_video)
    else:
        raise NotImplementedError(
            f"Unsupported motion correction method: '{method}'. "
            f"Supported methods: {MC_METHODS}"
        )

    del video

    if save_debug_video:
        fr = mc_cfg.fr
        save_video(mc_video, fr, paths.out_path + '/mc.avi', quality=avi_quality)

    logger.info(f'=======>save MC video to Zarr: {paths.mc_zarr_path}<=======\n')
    save_video_to_zarr(mc_video, paths.mc_zarr_path, zarr_chunk_size, dtype=np.uint8)

    return mc_video

def _correct_motion_caiman(mc_cfg, video, paths, logger, mc_chunk_size, save_debug_video):
    """Run CaImAn/NormCorre motion correction over movie chunks.

    A local CaImAn cluster is created for the duration of the stage. The
    template is carried across chunks so later chunks register against the
    evolving reference image. Temporary CaImAn outputs are written inside a
    temporary directory and removed automatically after each chunk.

    Args:
        mc_cfg: Motion-correction subsection of the pipeline config.
        video: Raw movie frames to correct.
        paths: Pipeline path bundle, passed through for consistency with other
            backends.
        logger: Pipeline logger for chunk progress.
        mc_chunk_size: Number of frames to process in each chunk.
        save_debug_video: Whether backend diagnostics should be written.

    Returns:
        List of corrected frames emitted by NormCorre.
    """
    caiman_cfg = mc_cfg.caiman
    fr = mc_cfg.fr
    max_shifts = tuple(caiman_cfg.max_shifts)
    strides = tuple(caiman_cfg.strides)
    overlaps = tuple(caiman_cfg.overlaps)
    max_deviation_rigid = caiman_cfg.max_deviation_rigid
    pw_rigid = caiman_cfg.pw_rigid
    shifts_opencv = caiman_cfg.shifts_opencv
    border_nan = caiman_cfg.border_nan
    downsample_ratio = caiman_cfg.downsample_ratio

    N_chunk = math.ceil(len(video) / mc_chunk_size)
    template = None
    mc_video = []
    c, dview, n_processes = caiman.cluster.setup_cluster(
        backend='local', n_processes=24, single_thread=False)

    try:
        for i in tqdm(range(N_chunk)):
            image_stack = np.stack(video[i * mc_chunk_size:(i + 1) * mc_chunk_size], axis=0)
            logger.info(f'MC chunk {i}: shape={image_stack.shape}')

            with tempfile.TemporaryDirectory(prefix='airscope_mc_') as tmp_outpath:
                m_nonrig, bord_px_rig, bord_px_els, template = normcorre_function(
                    video=image_stack,
                    fr=fr,
                    max_shifts=max_shifts,
                    strides=strides,
                    overlaps=overlaps,
                    max_deviation_rigid=max_deviation_rigid,
                    pw_rigid=pw_rigid,
                    shifts_opencv=shifts_opencv,
                    border_nan=border_nan,
                    downsample_ratio=downsample_ratio,
                    outpath=tmp_outpath,
                    template=template,
                    save_movie=bool(caiman_cfg.save_movie) and save_debug_video,
                    save_diagnostics=save_debug_video,
                    dview=dview)

            frames_list = [m_nonrig[i] for i in range(m_nonrig.shape[0])]
            mc_video.extend(frames_list)
    finally:
        caiman.stop_server(dview=dview)

    return mc_video


def _correct_motion_suite2p(mc_cfg, video, paths, logger, mc_chunk_size, save_debug_video):
    """Run Suite2p rigid registration over movie chunks.

    The Suite2p wrapper registers each chunk in place, accumulates rigid
    ``xoff`` and ``yoff`` traces, and writes diagnostic plots for shift traces
    and the reference image. The first chunk establishes the reference image
    used for diagnostics.

    Args:
        mc_cfg: Motion-correction subsection of the pipeline config.
        video: Raw movie frames to correct.
        paths: Pipeline path bundle containing ``mc_out`` for diagnostics.
        logger: Pipeline logger for chunk progress.
        mc_chunk_size: Number of frames to register per chunk.
        save_debug_video: Present for backend signature symmetry; Suite2p
            diagnostics are currently saved regardless of this flag.

    Returns:
        List of Suite2p-registered frames.
    """
    settings = default_settings()["registration"]
    overrides = mc_cfg.get('suite2p', {})
    if isinstance(overrides, DictConfig):
        overrides = OmegaConf.to_container(overrides, resolve=True)
    if overrides:
        settings = {**settings, **overrides}

    device_str = str(mc_cfg.get('device', 'cpu'))
    device = torch.device(device_str)

    N_chunk = math.ceil(len(video) / mc_chunk_size)
    template = None
    mc_video = []
    all_yoff = []
    all_xoff = []

    for i in tqdm(range(N_chunk)):
        image_stack = np.stack(video[i * mc_chunk_size:(i + 1) * mc_chunk_size], axis=0).copy()
        logger.info(f'MC chunk {i}: shape={image_stack.shape}')

        reg_outputs = registration_wrapper(
            f_reg=image_stack,
            f_raw=None,
            f_reg_chan2=None,
            f_raw_chan2=None,
            refImg=template,
            align_by_chan2=False,
            settings=settings,
            save_path=None,
            aspect=1,
            badframes=None,
            device=device,
        )

        updated_template = reg_outputs["refImg"]
        if template is None:
            template = updated_template

        frames_list = [image_stack[j] for j in range(image_stack.shape[0])]
        mc_video.extend(frames_list)

        all_yoff.append(reg_outputs["yoff"])
        all_xoff.append(reg_outputs["xoff"])

    # ── save registration diagnostics ────────────────────────────────────────
    _save_suite2p_diagnostics(
        save_dir=paths.mc_out,
        yoff=np.concatenate(all_yoff),
        xoff=np.concatenate(all_xoff),
        refImg=template,
    )

    return mc_video


def _save_suite2p_diagnostics(save_dir, yoff, xoff, refImg):
    """Save Suite2p registration diagnostics.

    Two diagnostic products are written into ``save_dir``: a PNG plot of rigid
    y/x offsets across frames and, when available, the reference image as both a
    PNG preview and ``.npy`` array for later inspection.

    Args:
        save_dir: Directory where diagnostic files should be created.
        yoff: Per-frame rigid vertical offsets from Suite2p.
        xoff: Per-frame rigid horizontal offsets from Suite2p.
        refImg: Suite2p reference image, or ``None`` when unavailable.
    """
    os.makedirs(save_dir, exist_ok=True)

    # shift plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    frames = np.arange(len(yoff))
    axes[0].plot(frames, yoff, linewidth=0.8, color='steelblue')
    axes[0].set_ylabel('Y offset (px)')
    axes[0].set_title('Suite2p rigid shifts')
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[1].plot(frames, xoff, linewidth=0.8, color='tomato')
    axes[1].set_ylabel('X offset (px)')
    axes[1].set_xlabel('Frame')
    axes[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'suite2p_shifts.png'), dpi=150)
    plt.close(fig)

    # reference image
    if refImg is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(refImg, cmap='gray')
        ax.set_title('Suite2p reference image')
        ax.axis('off')
        fig.savefig(os.path.join(save_dir, 'suite2p_refImg.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        np.save(os.path.join(save_dir, 'suite2p_refImg.npy'), refImg)


def _remove_chunk_dirs(parent_dir, logger):
    """Remove stale temporary chunk directories from a stage output folder.

    Args:
        parent_dir: Directory that may contain ``chunk_*`` temporary folders.
        logger: Pipeline logger used to report each removed directory.
    """
    if not os.path.isdir(parent_dir):
        return
    for name in os.listdir(parent_dir):
        path = os.path.join(parent_dir, name)
        if name.startswith('chunk_') and os.path.isdir(path):
            shutil.rmtree(path)
            logger.info(f'Removed stale temporary chunk directory: {path}')
