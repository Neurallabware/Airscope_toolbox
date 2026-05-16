import os
import math
import concurrent.futures
import threading
import shutil
import torch.multiprocessing as mp

import numpy as np
from tqdm import tqdm

from Airscope_ca.background_rejection import background_rejection
from Airscope_ca.background_rejection.inference import load_bg_rejection_model
from Airscope_ca.utils.io_videos import open_zarr_array
from Airscope_ca.Visualization import save_video


def remove_background(cfg, paths, logger):
    """Remove slowly varying background from the preprocessed calcium movie.

    The preprocessed movie is read from ``paths.preprocess_zarr_path`` and
    processed in frame chunks by the background-rejection neural network. The
    neuron-enhanced movie is written directly to ``paths.rmbg_zarr_path`` as
    ``uint8`` data. The stage supports both single-device inference and
    multi-GPU ``mp.spawn`` execution using the same model checkpoint.

    Args:
        cfg: Composed pipeline configuration. Uses ``rmbg`` model/inference
            settings, frame rate, debug-video settings, and AVI quality.
        paths: Pipeline path bundle containing preprocessed input and
            background-removal output locations.
        logger: Pipeline logger for chunk scheduling and output messages.
    """
    rmbg_cfg = cfg.rmbg
    rmbg_chunk_size = rmbg_cfg.rmbg_chunk_size
    zarr_chunk_size = rmbg_cfg.zarr_chunk_size
    rmbg_gsize = rmbg_cfg.rmbg_gsize
    ckpt_pth = rmbg_cfg.ckpt_pth
    device = rmbg_cfg.device
    gpu_ids = str(rmbg_cfg.gpu_ids)
    batch_size = int(rmbg_cfg.get('batch_size', 4))
    print_interval = max(0, int(rmbg_cfg.get('print_interval', 10)))
    use_amp = bool(rmbg_cfg.get('use_amp', True))
    copy_interval = max(1, int(rmbg_cfg.get('copy_interval', 4)))
    num_process_per_gpu = max(1, int(rmbg_cfg.get('num_process_per_gpu', 1)))
    multi_gpu = bool(rmbg_cfg.get('multi_gpu', False))
    save_debug_video = cfg.save_debug_video
    avi_quality = cfg.avi_quality
    fr = cfg.motion.fr

    logger.info('=======>background subtraction<=======\n')
    _remove_chunk_dirs(paths.rmbg_out, logger)

    preprocess_arr = open_zarr_array(paths.preprocess_zarr_path, mode='r')
    rmbg_chunk_num = math.ceil(preprocess_arr.shape[0] / rmbg_chunk_size)
    rmbg_arr = open_zarr_array(
        paths.rmbg_zarr_path,
        mode='w',
        shape=preprocess_arr.shape,
        chunks=(
            min(zarr_chunk_size, preprocess_arr.shape[0]),
            preprocess_arr.shape[1],
            preprocess_arr.shape[2],
        ),
        dtype='uint8',
    )

    if multi_gpu:
        _remove_background_multi_gpu(
            rmbg_cfg=rmbg_cfg, preprocess_arr=preprocess_arr, rmbg_arr=rmbg_arr,
            paths=paths, logger=logger, rmbg_chunk_size=rmbg_chunk_size,
            rmbg_chunk_num=rmbg_chunk_num, rmbg_gsize=rmbg_gsize, ckpt_pth=ckpt_pth,
            gpu_ids=gpu_ids, batch_size=batch_size, print_interval=print_interval,
            use_amp=use_amp, copy_interval=copy_interval,
            num_process_per_gpu=num_process_per_gpu,
        )
    else:
        _remove_background_single_gpu(
            preprocess_arr=preprocess_arr, rmbg_arr=rmbg_arr,
            paths=paths, logger=logger, rmbg_chunk_size=rmbg_chunk_size,
            rmbg_chunk_num=rmbg_chunk_num, rmbg_gsize=rmbg_gsize, ckpt_pth=ckpt_pth,
            device=device, gpu_ids=gpu_ids, batch_size=batch_size,
            print_interval=print_interval, use_amp=use_amp,
            copy_interval=copy_interval,
        )

    logger.info(f'=======>RMBG Zarr saved: {paths.rmbg_zarr_path}<=======\n')

    if save_debug_video:
        neuron_video = np.asarray(open_zarr_array(paths.rmbg_zarr_path, mode='r'))
        save_video(neuron_video, fr, paths.out_path + '/rmbg.avi', quality=avi_quality)

def _background_removal_worker(rank, chunk_assignments, preprocess_zarr_path, rmbg_zarr_path,
                               rmbg_out, rmbg_chunk_size, rmbg_gsize, ckpt_pth, gpu_ids,
                               batch_size, print_interval, use_amp, copy_interval):
    """Process assigned background-removal chunks in a spawned worker.

    Each worker maps its rank to one configured GPU, loads its own model
    instance, opens the shared input/output Zarr arrays, and writes only the
    chunks assigned to that rank. This keeps interprocess communication small
    and avoids returning large movie blocks through ``mp.spawn``.

    Args:
        rank: Worker rank supplied by ``torch.multiprocessing.spawn``.
        chunk_assignments: List of chunk-index lists, one list per worker.
        preprocess_zarr_path: Path to the preprocessed input Zarr movie.
        rmbg_zarr_path: Path to the output neuron-enhanced Zarr movie.
        rmbg_out: Directory for model debug artifacts when enabled downstream.
        rmbg_chunk_size: Number of frames per inference chunk.
        rmbg_gsize: Expected neuron radius used by the background model.
        ckpt_pth: Model checkpoint path.
        gpu_ids: Comma-separated physical GPU ids exposed to workers.
        batch_size: Number of patches per model forward pass.
        print_interval: Patch-level progress print interval.
        use_amp: Whether CUDA autocast should be enabled.
        copy_interval: Number of patch batches to accumulate before CPU copy.
    """
    gpu_list = [g.strip() for g in gpu_ids.split(',') if g.strip()]
    gpu_index = rank % len(gpu_list)
    physical_gpu_id = gpu_list[gpu_index]
    device = f'cuda:{gpu_index}'
    net, _device = load_bg_rejection_model(
        ckpt_pth=ckpt_pth, gsize=rmbg_gsize, device=device, gpu_ids=gpu_ids,
    )
    preprocess_arr = open_zarr_array(preprocess_zarr_path, mode='r')
    rmbg_arr = open_zarr_array(rmbg_zarr_path, mode='r+')
    n_frames = preprocess_arr.shape[0]

    for i in chunk_assignments[rank]:
        start = i * rmbg_chunk_size
        end = min((i + 1) * rmbg_chunk_size, n_frames)
        tmp_video = preprocess_arr[start:end].astype(np.float16)

        print(
            f'[worker {rank} -> GPU {physical_gpu_id} / {_device}] '
            f'RMBG chunk {i} [{start}:{end}]',
            flush=True,
        )
        out_neuron = background_rejection(
            video=tmp_video,
            gsize=rmbg_gsize,
            ckpt_pth=ckpt_pth,
            batch_size=batch_size,
            output_dir=rmbg_out,
            output_zarr=None,
            save_bg=False,
            save_debug_video=False,
            return_bg=False,
            net=net,
            log_prefix=f'[rank {rank}] ',
            print_interval=print_interval,
            use_amp=use_amp,
            copy_interval=copy_interval,
        )
        rmbg_arr[start:end] = out_neuron.clip(0, 255).astype(np.uint8)


def _remove_background_single_gpu(*, preprocess_arr, rmbg_arr, paths, logger,
                                  rmbg_chunk_size, rmbg_chunk_num, rmbg_gsize, ckpt_pth,
                                  device, gpu_ids, batch_size, print_interval, use_amp,
                                  copy_interval):
    """Run background removal sequentially with one model instance.

    The model is loaded once, then reused for every frame chunk. Each output
    chunk is clipped to image range and written directly into the output Zarr
    array, avoiding a full in-memory copy of the background-removed movie.

    Args:
        preprocess_arr: Open Zarr array containing the preprocessed movie.
        rmbg_arr: Open writable Zarr array for neuron-enhanced output.
        paths: Pipeline path bundle containing ``rmbg_out``.
        logger: Pipeline logger for chunk progress.
        rmbg_chunk_size: Number of frames per inference chunk.
        rmbg_chunk_num: Total number of chunks to process.
        rmbg_gsize: Expected neuron radius used by the background model.
        ckpt_pth: Model checkpoint path.
        device: Torch device string for inference.
        gpu_ids: CUDA-visible GPU id string.
        batch_size: Number of patches per model forward pass.
        print_interval: Patch-level progress print interval.
        use_amp: Whether CUDA autocast should be enabled.
        copy_interval: Number of patch batches to accumulate before CPU copy.
    """
    net, _device = load_bg_rejection_model(
        ckpt_pth=ckpt_pth, gsize=rmbg_gsize, device=device, gpu_ids=gpu_ids,
    )
    n_frames = preprocess_arr.shape[0]
    for i in tqdm(range(rmbg_chunk_num)):
        start = i * rmbg_chunk_size
        end = min((i + 1) * rmbg_chunk_size, n_frames)
        tmp_video = preprocess_arr[start:end].astype(np.float16)

        logger.info(f'RMBG chunk {i} [{start}:{end}] on {_device}')
        out_neuron = background_rejection(
            video=tmp_video,
            gsize=rmbg_gsize,
            ckpt_pth=ckpt_pth,
            batch_size=batch_size,
            output_dir=paths.rmbg_out,
            output_zarr=None,
            save_bg=False,
            save_debug_video=False,
            return_bg=False,
            net=net,
            print_interval=print_interval,
            use_amp=use_amp,
            copy_interval=copy_interval,
        )
        rmbg_arr[start:end] = out_neuron.clip(0, 255).astype(np.uint8)


def _remove_background_multi_gpu(*, rmbg_cfg, preprocess_arr, rmbg_arr, paths, logger,
                                 rmbg_chunk_size, rmbg_chunk_num, rmbg_gsize, ckpt_pth,
                                 gpu_ids, batch_size, print_interval, use_amp,
                                 copy_interval, num_process_per_gpu):
    """Run background removal in parallel across configured GPUs.

    Chunks are dispatched round-robin across worker processes. Worker rank is
    mapped to GPU by rank % N_GPUs, so num_process_per_gpu > 1 allows multiple
    chunks to run concurrently on the same GPU.

    Args:
        rmbg_cfg: Background-removal config subsection, retained for call-site
            symmetry with ``remove_background``.
        preprocess_arr: Open Zarr array containing the preprocessed movie.
        rmbg_arr: Open writable Zarr array for neuron-enhanced output.
        paths: Pipeline path bundle containing input/output Zarr paths.
        logger: Pipeline logger for scheduling metadata.
        rmbg_chunk_size: Number of frames per inference chunk.
        rmbg_chunk_num: Total number of chunks to process.
        rmbg_gsize: Expected neuron radius used by the background model.
        ckpt_pth: Model checkpoint path.
        gpu_ids: Comma-separated physical GPU ids.
        batch_size: Number of patches per model forward pass.
        print_interval: Patch-level progress print interval.
        use_amp: Whether CUDA autocast should be enabled.
        copy_interval: Number of patch batches to accumulate before CPU copy.
        num_process_per_gpu: Number of worker processes assigned per GPU.

    Raises:
        ValueError: If multi-GPU mode is requested without any configured GPU id.
    """
    gpu_list = [g.strip() for g in gpu_ids.split(',') if g.strip()]
    n_gpus = len(gpu_list)
    if n_gpus == 0:
        raise ValueError('rmbg.gpu_ids must contain at least one GPU id for multi_gpu mode.')
    n_workers = max(1, min(rmbg_chunk_num, n_gpus * num_process_per_gpu))

    # Assign chunks round-robin across worker processes. Workers are mapped to
    # GPUs by rank % n_gpus, allowing multiple workers to share one GPU.
    chunk_assignments = [[] for _ in range(n_workers)]
    for i in range(rmbg_chunk_num):
        chunk_assignments[i % n_workers].append(i)

    logger.info(
        f'mp.spawn RMBG: {n_gpus} GPU(s) {gpu_list}, '
        f'{num_process_per_gpu} process(es)/GPU, '
        f'{n_workers} worker(s), {rmbg_chunk_num} chunks'
    )

    mp.spawn(
        _background_removal_worker,
        args=(
            chunk_assignments,
            paths.preprocess_zarr_path,
            paths.rmbg_zarr_path,
            paths.rmbg_out,
            rmbg_chunk_size,
            rmbg_gsize,
            ckpt_pth,
            gpu_ids,
            batch_size,
            print_interval,
            use_amp,
            copy_interval,
        ),
        nprocs=n_workers,
        join=True,
    )


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
