import os
import time
import datetime
import tifffile
import numpy as np
import torch
from .utils import crop_patches, concat_patches, save_video
from .model import BG_Rejection
try:
    import zarr
except ImportError:
    zarr = None

from tqdm import tqdm

def load_bg_rejection_model(ckpt_pth, gsize=6, device='cuda', gpu_ids='0',
                             in_channels=1, out_channels=1, f_maps=32):
    """Load BG_Rejection model once and return it for reuse across chunks.

    Call this once before the chunk loop in ``pipeline.background_removal`` and pass the returned
    ``net`` to :func:`background_rejection` to avoid reloading the checkpoint
    on every chunk iteration.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = torch.device(device)
    net = BG_Rejection(in_channels=in_channels, out_channels=out_channels,
                       f_maps=f_maps, gsize=gsize, infer=True)
    net.to(device)
    checkpoint = torch.load(ckpt_pth, map_location='cpu')
    try:
        net.load_state_dict(checkpoint['net'])
    except Exception:
        new_state_dict = {"module." + k: v for k, v in checkpoint['net'].items()}
        net.load_state_dict(new_state_dict)
    net.eval()
    return net, device


def background_rejection(
                         video,
                         gsize = 6, # neuron radius
                         ckpt_pth = '',
                         patch_size ='(128, 128, 128)', # background rejection patch size
                         stride_size = '(96, 96, 96)', # background rejection stride size
                         batch_size = 4,
                         device = 'cuda', # utilize GPU or CPU
                         gpu_ids ='0', # GPU ids
                         output_dir = '',
                         in_channels = 1,
                         out_channels = 1,
                         f_maps = 32,
                         output_zarr = None,
                         save_bg = True,
                         save_debug_video = True,
                         return_bg = True,
                         net = None,
                         log_prefix = '',
                         print_interval = 1,
                         use_amp = False,
                         copy_interval = 1):
    """Run background rejection on a video chunk.

    Pass a pre-loaded ``net`` (from :func:`load_bg_rejection_model`) to avoid
    reloading the model for every chunk. If ``net`` is None the model is loaded
    from ``ckpt_pth`` on the fly (backward compatible).
    """
    if net is None:
        net, device = load_bg_rejection_model(
            ckpt_pth=ckpt_pth, gsize=gsize, device=device, gpu_ids=gpu_ids,
            in_channels=in_channels, out_channels=out_channels, f_maps=f_maps,
        )
    else:
        device = next(net.parameters()).device
    copy_interval = max(1, int(copy_interval))
    amp_enabled = bool(use_amp) and device.type == 'cuda'

    # mkdir
    if save_debug_video:
        os.makedirs(output_dir, exist_ok=True)
    if save_bg:
        os.makedirs(os.path.join(output_dir, 'bg'), exist_ok=True)

    # patching
    patch_size = eval(patch_size)
    stride_size = eval(stride_size)
    overlap_size = (patch_size[0] - stride_size[0],
                    patch_size[1] - stride_size[1],
                    patch_size[2] - stride_size[2])

    # T x H x W format; avoid copying when caller already passes an ndarray.
    if isinstance(video, np.ndarray):
        video = np.asarray(video)
    else:
        video = np.stack(video, axis=0)
    frameN = video.shape[0]
    
    with torch.inference_mode():
        start_time = time.time()
        origin_shape = video.shape

        # preprocessing
        video_mean = np.mean(video, axis=0, keepdims=True)
        video = video - video_mean
        
        # patch chopping
        video_patches, patch_num = crop_patches(video, patch_size, stride_size)
        iter_num = len(video_patches) // batch_size if len(video_patches) % batch_size == 0 else len(
            video_patches) // batch_size + 1

        patch_shape = video_patches[0].shape
        bg_patches = (
            np.empty((len(video_patches), *patch_shape), dtype=np.float16)
            if (save_bg or return_bg) else None
        )
        neuron_patches = np.empty((len(video_patches), *patch_shape), dtype=np.float16)
        pending_bg = []
        pending_neuron = []

        def flush_patch_outputs(end_idx):
            nonlocal pending_bg, pending_neuron
            if not pending_neuron:
                return

            neuron_cpu = torch.cat(pending_neuron, dim=0).to(
                'cpu', dtype=torch.float16
            ).numpy()
            start_idx = end_idx - neuron_cpu.shape[0]
            neuron_patches[start_idx:end_idx] = neuron_cpu
            pending_neuron = []

            if bg_patches is not None:
                bg_cpu = torch.cat(pending_bg, dim=0).to(
                    'cpu', dtype=torch.float16
                ).numpy()
                bg_patches[start_idx:end_idx] = bg_cpu
                pending_bg = []

        for j in range(iter_num):
            patch_start = j * batch_size
            patch_end = min((j + 1) * batch_size, len(video_patches))
            batch = np.stack(video_patches[patch_start:patch_end], axis=0)
            input_dtype = torch.float16 if amp_enabled else torch.float32
            input = torch.from_numpy(batch).to(
                device=device, dtype=input_dtype, non_blocking=True
            ).unsqueeze(1)
            with torch.autocast(
                device_type='cuda', dtype=torch.float16, enabled=amp_enabled
            ):
                out_bg, out_neuron = net(input)

            out_bg = out_bg.squeeze(1)
            out_neuron = out_neuron.squeeze(1)

            if bg_patches is not None:
                pending_bg.append(out_bg.detach())
            pending_neuron.append(out_neuron.detach())
            if len(pending_neuron) >= copy_interval or j == iter_num - 1:
                flush_patch_outputs(patch_end)

            should_print = (
                print_interval > 0
                and (j == 0 or (j + 1) % print_interval == 0 or j == iter_num - 1)
            )
            if should_print:
                elapsed = time.time() - start_time
                left_time = datetime.timedelta(
                    seconds=int(elapsed / (j + 1) * (iter_num - j - 1))
                )
                print(
                    '%s[Inference Video] [Patches: %d/%d] [ETA: %s]'
                    % (log_prefix, j + 1, iter_num, str(left_time))
                )

        # do the stitching
        out_bg = None
        if bg_patches is not None:
            out_bg = concat_patches(bg_patches, origin_shape, patch_num, patch_size, stride_size, overlap_size)
        out_neuron = concat_patches(neuron_patches, origin_shape, patch_num, patch_size, stride_size,
                                            overlap_size)
        # save file
        if out_bg is not None:
            out_bg = (out_bg + video_mean).clip(0, 255)
        out_neuron = out_neuron.clip(0, 255)
        
        if output_zarr is not None:
            if zarr is None:
                raise ImportError("zarr is required for output_zarr. Install it with `pip install zarr`.")
            rmbg_arr = zarr.open(
                output_zarr,
                mode='w',
                shape=out_neuron.shape,
                chunks=(min(frameN, patch_size[0]), out_neuron.shape[1], out_neuron.shape[2]),
                dtype='uint8',
            )
            rmbg_arr[:] = out_neuron.astype(np.uint8)

        if save_debug_video:
            if save_bg and out_bg is not None:
                save_video(out_bg, 30, os.path.join(output_dir, 'bg', 'bg.avi'))
            save_video(out_neuron,  30, os.path.join(output_dir, 'rmbg.avi'))
        
        if return_bg:
            return out_bg, out_neuron
        return out_neuron
