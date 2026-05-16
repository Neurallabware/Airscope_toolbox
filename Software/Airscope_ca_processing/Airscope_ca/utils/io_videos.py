import multiprocessing as mp
import os

import cv2
import numpy as np
from tqdm import tqdm

try:
    import zarr
except ImportError:
    zarr = None


def require_zarr():
    if zarr is None:
        raise ImportError("zarr is required for this pipeline. Install it with `pip install zarr`.")


def open_zarr_array(path, mode="r", shape=None, chunks=None, dtype=None):
    require_zarr()
    if mode == "w":
        return zarr.open(path, mode=mode, shape=shape, chunks=chunks, dtype=dtype)
    return zarr.open(path, mode=mode)


def read_gray_frame(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def read_frames_parallel(data_path, frame_num, num_workers, data_format="jpg"):
    data_format = normalize_data_format(data_format)
    if data_format == "mp4":
        return read_mp4_frames(data_path, frame_num)

    paths = [
        os.path.join(data_path, f"frame_{i}.{data_format}")
        for i in range(frame_num)
    ]
    if num_workers <= 1:
        return [read_gray_frame(path) for path in tqdm(paths)]

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=num_workers) as pool:
        return list(tqdm(pool.imap(read_gray_frame, paths, chunksize=32), total=len(paths)))


def normalize_data_format(data_format):
    data_format = str(data_format).strip().lower().lstrip(".")
    if data_format not in {"jpg", "tif", "mp4"}:
        raise ValueError(
            f"Unsupported data_format: {data_format!r}. "
            "Supported formats are: jpg, tif, mp4."
        )
    return data_format


def resolve_mp4_path(data_path):
    if os.path.isdir(data_path):
        mp4_paths = sorted(
            os.path.join(data_path, name)
            for name in os.listdir(data_path)
            if name.lower().endswith(".mp4")
        )
        if len(mp4_paths) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .mp4 file in directory {data_path}, "
                f"found {len(mp4_paths)}."
            )
        return mp4_paths[0]
    return data_path


def count_mp4_frames(data_path):
    mp4_path = resolve_mp4_path(data_path)
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open MP4 file: {mp4_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if frame_count <= 0:
        raise ValueError(f"Failed to infer frame count from MP4 file: {mp4_path}")
    return frame_count


def read_mp4_frames(data_path, frame_num):
    mp4_path = resolve_mp4_path(data_path)
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open MP4 file: {mp4_path}")

    frames = []
    try:
        with tqdm(total=frame_num, desc="read mp4") as progress:
            while len(frames) < frame_num:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                progress.update(1)
    finally:
        cap.release()

    if len(frames) != frame_num:
        raise ValueError(
            f"Requested {frame_num} MP4 frames from {mp4_path}, "
            f"but only read {len(frames)}."
        )
    return frames


def build_remap_maps(img_shape, error_XX_new, error_YY_new):
    x = np.arange(0, img_shape[1], dtype=np.float32)
    y = np.arange(0, img_shape[0], dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    map_x = xx + error_XX_new.astype(np.float32, copy=False)
    map_y = yy + error_YY_new.astype(np.float32, copy=False)
    return map_x.astype(np.float32, copy=False), map_y.astype(np.float32, copy=False)


def preprocess_frame(frame, weight_map, map_x, map_y, crop_parameter, up_sample):
    img_change_frame = frame.astype(np.float32) * weight_map
    img_change_frame = cv2.remap(img_change_frame, map_x, map_y, cv2.INTER_CUBIC)
    img_change_frame = img_change_frame[
        crop_parameter[0]:crop_parameter[0] + crop_parameter[2],
        crop_parameter[1]:crop_parameter[1] + crop_parameter[3],
    ]
    frame_max = float(np.max(img_change_frame))

    if up_sample > 1:
        img_change_frame = cv2.resize(
            img_change_frame.astype(np.float32, copy=False),
            (img_change_frame.shape[1] * up_sample, img_change_frame.shape[0] * up_sample),
            interpolation=cv2.INTER_CUBIC,
        )

    return img_change_frame.astype(np.float16), frame_max


def _preprocess_frame_chunk(args):
    start, frames, weight_map, map_x, map_y, crop_parameter, up_sample = args
    processed_frames = []
    max_v = 0
    for offset, frame in enumerate(frames):
        processed, frame_max = preprocess_frame(frame, weight_map, map_x, map_y, crop_parameter, up_sample)
        processed_frames.append((start + offset, processed))
        max_v = max(max_v, frame_max)
    return processed_frames, max_v


def normalize_frame(frame, max_v):
    if max_v <= 0:
        return np.zeros_like(frame, dtype=np.uint8)
    return (frame / max_v * 255).clip(0, 255).astype(np.uint8)


def _normalize_frame_chunk(args):
    start, frames, max_v = args
    normalized_frames = []
    for offset, frame in enumerate(frames):
        normalized_frames.append((start + offset, normalize_frame(frame, max_v)))
    return normalized_frames


def preprocess_frames_parallel(
    video,
    weight_map,
    map_x,
    map_y,
    crop_parameter,
    up_sample,
    num_workers,
    task_chunk_size=8,
):
    if num_workers <= 1:
        frames = []
        max_v = 0
        for frame in tqdm(video):
            processed, frame_max = preprocess_frame(frame, weight_map, map_x, map_y, crop_parameter, up_sample)
            frames.append(processed)
            max_v = max(max_v, frame_max)
        return frames, max_v

    tasks = []
    for start in range(0, len(video), task_chunk_size):
        end = min(start + task_chunk_size, len(video))
        tasks.append((start, video[start:end], weight_map, map_x, map_y, crop_parameter, up_sample))

    frames = [None] * len(video)
    max_v = 0
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=num_workers) as pool:
        for processed_chunk, chunk_max in tqdm(
            pool.imap(_preprocess_frame_chunk, tasks, chunksize=1),
            total=len(tasks),
        ):
            for idx, processed in processed_chunk:
                frames[idx] = processed
            max_v = max(max_v, chunk_max)

    return frames, max_v


def normalize_frames_parallel(video, max_v, num_workers, task_chunk_size=8):
    if num_workers <= 1:
        return np.stack(
            [normalize_frame(frame, max_v) for frame in tqdm(video, desc="normalize")],
            axis=0,
        )

    tasks = []
    for start in range(0, len(video), task_chunk_size):
        end = min(start + task_chunk_size, len(video))
        tasks.append((start, video[start:end], max_v))

    frames = [None] * len(video)
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=num_workers) as pool:
        for normalized_chunk in tqdm(
            pool.imap(_normalize_frame_chunk, tasks, chunksize=1),
            total=len(tasks),
            desc="normalize",
        ):
            for idx, normalized in normalized_chunk:
                frames[idx] = normalized

    return np.stack(frames, axis=0)


def save_video_to_zarr(frames, zarr_path, chunk_size, dtype=np.uint8):
    frame_num = len(frames)
    if frame_num == 0:
        raise ValueError("Cannot save an empty video to Zarr.")

    shape = (frame_num, frames[0].shape[0], frames[0].shape[1])
    chunks = (min(chunk_size, frame_num), shape[1], shape[2])
    arr = open_zarr_array(zarr_path, mode="w", shape=shape, chunks=chunks, dtype=dtype)
    for start in tqdm(range(0, frame_num, chunk_size)):
        end = min(start + chunk_size, frame_num)
        arr[start:end] = np.asarray(frames[start:end], dtype=dtype)
    return arr
