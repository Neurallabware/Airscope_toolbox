import os
import cv2
import gzip
import pickle
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Sequence

from tqdm import tqdm


def load_pickle_maybe_gzip(path: str):
    """Load pickle file; try gzip first, fall back to plain pickle."""
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):  # not gzipped
        with open(path, 'rb') as f:
            return pickle.load(f)


def ensure_dir(p: Path) -> None:
    """Create a directory and all parents if needed."""
    p.mkdir(parents=True, exist_ok=True)


def load_video_frame_at_index(video_path: str, index: int) -> np.ndarray | None:
    """Load one frame from a video by zero-based frame index."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def video_shape(video_path: str) -> Tuple[int, int]:
    """Return video frame shape as (height, width)."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from video: {video_path}")
    h, w = frame.shape[:2]
    return h, w


def video_len(video_path: str) -> int:
    """Return the frame count reported by OpenCV for a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def mask_to_boxes(mask: np.ndarray, min_area: int = 900) -> List[Tuple[int, int, int, int]]:
    """Return list of (x1,y1,x2,y2) boxes from external contours with area >= min_area."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, x + w, y + h))
    return boxes


def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert absolute xyxy box coordinates to normalized YOLO cxcywh."""
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def write_yolo(label_path: Path, objs: List[Tuple[int, Tuple[int, int, int, int]]], img_w: int, img_h: int) -> None:
    """Write one YOLO label file from class IDs and xyxy boxes."""
    lines: List[str] = []
    for cid, (x1, y1, x2, y2) in objs:
        cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(label_path, 'w') as f:
        f.write("\n".join(lines))


def write_classes(labels_dir: Path, names: Sequence[str]) -> None:
    """Write class names to a YOLO classes.txt file."""
    with open(labels_dir / 'classes.txt', 'w') as f:
        for n in names:
            f.write(n + "\n")


def select_frames_random(n: int, total: int, seed: int | None = None) -> List[int]:
    """Select sorted random frame indices from a fixed frame count."""
    rng = random.Random(seed)
    n = min(n, total)
    return sorted(rng.sample(range(total), n)) if n > 0 else []


def select_frames_kmeans(video_path: str, n: int, resize_w: int = 32, seed: int | None = None) -> List[int]:
    """Cluster frames using MiniBatchKMeans on downscaled grayscale frames; pick 1 per cluster."""
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:  # pragma: no cover
        raise RuntimeError("KMeans mode requires scikit-learn. Please install: pip install scikit-learn") from e

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        return []
    h0, w0 = frame0.shape[:2]
    ratio = resize_w / float(w0)
    resize_h = max(1, int(round(h0 * ratio)))

    feats: List[np.ndarray] = []
    indices: List[int] = []
    rng = np.random.default_rng(seed)

    # Read all frames (can be slow on long videos). For speed, you could stride.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    with tqdm(total=total, desc='Scanning frames for KMeans', leave=False) as pbar:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            feats.append(small.reshape(-1).astype(np.float32))
            indices.append(idx)
            idx += 1
            pbar.update(1)
    cap.release()

    if not feats:
        return []
    X = np.stack(feats, axis=0)
    X -= X.mean(axis=0, keepdims=True)
    k = min(n, len(indices))
    if k <= 0:
        return []
    mbk = MiniBatchKMeans(n_clusters=k, random_state=seed or 0, batch_size=256, n_init='auto')
    labels = mbk.fit_predict(X)

    chosen: List[int] = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        m = int(rng.choice(members))
        chosen.append(indices[m])
    return sorted(chosen)


def convert_masks_to_yolo(
    video_path: str,
    pickle_path: str,
    images_dir: str,
    labels_dir: str,
    mode: str,
    num_frames: int | None,
    specific_frames: Sequence[int] | None,
    min_area: int = 900,
    class_mode: str = 'single',  # 'single' -> class 0 for all, 'object' -> use obj_id
    seed: int | None = None,
) -> None:
    """Export selected video frames and matching packed masks as YOLO data."""
    segments: List[Dict[int, np.ndarray]] = load_pickle_maybe_gzip(pickle_path)
    total_seg = len(segments)

    total_vid = video_len(video_path)
    total = min(total_seg, total_vid)
    if total <= 0:
        raise RuntimeError("No frames to process: empty video or segments.")

    h, w = video_shape(video_path)

    # frame selection
    if mode == 'random':
        n = num_frames or min(200, total)
        chosen = select_frames_random(n, total, seed)
    elif mode == 'kmeans':
        n = num_frames or min(200, total)
        chosen = select_frames_kmeans(video_path, n=n, resize_w=32, seed=seed)
    elif mode == 'specific':
        if not specific_frames:
            raise ValueError("--frames must be provided when mode=='specific'")
        chosen = sorted([i for i in specific_frames if 0 <= i < total])
    else:
        raise ValueError("mode must be one of 'random', 'kmeans', 'specific'")

    if not chosen:
        print("No frames selected; nothing to do.")
        return

    img_out = Path(images_dir)
    lbl_out = Path(labels_dir)
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    # Optionally write classes.txt
    if class_mode == 'single':
        write_classes(lbl_out, ["mouse"])  # single class
    else:
        # infer max obj id to build class list
        max_id = -1
        for seg in segments:
            if seg:
                max_id = max(max_id, max(seg.keys()))
        names = [f"mouse_{i}" for i in range(max(1, max_id + 1))]
        write_classes(lbl_out, names)

    for idx in tqdm(chosen, desc='Exporting YOLO data'):
        frame = load_video_frame_at_index(video_path, idx)
        if frame is None:
            continue
        # save image
        stem = f"frame_{idx:05d}"
        img_path = img_out / f"{stem}.jpg"
        cv2.imwrite(str(img_path), frame)

        seg = segments[idx]
        objs: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for obj_id, mask_bits in seg.items():
            mask = np.unpackbits(mask_bits).astype(np.uint8).reshape((h, w))
            for (x1, y1, x2, y2) in mask_to_boxes(mask, min_area=min_area):
                cid = 0 if class_mode == 'single' else int(obj_id)
                objs.append((cid, (x1, y1, x2, y2)))

        write_yolo(lbl_out / f"{stem}.txt", objs, w, h)


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser for mask-to-YOLO conversion."""
    p = argparse.ArgumentParser(description='Convert packed masks to YOLO labels and extract frames.')
    p.add_argument('--video', required=True, help='Path to video file')
    p.add_argument('--pickle', required=True, help='Path to processed_segments pickle (gz or plain)')
    p.add_argument('--images-out', required=True, help='Output directory for images')
    p.add_argument('--labels-out', required=True, help='Output directory for YOLO labels')
    p.add_argument('--mode', choices=['random', 'kmeans', 'specific'], default='random', help='Frame selection mode')
    p.add_argument('--num-frames', type=int, default=None, help='Number of frames to select (random/kmeans)')
    p.add_argument('--frames', type=int, nargs='*', default=None, help='Specific frame indices when mode=specific')
    p.add_argument('--min-area', type=int, default=900, help='Minimum bbox area to keep')
    p.add_argument('--class-mode', choices=['single', 'object'], default='single', help='single=all class 0, object=use obj_id as class id')
    p.add_argument('--seed', type=int, default=None, help='Random seed')
    return p


def main():
    """Parse command-line arguments and run the conversion command."""
    args = build_argparser().parse_args()
    convert_masks_to_yolo(
        video_path=args.video,
        pickle_path=args.pickle,
        images_dir=args.images_out,
        labels_dir=args.labels_out,
        mode=args.mode,
        num_frames=args.num_frames,
        specific_frames=args.frames,
        min_area=args.min_area,
        class_mode=args.class_mode,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
