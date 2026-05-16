"""Visualization helpers for Airscope calcium-processing notebooks.

The functions in this module are intentionally lightweight and notebook-safe:
they avoid mutating pipeline outputs, handle missing intermediate files with
clear messages, and keep plotting defaults consistent across stage notebooks.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from scipy.io import loadmat

try:
    import zarr
except ImportError:  # pragma: no cover - notebooks report this at runtime.
    zarr = None


def set_publication_style():
    """Set compact, publication-oriented matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "image.cmap": "gray",
        }
    )


def require_zarr():
    """Return the zarr module or raise a notebook-friendly error."""
    if zarr is None:
        raise ImportError("zarr is required for these QC cells.")
    return zarr


def open_zarr(path, mode="r"):
    """Open a Zarr movie from ``path``."""
    return require_zarr().open(str(path), mode=mode)


def sample_indices(n_frames, n=8):
    """Return evenly spaced frame indices for a movie."""
    n = max(1, min(int(n), int(n_frames)))
    return np.linspace(0, int(n_frames) - 1, n, dtype=int)


def robust_limits(image, low=1, high=99):
    """Return percentile display limits for robust image contrast."""
    arr = np.asarray(image)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0, 1
    vmin, vmax = np.percentile(finite, [low, high])
    if vmin == vmax:
        vmax = vmin + 1
    return float(vmin), float(vmax)


def load_movie_samples(movie, indices):
    """Load selected frames from a list-like movie or Zarr array."""
    return np.asarray([np.asarray(movie[int(idx)]) for idx in indices])


def plot_frame_montage(movie, indices=None, n=8, title="Frame montage", cmap="gray"):
    """Plot a row of representative movie frames."""
    n_frames = getattr(movie, "shape", (len(movie),))[0]
    if indices is None:
        indices = sample_indices(n_frames, n=n)
    frames = load_movie_samples(movie, indices)
    vmin, vmax = robust_limits(frames)

    fig, axes = plt.subplots(1, len(indices), figsize=(2.2 * len(indices), 2.3))
    axes = np.atleast_1d(axes)
    for ax, frame, idx in zip(axes, frames, indices):
        ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"t={int(idx)}")
        ax.axis("off")
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return fig


def plot_projection_panel(movie, title="Movie projections", n_samples=128):
    """Plot mean, standard deviation, and max projections from sampled frames."""
    n_frames = getattr(movie, "shape", (len(movie),))[0]
    frames = load_movie_samples(movie, sample_indices(n_frames, n=n_samples)).astype(np.float32)
    projections = [
        ("Mean", frames.mean(axis=0)),
        ("Std", frames.std(axis=0)),
        ("Max", frames.max(axis=0)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (label, image) in zip(axes, projections):
        vmin, vmax = robust_limits(image)
        im = ax.imshow(image, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_before_after_frames(before, after, indices=None, n=5, title="Before/after frames"):
    """Compare matched frames from two movie-like objects."""
    n_frames = min(getattr(before, "shape", (len(before),))[0], getattr(after, "shape", (len(after),))[0])
    if indices is None:
        indices = sample_indices(n_frames, n=n)
    before_frames = load_movie_samples(before, indices)
    after_frames = load_movie_samples(after, indices)
    b_vmin, b_vmax = robust_limits(before_frames)
    a_vmin, a_vmax = robust_limits(after_frames)

    fig, axes = plt.subplots(2, len(indices), figsize=(2.2 * len(indices), 4.2))
    for col, idx in enumerate(indices):
        axes[0, col].imshow(before_frames[col], vmin=b_vmin, vmax=b_vmax)
        axes[0, col].set_title(f"t={int(idx)}")
        axes[0, col].axis("off")
        axes[1, col].imshow(after_frames[col], vmin=a_vmin, vmax=a_vmax)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("Before")
    axes[1, 0].set_ylabel("After")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_difference_panel(before, after, index=None, title="Difference panel"):
    """Plot before, after, and after-minus-before for one matched frame."""
    n_frames = min(getattr(before, "shape", (len(before),))[0], getattr(after, "shape", (len(after),))[0])
    if index is None:
        index = int(n_frames // 2)
    b = np.asarray(before[index]).astype(np.float32)
    a = np.asarray(after[index]).astype(np.float32)
    diff = a - b
    vmin, vmax = robust_limits(np.stack([b, a]))
    dlim = max(abs(np.percentile(diff, 1)), abs(np.percentile(diff, 99)), 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(b, vmin=vmin, vmax=vmax)
    axes[0].set_title("Before")
    axes[1].imshow(a, vmin=vmin, vmax=vmax)
    axes[1].set_title("After")
    im = axes[2].imshow(diff, cmap="coolwarm", vmin=-dlim, vmax=dlim)
    axes[2].set_title("After - before")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.suptitle(f"{title} (frame {index})", y=1.02)
    fig.tight_layout()
    return fig


def overlay_mask(image, mask, title="Mask overlay", alpha=0.35):
    """Overlay a binary mask on an image."""
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(5, 5))
    vmin, vmax = robust_limits(image)
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    masked = np.ma.masked_where(np.asarray(mask) <= 0, mask)
    yellow_cmap = ListedColormap(["yellow"])
    ax.imshow(masked, cmap=yellow_cmap, alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def display_existing_image(path, title=None):
    """Display a saved image if it exists, otherwise print a clear message."""
    path = Path(path)
    if title:
        print(title)
    if path.exists():
        display(Image(filename=str(path)))
    else:
        print(f"Missing file: {path}")


def list_stage_outputs(path, patterns=("*",), max_items=30):
    """Print selected files in a stage output directory."""
    path = Path(path)
    print(f"Output directory: {path}")
    if not path.exists():
        print("Directory does not exist yet.")
        return []
    files = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    files = sorted(set(p for p in files if p.is_file()))
    for item in files[:max_items]:
        print(item.name)
    if len(files) > max_items:
        print(f"... {len(files) - max_items} more files")
    return files


def plot_background_removal_qc(preprocess_arr, rmbg_arr, index=None):
    """Compare preprocessed and background-removed frames plus residual.

    Uses independent robust_limits for before and after so the neuron-enhanced
    frame is not washed out by the much brighter preprocessed input range.
    """
    n_frames = min(
        getattr(preprocess_arr, "shape", (len(preprocess_arr),))[0],
        getattr(rmbg_arr, "shape", (len(rmbg_arr),))[0],
    )
    if index is None:
        index = int(n_frames // 2)
    b = np.asarray(preprocess_arr[index]).astype(np.float32)
    a = np.asarray(rmbg_arr[index]).astype(np.float32)
    diff = a - b
    b_vmin, b_vmax = robust_limits(b)
    a_vmin, a_vmax = robust_limits(a)
    dlim = max(abs(np.percentile(diff, 1)), abs(np.percentile(diff, 99)), 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(b, cmap="gray", vmin=b_vmin, vmax=b_vmax)
    axes[0].set_title("Preprocessed")
    axes[1].imshow(a, cmap="gray", vmin=a_vmin, vmax=a_vmax)
    axes[1].set_title("Neuron-enhanced")
    im = axes[2].imshow(diff, cmap="coolwarm", vmin=-dlim, vmax=dlim)
    axes[2].set_title("After - before")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.suptitle(f"Background removal QC (frame {index})", y=1.02)
    fig.tight_layout()
    return fig


def plot_roi_centers(cm, d1, d2, title="ROI center map"):
    """Plot ROI center coordinates in image space."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(cm[:, 0], cm[:, 1], s=4, alpha=0.65, c=np.arange(len(cm)), cmap="viridis")
    ax.set_xlim(0, d1)
    ax.set_ylim(d2, 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"{title} (n={len(cm)})")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_roi_area_histogram(A, title="ROI area distribution"):
    """Plot ROI area distribution from a mask stack."""
    areas = np.asarray(A, dtype=bool).sum(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(areas, bins=50, color="0.25")
    ax.axvline(np.median(areas), color="tab:red", lw=1, label=f"median={np.median(areas):.0f}")
    ax.set_xlabel("ROI area (pixels)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_trace_heatmap(C, max_rois=300, title="Calcium trace heatmap"):
    """Plot normalized trace heatmap for a subset of ROIs."""
    traces = np.asarray(C[: min(max_rois, C.shape[0])], dtype=np.float32)
    traces = traces - np.nanmedian(traces, axis=1, keepdims=True)
    denom = np.nanpercentile(np.abs(traces), 99, axis=1, keepdims=True)
    traces = traces / np.maximum(denom, 1e-6)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(traces, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("ROI")
    ax.set_title(f"{title} (first {traces.shape[0]} ROIs)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="normalized dF")
    fig.tight_layout()
    return fig


def plot_selected_traces(C, roi_indices=None, n=12, title="Representative calcium traces"):
    """Plot offset representative traces."""
    C = np.asarray(C, dtype=np.float32)
    if roi_indices is None:
        roi_indices = sample_indices(C.shape[0], n=n)
    fig, ax = plt.subplots(figsize=(10, 5))
    offset = 0
    yticks = []
    ylabels = []
    for idx in roi_indices:
        trace = C[int(idx)]
        trace = trace - np.nanmedian(trace)
        scale = np.nanpercentile(np.abs(trace), 99)
        trace = trace / max(scale, 1e-6)
        ax.plot(trace + offset, lw=0.8)
        yticks.append(offset)
        ylabels.append(str(int(idx)))
        offset += 2.5
    ax.set_xlabel("Frame")
    ax.set_ylabel("ROI")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def load_signal_outputs(seg_out):
    """Load saved neural signal outputs from ``seg_out``."""
    seg_out = Path(seg_out)
    C = loadmat(seg_out / "infer_results.mat")["C"]
    cm = loadmat(seg_out / "cm.mat")["cm"]
    return C, cm


def find_raw_frame_paths(data_path, data_format, limit=12):
    """Return representative raw frame paths for image-sequence inputs."""
    data_format = str(data_format).strip().lower().lstrip(".")
    if data_format == "mp4":
        return []
    paths = sorted(glob.glob(os.path.join(str(data_path), f"frame_*.{data_format}")))
    if not paths:
        return []
    idx = sample_indices(len(paths), n=min(limit, len(paths)))
    return [paths[int(i)] for i in idx]


def read_raw_frame_preview(path):
    """Read a grayscale raw frame preview from an image path."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image
