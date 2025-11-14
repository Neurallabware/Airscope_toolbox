import glob
import os
import pickle

import pandas as pd

import warnings

import motmetrics as mm
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from typing import Tuple, Dict, Any

# from deeplabcut.core import trackingutils
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 14})

def convert_bboxes_to_xywh(bboxes: NDArray, inplace: bool = False) -> NDArray:
    """
    Converts bounding box coordinates from [x_min, y_min, x_max, y_max] format
    to [x, y, width, height] format.

    Parameters
    ----------
    bbox : numpy.ndarray
        A 2D array of shape (N, M), where N is the number of bounding boxes
        and M >= 4. The first four columns represent the bounding box in the format
        [x_min, y_min, x_max, y_max].
    inplace : bool, optional
        If True, modifies the input array in place. If False, returns a copy of
        the array with the converted bounding box format. Defaults to False.

    Returns
    -------
    numpy.ndarray or None
        If `inplace` is False, returns a new array of the same shape as `bbox`
        with the format [x, y, width, height]. If `inplace` is True, the input
        array is modified directly, and nothing is returned.
    """
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    if not inplace:
        new_bboxes = bboxes.copy()
        new_bboxes[:, 2] = w
        new_bboxes[:, 3] = h
        return new_bboxes
    bboxes[:, 2] = w
    bboxes[:, 3] = h

_convert_bboxes_to_xywh = convert_bboxes_to_xywh


def reconstruct_bboxes_from_bodyparts(
    data: pd.DataFrame, margin: float, to_xywh: bool = False
) -> NDArray:
    """
    Reconstructs bounding boxes from body part coordinates and likelihoods.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing body part data with a multi-level column index.
        The expected levels include 'x', 'y', and 'likelihood', where:
        - 'x' and 'y' contain the coordinates of the body parts.
        - 'likelihood' contains the confidence scores for each body part.
    margin : float
        The margin to add/subtract from the minimum/maximum coordinates when defining the bounding box.
    to_xywh : bool, optional
        If True, converts the bounding box format from [x_min, y_min, x_max, y_max]
        to [x, y, width, height]. Defaults to False.

    Returns
    -------
    numpy.ndarray
        An array of shape (N, 5), where N is the number of rows in `data`.
        Each row represents a bounding box with the following values:
        - [x_min, y_min, x_max, y_max, likelihood]
        If `to_xywh` is True, the format will be [x, y, width, height, likelihood].

    Notes
    -----
    - NaN values in the input data are ignored when computing the bounding box dimensions.
    - Warnings related to NaN values are suppressed during calculations.
    """
    x = data.xs("x", axis=1, level="coords")
    y = data.xs("y", axis=1, level="coords")
    p = data.xs("likelihood", axis=1, level="coords")
    xy = np.stack([x, y], axis=2)
    bboxes = np.full((data.shape[0], 5), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bboxes[:, :2] = np.nanmin(xy, axis=1) - margin
        bboxes[:, 2:4] = np.nanmax(xy, axis=1) + margin
        bboxes[:, 4] = np.nanmean(p, axis=1)
    if to_xywh:
        convert_bboxes_to_xywh(bboxes, inplace=True)
    return bboxes


def reconstruct_all_bboxes(
    data: pd.DataFrame, margin: float, to_xywh: bool = False
) -> NDArray:
    """
    Reconstructs bounding boxes for multiple individuals from body part data.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing body part data with a multi-level column index.
        The expected levels include:
        - 'individuals': Names of the individuals (e.g., animals).
        - 'x', 'y', and 'likelihood': Coordinate and confidence data for body parts.
    margin : float
        The margin to add/subtract from the minimum/maximum coordinates when defining the bounding box.
    to_xywh : bool
        If True, converts the bounding box format from [x_min, y_min, x_max, y_max]
        to [x, y, width, height].

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (A, F, 5), where:
        - A is the number of individuals (excluding 'single', if present).
        - F is the number of frames (rows) in the input `data`.
        - Each bounding box is represented as [x_min, y_min, x_max, y_max, likelihood].
          If `to_xywh` is True, the format will be [x, y, width, height, likelihood].

    Notes
    -----
    - Individuals are extracted from the 'individuals' level of the DataFrame columns.
    - If an individual named 'single' exists, it is excluded from the bounding box computation.
    - NaN values in the input data are ignored during calculations.
    """
    animals = data.columns.get_level_values("individuals").unique().tolist()
    try:
        animals.remove("single")
    except ValueError:
        pass
    bboxes = np.full((len(animals), data.shape[0], 5), np.nan)
    for n, animal in enumerate(animals):
        bboxes[n] = reconstruct_bboxes_from_bodyparts(
            data.xs(animal, axis=1, level="individuals"), margin, to_xywh
        )
    return bboxes



# — helper to convert normalized YOLO boxes into absolute [x_min,y_min,x_max,y_max]
def yolo_to_xyxy(box: np.ndarray, img_shape: tuple[int,int]) -> np.ndarray:
    """
    box: [class, x_center_norm, y_center_norm, w_norm, h_norm, (opt)conf]
    img_shape: (width, height)
    returns [x_min, y_min, x_max, y_max, score]
    """
    _, xc, yc, w, h, *rest = box
    W, H = img_shape
    xc, yc, w, h = xc * W, yc * H, w * W, h * H
    x1, y1 = xc - w/2, yc - h/2
    x2, y2 = xc + w/2, yc + h/2
    score = rest[0] if rest else 1.0
    return np.array([x1, y1, x2, y2, score])

def read_yolo_folder(folder: str, img_shape: tuple[int,int]) -> NDArray:
    """
    Reads all .txt in `folder` (sorted by frame index), each containing
    YOLO detections per line, and returns a fixed‐shape array:
      (M, F, 5)  where M = max objects in any frame, F = num frames.
    Missing detections are NaN-padded.
    """
    # list and sort files like 00000.txt, 00001.txt, …
    files = sorted([os.path.join(folder, tmp) for tmp in os.listdir(folder) if "class" not in tmp])
    F = len(files)
    # parse each frame
    all_boxes = []
    for fn in files:
        data = np.loadtxt(fn).reshape(-1,5 if os.path.getsize(fn)>0 else (0,5))
        # if file empty, data.shape=(0,5)
        id_list = data[:, 0]
        data = data[np.argsort(id_list)]

        boxes = np.array([yolo_to_xyxy(b, img_shape) for b in data])  # (n_i,5)

        all_boxes.append(boxes)
    # find max detections in any frame
    M = max(b.shape[0] for b in all_boxes)

    # allocate and pad
    arr = np.full((M, F, 5), np.nan)
    for i, boxes in enumerate(all_boxes):
        arr[:boxes.shape[0], i, :] = boxes
    return arr



def calculate_mot_metrics(gt_data: np.ndarray, pred_data: np.ndarray,
                          iou_threshold: float = 0.5,
                          ) -> Dict[str, Any]:
    """
    Calculate Multiple Object Tracking (MOT) evaluation metrics.

    Args:
        gt_data: Ground truth data, shape (num_objects, num_frames, 5)
                 The 5 values are: xmin, ymin, xmax, ymax, confidence
        pred_data: Predicted data, shape (num_objects, num_frames, 5)
        iou_threshold: IoU threshold for determining matches

    Returns:
        A dictionary containing various MOT metrics.
    """

    def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1[:4]
        x2_min, y2_min, x2_max, y2_max = box2[:4]

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    # Create MOT accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    num_objects_gt, num_frames, _ = gt_data.shape
    num_objects_pred = pred_data.shape[0]

    # Process frame by frame
    for frame_id in range(num_frames):
        gt_frame = gt_data[:, frame_id, :]
        pred_frame = pred_data[:, frame_id, :]

        # Filter out invalid detections (confidence == 0)
        valid_gt_mask = gt_frame[:, 4] > 0
        valid_pred_mask = pred_frame[:, 4] > 0

        valid_gt = gt_frame[valid_gt_mask]
        valid_pred = pred_frame[valid_pred_mask]

        # Get valid object IDs
        gt_ids = np.where(valid_gt_mask)[0]
        pred_ids = np.where(valid_pred_mask)[0]

        if len(valid_gt) == 0 and len(valid_pred) == 0:
            acc.update([], [], [])
            continue
        elif len(valid_gt) == 0:
            acc.update([], pred_ids, [])
            continue
        elif len(valid_pred) == 0:
            acc.update(gt_ids, [], [])
            continue

        # Distance matrix (1 - IoU, ∞ if IoU < threshold)
        distances = np.zeros((len(valid_gt), len(valid_pred)))

        for i, gt_box in enumerate(valid_gt):
            for j, pred_box in enumerate(valid_pred):
                iou = bbox_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    distances[i, j] = 1 - iou
                else:
                    distances[i, j] = np.inf

        # Update accumulator
        acc.update(gt_ids, pred_ids, distances)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'num_frames', 'idf1', 'idp', 'idr', 'recall', 'precision',
        'num_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost',
        'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations',
        'mota', 'motp', 'num_transfer', 'num_ascend', 'num_migrate'
    ], name='MOT_Evaluation')

    # Convert to dictionary
    # Only keep the requested metrics: MOTA, recall, FN, IDF1, IDP, IDR, ID switches (IDs), MT, FM
    results = {
        'MOTA': summary['mota'].iloc[0] if not summary['mota'].isna().iloc[0] else 0,
        'Rcll': summary['recall'].iloc[0] if not summary['recall'].isna().iloc[0] else 0,
        'FN': summary['num_misses'].iloc[0] if not summary['num_misses'].isna().iloc[0] else 0,
        'IDF1': summary['idf1'].iloc[0] if not summary['idf1'].isna().iloc[0] else 0,
        'IDP': summary['idp'].iloc[0] if not summary['idp'].isna().iloc[0] else 0,
        'IDR': summary['idr'].iloc[0] if not summary['idr'].isna().iloc[0] else 0,
        'IDs': summary['num_switches'].iloc[0] if not summary['num_switches'].isna().iloc[0] else 0,
        'MT': summary['mostly_tracked'].iloc[0] if not summary['mostly_tracked'].isna().iloc[0] else 0,
        'FM': summary['num_fragmentations'].iloc[0] if not summary['num_fragmentations'].isna().iloc[0] else 0,
    }

    return results, summary


def print_results(results: Dict[str, Any]):
    """Pretty-print formatted MOT results."""
    print("=" * 60)
    print("MOT Evaluation Results")
    print("=" * 60)
    # Print only the selected metrics requested by the user
    # Order: MOTA, Recall, FN, IDF1, IDP, IDR, ID switches (IDs), MT, FM
    print(f"  MOTA: {results.get('MOTA', 0):.3f}")
    print(f"  Recall: {results.get('Rcll', results.get('recall', 0)):.3f}")
    print(f"  FN (False Negatives): {int(results.get('FN', 0))}")
    print(f"  IDF1: {results.get('IDF1', 0):.3f}")
    print(f"  IDP: {results.get('IDP', 0):.3f}")
    print(f"  IDR: {results.get('IDR', 0):.3f}")
    print(f"  ID switches (IDs): {int(results.get('IDs', 0))}")
    print(f"  MT (Mostly Tracked): {int(results.get('MT', 0))}")
    print(f"  FM (Fragmentations): {int(results.get('FM', 0))}")


def compute_mot_metrics(
    gt_path: str,
    pred_path: str,
    tracker_type: str = "bbox",
    # only needed when using YOLO folders:
    yolo_img_shape: tuple[int,int] = (1920, 1080),
    match_in_first_frame=False,
    save_path="",
    **kwargs
) :
    """
    Compute MOT metrics from either two HDF5 files (gt_path, pred_path),
    or from two directories of YOLO-format .txt files.
    If gt_path (or pred_path) is a directory, it is read as YOLO; otherwise as HDF5.

    Parameters
    ----------
    gt_path : str
        Path to ground-truth .h5 file or YOLO .txt directory.
    pred_path : str
        Path to prediction .h5 file or YOLO .txt directory.
    tracker_type : str
        "bbox" or "ellipse".
    yolo_img_shape : (W, H)
        Image dimensions used to convert normalized YOLO → pixel coords.
    margin, to_xywh
        Passed to reconstruct_all_bboxes when using HDF5.
    """
    # Loader depending on input type
    def load(path):
        if os.path.isdir(path):
            return read_yolo_folder(path, yolo_img_shape)
        else:
            df = pd.read_hdf(path)
            if tracker_type == "bbox":
                func = reconstruct_all_bboxes
            elif tracker_type == "ellipse":
                # import trackingutils lazily because it is optional / external
                try:
                    from deeplabcut.core import trackingutils
                except Exception as e:
                    raise ImportError(
                        "deeplabcut is required for 'ellipse' tracker_type: "
                        + str(e)
                    )
                func = trackingutils.reconstruct_all_ellipses
            else:
                raise ValueError(f"Unrecognized tracker type {tracker_type}.")

            return func(df, **kwargs)

    trackers_gt = load(gt_path)
    trackers    = load(pred_path)

    print("Computing MOT metrics...")
    results, summary = calculate_mot_metrics(
        trackers_gt, trackers, iou_threshold=0.5,
    )

    print_results(results)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("\nDetailed summary:")
    print(summary)

    return results, summary




if __name__ == "__main__":


    zero_shot_dlc_pred = r"Y:\LAR\pico\Analysis\tracking\segmentation\benchmark\openfield_five_mice2\openfield_five_mouse_superanimal_topviewmouse_fasterrcnn_resnet50_fpn_v2_hrnet_w32.h5"
    five_mouse_dlc_trained = r"Y:\LAR\pico\Analysis\tracking\segmentation\benchmark\openfield_five_mice2\DLC_trained\openfield_five_mouseDLC_resnet50_PICO_five_miceJun26shuffle0_50000_el_filtered.h5"
    
    SAM2 = r"Y:\LAR\pico\Analysis\tracking\segmentation\benchmark\openfield_five_mice2\SAM2_labels"
    h5_file_gt = r"Y:\LAR\pico\Analysis\tracking\segmentation\benchmark\openfield_five_mice2\bb_labels"


    save_dir ="five_mouse"
    os.makedirs(save_dir)

    results_zero_shot_dlc, _ = compute_mot_metrics(
        h5_file_gt,
        zero_shot_dlc_pred,
        tracker_type="bbox",
        margin=5.0,
        yolo_img_shape=(1224, 1024),
        match_in_first_frame=False,
        save_path=os.path.join(save_dir, "zeroshot_dlc.pkl")

    )

    results_five_mouse_trained_dlc, _ = compute_mot_metrics(
        h5_file_gt,
        five_mouse_dlc_trained,
        tracker_type="bbox",
        margin=5.0,
        yolo_img_shape=(1224, 1024),
        match_in_first_frame=False,
        save_path=os.path.join(save_dir, "trained_dlc.pkl")
    )

    results_SAM2, _ = compute_mot_metrics(
        h5_file_gt,
        SAM2,
        tracker_type="bbox",
        margin=5.0,
        yolo_img_shape=(1224, 1024),
        match_in_first_frame=False,
        save_path=os.path.join(save_dir, "sam2_mice.pkl")
    )

    methods = {
        'Zero-shot DLC': results_zero_shot_dlc,
        'Five-mice DLC': results_five_mouse_trained_dlc,
        'SAM2Mice': results_SAM2,
    }

    # Only plot the requested metrics
    metrics = [
        ('MOTA', 'MOTA', 'Percentage'),
        ('Rcll', 'Recall', 'Percentage'),
        ('FN', 'False negatives','Count'),
        ('IDF1','IDF1', 'Percentage'),
        ('IDP', 'IDP','Percentage'),
        ('IDR', 'IDR','Percentage'),
        ('IDs', "ID switches",  'Count'),
        ('MT', 'MT','Count'),
        ('FM', 'FM', 'Count'),
    ]

    # grid size: 3x3
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    fig.tight_layout(pad=4)

    colors = ['skyblue', 'plum', 'gold']

    for idx, (metric, metric_full, ylabel) in enumerate(metrics):
        ax = axes[idx // n_cols, idx % n_cols]
        x = np.arange(len(methods))
        width = 0.6

        vals = [methods[m].get(metric, 0) for m in methods]
        ax.bar(x, vals, width, label=list(methods.keys()), color=colors[:len(methods)])

        ax.set_xticks(x)
        ax.set_xticklabels(methods.keys(), rotation=30, ha='right')
        ax.set_title(metric_full)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # remove any unused axes (shouldn't be any for 3x3 but keep robust)
    total_plots = n_rows * n_cols
    for idx in range(len(metrics), total_plots):
        fig.delaxes(axes[idx // n_cols, idx % n_cols])

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()








