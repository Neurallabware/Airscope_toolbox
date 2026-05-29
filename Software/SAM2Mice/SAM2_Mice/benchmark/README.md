## Benchmark with DeepLabCut and SuperAnimal

We benchmarked SAM2Mice against [SuperAnimal](https://www.nature.com/articles/s41467-024-48792-2) and [DeepLabCut](https://www.nature.com/articles/s41592-022-01443-0) using the MOT metrics computed with *pymotmetrics*. For evaluation, we converted SAM2Mice's segmentation masks and DeepLabCut's keypoints into bounding boxes.

### DeepLabCut Environment
For DeepLabCut we used the TensorFlow-based 2.3.10 release. We provide the output files for evaluation data in the link below.
If you want to run with yourself, install with:

```bash
pip install 'deeplabcut[gui,tf]==2.3.10'
```

If you only need inference (no GUI), you can omit `gui`.

### Additional Python Dependencies
Besides project requirements, the benchmarking notebook/code uses:

```bash
pip install motmetrics tables h5py matplotlib numpy pillow
```

`motmetrics` is sometimes imported internally via `pymotmetrics`; installing `motmetrics` covers it.

You could download the validation data from [google drive](https://drive.google.com/drive/folders/122uAEX-X8AzZk4sBm5zemSk0FWKehDhZ?usp=drive_link). The data folder will be as follows:


```bash
DATA_ROOT/
├── openfield_five_mouse/           # The raw frames of the video
├── ground_truth_bb_labels/         # Bounding box annotated, stored as labelimg format
├── SuperAnimal_output/             # Output files inferenced by SuperAnimal
├── DLC_trained_output/             # Output files inferenced by DeepLabCut
└── openfield_five_mouse_seg/       # Segment masks saved in a pickle file
```

### Overview of the Benchmarking Steps
1. Convert SAM2Mice segmentation masks to YOLO-format bounding boxes (using `mask_to_box_labelimg.py`).
2. (Optional) Visualize converted labels over the raw frames to sanity‑check quality.
3. Compute MOT metrics (MOTA, IDF1, IDs, FN, etc.) for SuperAnimal, DeepLabCut, and SAM2Mice via `compute_metrics.py`.
4. Plot aggregated metrics for comparison.

You can follow the interactive workflow in the notebook [bench_mark.ipynb](bench_mark.ipynb).

## Tracking Metrics Evaluation

This repo uses py-motmetrics to compute standard MOT metrics, including MOTA, MOTP, IDF1, ID switches, FP, and FN.

Supported inputs:

- Keypoints HDF5: DLC-style multi-index DataFrame with `x`, `y`, and `likelihood` coordinates per individual. We reconstruct per-individual boxes with a margin and average likelihood as confidence.
- YOLO label directories: per-frame `.txt` files such as `00000.txt`, `00001.txt`, and `classes.txt`. Each label line is `class_id cx cy w h [optional_conf]`. Boxes are normalized to `[0, 1]` and converted to pixels using the provided image width and height.

### Convert SAM2Mice Masks to YOLO Boxes

If you have packed binary masks per frame, convert them into YOLO boxes:

```bash
python -m SAM2_Mice.benchmark.mask_to_box_labelimg convert \
  --pickle /path/to/processed_segments.pkl \
  --out /path/to/SAM2_labels \
  --height 1024 \
  --width 1224 \
  --format yolo \
  --min-area 900
```

Optionally visualize boxes over images:

```bash
python -m SAM2_Mice.benchmark.mask_to_box_labelimg vis \
  --images /path/to/images \
  --labels /path/to/SAM2_labels \
  --out /path/to/vis_out \
  --format yolo
```

### Evaluate Keypoint Predictions

Use `compute_metrics.py` to evaluate ground-truth YOLO boxes against predicted HDF5 keypoints. The keypoints are converted to boxes internally.

```python
from SAM2_Mice.benchmark.compute_metrics import compute_mot_metrics

results, summary = compute_mot_metrics(
    gt_path="/path/to/gt_keypoints.h5",
    pred_path="/path/to/pred_keypoints.h5",
    tracker_type="bbox",
    margin=5.0,
    match_in_first_frame=False,
    save_path="metrics_keypoints.pkl",
)
print(summary)
```

### Evaluate YOLO Box Predictions

Use `compute_metrics.py` to evaluate ground-truth YOLO boxes against predicted YOLO boxes from masks.

```python
from SAM2_Mice.benchmark.compute_metrics import compute_mot_metrics

results, summary = compute_mot_metrics(
    gt_path="/path/to/GT_labels_yolo",
    pred_path="/path/to/SAM2_labels",
    tracker_type="bbox",
    yolo_img_shape=(1224, 1024),
    match_in_first_frame=False,
    save_path="metrics_sam2.pkl",
)
print(summary)
```

## Results

We benchmarked SAM2Mice with SuperAnimal and DeepLabCut on 3-mouse and 5-mouse videos. Compared with keypoint-based tracking methods, SAM2Mice provides richer spatial descriptions, greater noise robustness, and resistance to long-duration drift.

![Comparison with keypoints](../../assets/compare_with_keypoins.png)

### 3 Mice Open Field

| Method | MOTA (up) | FN (down) | Recall (up) | IDF1 (up) | ID switches (down) | FM (down) |
| --- | --- | --- | --- | --- | --- | --- |
| SuperAnimal | 0.478 +/- 0.040 | 276.3 +/- 95.8 | 0.724 +/- 0.095 | 0.395 +/- 0.024 | 234.7 +/- 56.9 | 54.7 +/- 10.5 |
| DeepLabCut | 0.881 +/- 0.055 | 60.0 +/- 27.7 | 0.940 +/- 0.028 | 0.940 +/- 0.027 | 0.0 +/- 0.0 | 20.0 +/- 2.9 |
| **SAM2Mice** | **0.979 +/- 0.029** | **10.3 +/- 14.6** | **0.990 +/- 0.015** | **0.990 +/- 0.015** | **0.0 +/- 0.0** | **2.3 +/- 3.3** |

### 5 Mice Open Field

| Method | MOTA (up) | FN (down) | Recall (up) | IDF1 (up) | ID switches (down) | FM (down) |
| --- | --- | --- | --- | --- | --- | --- |
| SuperAnimal | 0.179 +/- 0.016 | 863.0 +/- 11.3 | 0.482 +/- 0.008 | 0.183 +/- 0.009 | 445.0 +/- 31.4 | 218.0 +/- 26.1 |
| DeepLabCut | 0.703 +/- 0.042 | 280.0 +/- 56.4 | 0.832 +/- 0.034 | 0.757 +/- 0.109 | 4.3 +/- 4.8 | 86.3 +/- 1.9 |
| **SAM2Mice** | **0.958 +/- 0.026** | **34.7 +/- 21.5** | **0.979 +/- 0.013** | **0.979 +/- 0.013** | **0.0 +/- 0.0** | **12.3 +/- 4.7** |
