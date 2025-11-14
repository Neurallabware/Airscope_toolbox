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

