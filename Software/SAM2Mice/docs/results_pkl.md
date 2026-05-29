# Saving and Loading Segmentation Results

After every `run()` / `run_bootstrapping()` call, SAM2Mice automatically saves results to the `save_dir` you specified:

```
save_dir/
├── segment_masks.pickle   # gzip-compressed mask data (main output)
├── segmented_video.mp4    # visualised overlay video
└── frames/ (or saved_masks/)   # per-frame overlay images
```

The `.pickle` file stores all per-frame, per-object binary masks in a memory-efficient packbits format. It must be loaded together with the **mask shape** `(H, W)` that is returned by `run()`.

---

## Saving

`save_video_segments` is called automatically inside `run()` and `run_bootstrapping()`. The return value of those calls includes the shape tuple needed for loading.

```python
video_segments, output_video_path = predictor.run(
    frames_dir=frames_dir,
    prompt_source="manual",
    prompt_type="point",
    save_dir="results/base",
    fps=20,
)
# shape is stored automatically; retrieve it from the manager if needed:
shape = predictor.segment_manager._mask_shape   # (H, W)
```

If you need to save manually (e.g. after post-processing):

```python
shape = predictor.segment_manager.save_video_segments("results/base/segment_masks.pickle")
print(f"Saved. Mask shape: {shape}")   # e.g. (1080, 1920)
```

---

## Loading

Use `load_segments` on any `VideoSegmentationInference` or `BootstrappingVideoSegmentationInference` instance. You must supply the `(H, W)` shape that was returned when the file was saved.

```python
from SAM2_Mice.segmentation import VideoSegmentationInference

predictor = VideoSegmentationInference(
    model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
    checkpoint_path="checkpoints/SAM2_Mice_base_plus.pt",
)

import os
frames_dir = "data/frames"
frame_names = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
frame_paths = [os.path.join(frames_dir, n) for n in frame_names]

shape = (1080, 1920)   # must match the shape returned when saving

segments = predictor.load_segments(
    pickle_path="results/base/segment_masks.pickle",
    shape=shape,
    frame_paths=frame_paths,   # optional; required for re-rendering
)
```

`segments` is a list aligned with `frame_paths`. Each element is either `None` (no detection on that frame) or a dict mapping object ID → mask array of shape `(1, H, W)` with `int8` values (0 or 1).

```python
frame_idx = 0
seg = segments[frame_idx]   # None  OR  {obj_id: np.ndarray (1,H,W)}

if seg is not None:
    for obj_id, mask in seg.items():
        print(f"frame {frame_idx}, object {obj_id}: mask sum = {mask.sum()}")
```

---

## Re-rendering after loading

Once frames are loaded via `load_segments`, you can re-generate the overlay video without re-running inference:

```python
predictor.segment_manager.generate_masked_video_and_image(
    output_video_path="results/base/rerendered_video.mp4",
    fps=20,
    temp_folder="results/base/rerendered_frames",
    save_masks=True,   # set False to delete per-frame images after video is written
)
```

Or use the `supervision`-based annotator which also draws bounding boxes and labels:

```python
predictor.segment_manager.generate_masked_video_supervision(
    output_video_path="results/base/annotated_video.mp4",
    fps=20,
    mask_save_folder="results/base/annotated_frames",
)
```

---

## Direct access to decompressed masks

If you want to iterate over all frames without loading the full list into memory at once, use the internal method directly:

```python
import gzip, pickle
import numpy as np

shape = (1080, 1920)  # (H, W)

with gzip.open("results/base/segment_masks.pickle", "rb") as f:
    packed_segments = pickle.load(f)  # list of None | {obj_id: uint8 packbits array}

for frame_idx, packed in enumerate(packed_segments):
    if packed is None:
        continue
    for obj_id, bits in packed.items():
        mask = np.unpackbits(bits)[: shape[0] * shape[1]].reshape(1, *shape).astype(np.int8)
        # mask shape: (1, H, W), values 0 or 1
```

> **Note:** The packbits format pads the last byte if `H * W` is not a multiple of 8. Always slice with `[: H * W]` before reshaping.

---

## Bootstrapping results

`run_bootstrapping()` writes a single merged `segment_masks.pickle` covering all batches, at the same path and in the same format as the basic workflow. Loading is identical — use the same `load_segments` call shown above.
