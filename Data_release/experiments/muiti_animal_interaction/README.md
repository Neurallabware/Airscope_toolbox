# Multi-animal PICO data release

Release data is available in the shared Google Drive folder [here][release-data].

[release-data]: https://drive.google.com/drive/folders/1g2-mHKbvAN6hFg_vgefyA6oTYkXPSHUd?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

## Quick start

1. Download the released files from the Google Drive link above.
2. Put the downloaded files in one local directory. The directory should contain:
   - `behavior.mp4`
   - `behavior_with_mask.avi`
   - `segment_mask.pickle`
   - `data_aligned.mat`
3. Open `visualize_data.ipynb`.
4. In the path configuration cell, set:

```python
DATA_DIR = Path("/path/to/downloaded/multi_animal_data")
```

If the four data files are in the same directory where the notebook is running,
keep the default:

```python
DATA_DIR = Path(".")
```

5. Run the notebook cells from top to bottom. Generated figures and cached
interaction files are written to `outputs/multi_animal_interaction/`.


## Files

| File | Description |
| --- | --- |
| `behavior.mp4` | Raw multi-animal behavior video. |
| `behavior_with_mask.avi` | Behavior video with SAM2 instance masks overlaid for visual inspection. |
| `segment_mask.pickle` | Gzip-compressed pickle file containing packed binary SAM2 masks for each video frame. |
| `data_aligned.mat` | Final aligned release file. Calcium traces, cortical labels, neuron centers, and mouse mask centers are aligned to the selected behavior-frame interval. |

Video metadata for both `behavior.mp4` and `behavior_with_mask.avi`:

| Field | Value |
| --- | --- |
| Frame size | `968 x 944` pixels, width x height |
| Frame rate | `10 fps` |
| Number of frames | `10805` |
| Duration | `1080.5 s` |

## `data_aligned.mat`

The aligned interval uses behavior frames `2918` to `10518`, inclusive. The
aligned arrays therefore contain `7601` time points.

| Variable | Shape | Description |
| --- | ---: | --- |
| `start_frame` | `1 x 1` | First behavior-video frame included in the aligned interval. |
| `end_frame` | `1 x 1` | Last behavior-video frame included in the aligned interval. |
| `behavior_stamped_times_in_interval` | `7601` | Behavior timestamps for aligned frames, formatted as `DD-Mon-YY/HH-MM-SS-ffffff`. |
| `calcium_<animal>` | `n_neurons x 7601` | Calcium activity interpolated to behavior-frame timestamps. |
| `neuron_name_<animal>` | `n_neurons` | Cortical area label/name for each neuron. |
| `neuron_center_<animal>` | `n_neurons x 2` | Neuron center coordinates from the calcium field of view. |
| `mouse_center_<animal>` | `2 x 7601` | Mouse mask centroid in behavior-video pixel coordinates, stored as `[x; y]`. |

Animals included in `data_aligned.mat`:

| Animal | Calcium shape | Neuron center shape | Mouse center shape |
| --- | ---: | ---: | ---: |
| `03_new` | `5148 x 7601` | `5148 x 2` | `2 x 7601` |
| `05_new` | `4609 x 7601` | `4609 x 2` | `2 x 7601` |
| `06_new` | `4775 x 7601` | `4775 x 2` | `2 x 7601` |
| `07` | `5129 x 7601` | `5129 x 2` | `2 x 7601` |
| `80` | `1736 x 7601` | `1736 x 2` | `2 x 7601` |

The first aligned timestamp is `04-Jan-25/16-27-06-479130`; the last aligned
timestamp is `04-Jan-25/16-37-41-576910`.

## `segment_mask.pickle`

`segment_mask.pickle` is a gzip-compressed Python pickle. It contains a list
with one entry per video frame. Each non-empty frame entry is a dictionary:

```python
{
    sam2_object_id: packed_binary_mask
}
```

The masks were stored with `numpy.packbits`. To restore a mask:

```python
import gzip
import pickle
import numpy as np

mask_path = "segment_mask.pickle"

with gzip.open(mask_path, "rb") as f:
    video_segments = pickle.load(f)

frame_index = 2918
obj_id = 1

packed_mask = video_segments[frame_index][obj_id]
mask = np.unpackbits(packed_mask).astype(np.int8).reshape((944, 968))
```

SAM2 object id mapping for this release:

| SAM2 object id | Animal |
| ---: | --- |
| `1` | `03_new` |
| `2` | `05_new` |
| `3` | `80` |
| `4` | `06_new` |
| `5` | `07` |

The first non-empty mask frame is frame `2916`.

## Minimal loading examples

Load the aligned MATLAB file in Python:

```python
import scipy.io as sio

mat_path = r"data_aligned.mat"
data = sio.loadmat(mat_path)

start_frame = int(data["start_frame"][0, 0])
end_frame = int(data["end_frame"][0, 0])
calcium_03 = data["calcium_03_new"]       # n_neurons x aligned_frames
mouse_center_03 = data["mouse_center_03_new"]  # 2 x aligned_frames, [x; y]
```

Load the aligned MATLAB file in MATLAB:

```matlab
data = load('data_aligned.mat');

start_frame = data.start_frame;
end_frame = data.end_frame;
calcium_03 = data.calcium_03_new;
mouse_center_03 = data.mouse_center_03_new;
```

