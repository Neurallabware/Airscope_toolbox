# Airscope Data Release

This directory contains supplementary datasets and visualization notebooks for
the Airscope release. Each experiment has its own README with the download
link, file descriptions, and minimal loading examples.

## Experiments

| Dataset | Local directory | Description |
| --- | --- | --- |
| Enriched habitat | [`experiments/enriched_habitat`](experiments/enriched_habitat) | Single-animal Airscope recording in a 5 m2 naturalistic habitat with synchronized multi-view behavior videos, detection boxes, and top-down arena-region annotations. |
| Multi-animal interaction | [`experiments/muiti_animal_interaction`](experiments/muiti_animal_interaction) | Multi-animal Airscope recording with behavior videos, SAM2 instance masks, aligned neural activity, cortical labels, neuron centers, and mouse mask centers. |

## Directory Layout

```text
Data_release/
├── MCU_timestamp/
├── calibration/
└── experiments/
    ├── enriched_habitat/
    │   ├── README.md
    │   └── visualize_data.ipynb
    └── muiti_animal_interaction/
        ├── README.md
        ├── visualize_data.ipynb
        └── outputs/
```

## Quick Start

1. Open the README for the dataset you want to use.
2. Download the data from the Google Drive link listed in that dataset README.
3. Place the downloaded files in the directory expected by the notebook, or
   update the notebook path configuration cell.
4. Run `visualize_data.ipynb` from top to bottom.

## Dataset Notes

### Enriched habitat

Main release files:

- `data_aligned.mat`: aligned calcium traces, behavior timestamps, cortical
  labels, and neuron centroid coordinates.
- `video/`: synchronized behavior videos from top-down and side-view cameras.
- `detection_box.json`: mouse detection bounding boxes for video frames.
- `CD00362AAK00005_0000.json`: LabelMe top-down arena-region annotation.

See [`experiments/enriched_habitat/README.md`](experiments/enriched_habitat/README.md)
for detailed variable descriptions and loading examples.

### Multi-animal interaction

Main release files:

- `data_aligned.mat`: aligned calcium traces and behavior-derived mouse centers
  for all animals.
- `behavior.mp4`: raw multi-animal behavior video.
- `behavior_with_mask.avi`: behavior video with SAM2 instance masks overlaid.
- `segment_mask.pickle`: packed SAM2 binary masks for each video frame.

See [`experiments/muiti_animal_interaction/README.md`](experiments/muiti_animal_interaction/README.md)
for detailed variable descriptions, mask unpacking, and loading examples.

