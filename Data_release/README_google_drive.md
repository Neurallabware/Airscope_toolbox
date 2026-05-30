# Airscope Google Drive Data Release

This Google Drive folder contains the code, released datasets, raw recordings,
model resources, and supporting files associated with the Airscope release.

Public Drive folder:
<https://drive.google.com/drive/folders/1z9ibX8Ob2NnCdjQDHI7Z4tdhdspn647B>

Associated manuscript:

> Deciphering cortex-wide neural dynamics of naturally behaving mice by a
> 1-gram wireless mesoscope

## Top-Level Contents

| Folder or file | Description |
| --- | --- |
| `code/` | Airscope toolbox source files, including hardware designs, firmware, acquisition software, processing pipelines, and documentation. |
| `enriched_habitat/` | Single-animal Airscope recording in a 5 m2 naturalistic habitat, with synchronized multi-view behaviour videos, neural-behaviour aligned data, detection boxes, and arena annotations. |
| `Multi_animal_interaction/` | Multi-animal Airscope recording with behaviour video, SAM2 instance masks, aligned neural activity, cortical labels, neuron centres, and mouse trajectories. |
| `MCU_timestamps/` | Microcontroller timestamp files used to validate Airscope frame rate and acquisition timing. |
| `raw_recordings/` | Raw Airscope recording files and related acquisition outputs. |
| `SAM2_Mice/` | SAM2Mice demo data, benchmark data, model checkpoints, and YOLO prompt-generation resources. |
| `README_google_drive.md` | This file. |

## Recommended Local Layout

After downloading the folders that you need, keep the same directory names when
possible:

```text
Airscope/
├── code/
├── enriched_habitat/
├── Multi_animal_interaction/
├── MCU_timestamps/
├── raw_recordings/
└── SAM2_Mice/
```

The notebooks in the code release assume that downloaded data are either placed
next to the notebook or that the path configuration cell is updated to point to
the downloaded folder.

## Released Datasets

### `enriched_habitat/`

This dataset contains a single-animal Airscope recording in a 5 m2 naturalistic
habitat.

Main files:

- `data_aligned.mat`: calcium traces aligned to behaviour timestamps, cortical
  labels, and neuron centroid coordinates.
- `video/`: synchronized behaviour videos from top-down and side-view cameras.
- `detection_box.json`: mouse detection bounding boxes for behaviour-video
  frames.
- `CD00362AAK00005_0000.json`: LabelMe annotation file for top-down arena
  regions.
- `README.md`: dataset-level variable descriptions and loading examples.

### `Multi_animal_interaction/`

This dataset contains a multi-animal Airscope recording with aligned neural and
behavioural measurements.

Main files:

- `data_aligned.mat`: aligned calcium traces, cortical labels, neuron centres,
  and behaviour-derived mouse centres.
- `behavior.mp4`: raw multi-animal behaviour video.
- `behavior_with_mask.avi`: behaviour video with SAM2 instance masks overlaid.
- `segment_mask.pickle`: gzip-compressed pickle file containing packed SAM2
  binary masks for each video frame.
- `README.md`: dataset-level variable descriptions, mask loading examples, and
  video metadata.

### `MCU_timestamps/`

This folder contains timestamp text files recorded by the Airscope
microcontroller. These files are used to estimate the actual imaging frame rate
and check acquisition timing stability.

### `raw_recordings/`

This folder contains raw acquisition outputs. These files are provided for
users who need to inspect or reprocess data from earlier stages of the Airscope
workflow.

## Code and Analysis

The `code/` folder contains the Airscope toolbox. The main modules are:

- hardware design files: mechanical CAD, optical design, and electronics;
- firmware for Airscope device control and image acquisition;
- host acquisition software for device discovery, preview, timestamping, and
  recording;
- calcium-imaging processing pipeline;
- SAM2Mice behavioural segmentation and tracking tools;
- Neuron-BERT neural decoding models.

For the latest repository documentation, see the root `README.md` in the code
folder.

## SAM2Mice Resources

The `SAM2_Mice/` folder contains data and resources for mouse segmentation and
tracking workflows, including benchmark datasets, demonstration data, model
checkpoints, and YOLO training or prompt-generation resources.

Use the SAM2Mice README and notebooks in the code release for installation and
workflow details.


## Licensing and Citation

Unless otherwise noted, released data, dataset documentation, figures, and
visualization notebooks are distributed under the Creative Commons Attribution
4.0 International License (CC-BY-4.0).

If you use Airscope hardware files, software, or released data, please cite:

```bibtex
@article{airscope2025,
  title   = {Deciphering cortex-wide neural dynamics of naturally
             behaving mice by a 1-gram wireless mesoscope},
  author  = {Zhang, Yuanlong and Li, Angran and Yuan, Lekang and
             Wang, Mingrui and Zhao, Weihao and Wang, Zhenbo and
             Zang, Boyang and Zhou, Yangxuan and Yu, Tao and Gao, Lin
             and Wu, Yu and Zhu, Rongkang and Tian, Mengyi and Li, Kun
             and Wu, Jiamin and Dai, Pu and Dai, Qionghai},
  journal = {Nature},
  year    = {2025},
  note    = {Manuscript in preparation}
}
```
