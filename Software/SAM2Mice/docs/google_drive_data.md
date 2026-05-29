# Google Drive Data

All shared SAM2Mice data is hosted in a single Google Drive folder:

**Root:** https://drive.google.com/drive/folders/1h3JZ6n-LhZRwAsLCKLH60XWioy7f3WPZ

The structure below was parsed from the public Google Drive folder listing with `gdown` on 2026-05-29. Large frame/label folders are summarized by pattern and count.

---

## Folder structure

```
SAM2_Mice/
|
├── benchmark_data/
|   ├── DLC_trained_output/
|   |   ├── plot-poses/
|   |   |   └── openfield_five_mouse/
|   |   ├── openfield_five_mouse.mp4
|   |   ├── openfield_five_mouseDLC_*_assemblies.pickle
|   |   ├── openfield_five_mouseDLC_*_el.csv
|   |   ├── openfield_five_mouseDLC_*_el.h5
|   |   ├── openfield_five_mouseDLC_*_el.pickle
|   |   ├── openfield_five_mouseDLC_*_el_filtered.csv
|   |   ├── openfield_five_mouseDLC_*_el_filtered.h5
|   |   ├── openfield_five_mouseDLC_*_el_filtered_id_labeled.mp4
|   |   ├── openfield_five_mouseDLC_*_full.pickle
|   |   └── openfield_five_mouseDLC_*_meta.pickle
|   ├── ground_truth_bb_labels/
|   |   └── 00000.txt ... 01000.txt                       # 1001 files
|   ├── openfield_five_mouse/
|   |   ├── 00000.jpg ... 00999.jpg                       # 1000 frame files
|   |   └── 00000.json                                    # prompt JSON
|   ├── openfield_five_mouse_seg/
|   |   ├── frames/                                       # 1000 jpg files
|   |   ├── prompts/                                      # 1 file
|   |   ├── SAM2Mice_label_vis/                           # 1001 jpg files
|   |   ├── SAM2Mice_yolo_labels/                         # 1001 txt files
|   |   ├── segment_masks.pickle
|   |   └── segmented_video.mp4
|   └── SuperAnimal_output/
|       ├── openfield_five_mouse_superanimal_*.h5
|       ├── openfield_five_mouse_superanimal_*_before_adapt.json
|       └── openfield_five_mouse_superanimal_*_labeled_before_adapt.mp4
|
├── ckpt/
|   ├── SAM2_Mice_base_plus.pt
|   └── yolo11l_openfield_five_mice.pt
|
├── demo_data/
|   ├── 1_habitat_ultra/                                  # Notebook 01
|   |   ├── habitat_ultra_seg/                            # listing returned 401
|   |   ├── prompts/
|   |   |   ├── 00000.json
|   |   |   └── 00009.json
|   |   └── habitat_ultra.mp4
|   ├── 2_auto_prompt/                                    # Notebook 01
|   |   └── openfield_three_mouse.mp4
|   ├── 3_bootstrapping/                                  # Notebook 02
|   |   ├── prompts/
|   |   |   └── 00000.json
|   |   └── openfield_five_mouse.mp4
|   ├── 4_dynamic_id/                                     # Notebook 03
|   |   └── three_mice_sequence.mp4
|   ├── 5_zero_shot_web_dataset/                          # Notebook 04
|   |   ├── 012609_A29_Block11_ovtBCfe1_t_505_6753.mp4
|   |   ├── 012609_A29_Block11_ovtBCfe1_t_505_6753_00000.jpg
|   |   ├── 012609_A29_Block11_ovtBCfe1_t_505_6753_00000.json
|   |   ├── 20080321162447_first5min.mp4
|   |   ├── 20080321162447_first5min_00000.jpg
|   |   ├── 20080321162447_first5min_00000.json
|   |   ├── OFT_5.mp4
|   |   ├── OFT_5_00100.jpg
|   |   └── OFT_5_00100.json
|   └── 6_example_dataset.zip                             # Notebook 05
|
└── YOLO_data/
    ├── images/
    |   ├── train/                                        # 1470 jpg files
    |   ├── train_vis/                                    # 1470 jpg files
    |   └── val/                                          # 164 jpg files
    └── labels/
        ├── train/                                        # 1470 txt files
        ├── val/                                          # 164 txt files
        ├── train.cache
        └── val.cache
```

---

## Demo download summary

| Notebook | Folder | Files downloaded |
|---|---|---|
| 01 | `demo_data/1_habitat_ultra/` | `habitat_ultra.mp4` |
| 01 | `demo_data/1_habitat_ultra/prompts/` | prompt JSONs |
| 01 | `demo_data/2_auto_prompt/` | `openfield_three_mouse.mp4` |
| 02 | `demo_data/3_bootstrapping/` | `openfield_five_mouse.mp4` |
| 02 | `demo_data/3_bootstrapping/prompts/` | prompt JSONs |
| 03 | `demo_data/4_dynamic_id/` | `three_mice_sequence.mp4` |
| 04 | `demo_data/5_zero_shot_web_dataset/` | 3 videos + 3 images + 3 prompt JSONs |
| 05 | `demo_data/` | `6_example_dataset.zip` |

---

## Manual download

To download all data at once, open the root folder link above and use **Download all** in Google Drive, or use `gdown` per-item as shown in each notebook's first cell.

Single file:
```bash
gdown https://drive.google.com/uc?id=<file_id> -O <destination>
```

Folder:
```bash
gdown --folder https://drive.google.com/drive/folders/<folder_id> -O <destination> --remaining-ok
```
