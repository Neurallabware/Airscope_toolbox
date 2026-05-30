<p align="center">
  <img src="./assets/1.jpg" alt="Airscope" width="40%"/>
</p>

<h1 align="center">Airscope</h1>

<p align="center">
  <b>A 1-gram wireless mesoscope for cortex-wide, single-cell-resolution imaging during unrestricted behaviour</b>
</p>

<p align="center">
  <a href="https://airscope.org/devkit/">Device Docs</a> ·
  <a href="https://airscope-docxs.readthedocs.io/en/latest/">Software Docs</a> ·
  <a href="https://drive.google.com/drive/folders/1z9ibX8Ob2NnCdjQDHI7Z4tdhdspn647B?usp=drive_link">Data Release</a>
</p>

---

Airscope is an open hardware and software platform for wireless mesoscopic calcium imaging in freely behaving mice. The device achieves a **6 mm field of view** at ~4 µm lateral resolution, streams or logs **1600 × 1200 px images at 10 Hz**, and weighs approximately **1 g**.

<p align="center">
  <img src="./assets/2.jpg" alt="Airscope examples" width="50%"/>
</p>

This repository accompanies the manuscript:

> **Deciphering cortex-wide neural dynamics of naturally behaving mice by a 1-gram wireless mesoscope**
> *Zhang et al., Nature, 2025* （Manuscript in preparation）

---

## Repository Structure

| Directory | Description |
|-----------|-------------|
| [`Structure/`](./Structure) | SolidWorks assemblies and part files — housing, baseplate, optical mount, PCB carrier, flexible LED board |
| [`Zemax/`](./Zemax) | Zemax optical design files for the aspheric module |
| [`Electronics/`](./Electronics) | KiCad projects for the main PCB and auxiliary boards |
| [`Firmware/`](./Firmware) | Embedded code for wireless control, camera acquisition, and device configuration |
| [`DAQ_software/`](./DAQ_software) | Python host software and Windows installer for device discovery, preview, and recording |
| [`Software/`](./Software) | Calcium-imaging processing, behavioural segmentation, and neural-decoding pipelines |
| [`Data_release/`](./Data_release) | Example datasets, visualization notebooks, and download instructions |

---

## Key Capabilities

- **Cortex-wide cellular imaging** — 6 mm FOV with single-cell spatial resolution across the dorsal cortex
- **Untethered operation** — wireless data transfer and on-board logging for freely moving, multi-animal, and aquatic paradigms
- **Integrated acquisition stack** — firmware + host software for discovery, preview, timestamping, and recording
- **Reproducible calcium processing** — motion correction → background removal → neuron segmentation → trace extraction
- **Behavioural analysis** — SAM2Mice for multi-animal tracking; Neuron-BERT for decoding social-interaction outcomes

---

## Software Modules

### Calcium-imaging processing · [`Software/Airscope_ca_processing`](./Software/Airscope_ca_processing)

End-to-end pipeline from raw frames to segmented ROI traces. Accepts image sequences, TIFF stacks, and MP4 input; configured with [Hydra](https://hydra.cc) for reproducible batch runs.

```bash
cd Software/Airscope_ca_processing
conda create -n PICO python=3.10 && conda activate PICO
pip install -r requirements.txt && pip install -e .
```

```bash
airscope-process \
  data_path=/path/to/session/frames \
  out_path=/path/to/session/analysis \
  rmbg.gpu_ids=0 rmbg.multi_gpu=false
```

Primary outputs: `seg_results_filtered.mat`, `infer_results_filtered.mat`, `cm_filtered.mat`

---

### SAM2Mice · [`Software/SAM2Mice`](./Software/SAM2Mice)

SAM 2–based mouse segmentation and tracking. Supports manual prompts, YOLOv11 auto-prompts, and bootstrapped inference for videos that exceed GPU memory.

```bash
cd Software/SAM2Mice
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[notebooks]" && python setup.py build_ext --inplace
```

Notebooks in [`Software/SAM2Mice/notebooks_SAM2-MICE`](./Software/SAM2Mice/notebooks_SAM2-MICE) cover basic segmentation, long-video bootstrapping, automatic tracking, and public-dataset examples.

---

### Neuron-BERT · [`Software/Neuron_BERT`](./Software/Neuron_BERT)

Transformer-based decoder for predicting social-interaction outcomes from dual-animal neural activity recordings.

---

### Host acquisition software · [`DAQ_software/`](./DAQ_software)

Dependency-free Python backend — runs without any additional packages:

```bash
cd DAQ_software/airscope_pybackend
python server.py
# UI available at http://127.0.0.1:8765/
```

A packaged Windows installer is also provided at [`DAQ_software/airscope.msi`](./DAQ_software/airscope.msi).

---

## Data Release

Example datasets are hosted on [Google Drive](https://drive.google.com/drive/folders/1z9ibX8Ob2NnCdjQDHI7Z4tdhdspn647B?usp=drive_link) and documented in [`Data_release/README.md`](./Data_release/README.md).
The Google Drive root README is maintained at [`Data_release/README_google_drive.md`](./Data_release/README_google_drive.md) and can be uploaded to Drive as `README_google_drive.md`.

| Dataset | Contents |
|---------|----------|
| Enriched habitat | Calcium traces, multi-view behaviour videos, detection boxes, arena annotations, cortical labels, neuron centroids |
| Multi-animal interaction | Calcium traces, SAM2 instance masks, mouse trajectories, cortical labels, neuron centres |
| MCU timestamp | Frame-rate validation files; use `calculate_frame_rate.ipynb` to verify |
| Optical characterisation | USAF 1951 target images and fluorescent grid slides |

Each dataset directory includes a `README.md` and a `visualize_data.ipynb` notebook.

---

## Requirements

| Component | Environment |
|-----------|-------------|
| Host acquisition software | Python ≥ 3.9, standard library only |
| Calcium processing | Linux, Python 3.10, CUDA-enabled PyTorch recommended |
| SAM2Mice | Python 3.11, PyTorch 2.6.0, CUDA 12.4 |
| Neuron-BERT | Python 3.x, PyTorch, NumPy, scikit-learn, pandas, matplotlib |

See the component-level `README.md` files for exact commands and configuration.

---

## Licensing

Airscope is distributed as a multi-licence repository:

| Material | Licence |
|----------|---------|
| Hardware design (`Structure/`, `Electronics/`, `Zemax/`) | CERN-OHL-S-2.0 |
| Airscope software (`DAQ_software/`, `Firmware/`, `Software/Neuron_BERT/`) | Apache-2.0 |
| Calcium-processing software (`Software/Airscope_ca_processing/`) | GPL-2.0-only |
| SAM2Mice (`Software/SAM2Mice/`) | Apache-2.0 |
| Documentation, figures, notebooks, released data | CC-BY-4.0 |

See [`LICENSE`](./LICENSE), [`LICENSES/`](./LICENSES), and [`NOTICE.md`](./NOTICE.md) for the authoritative licence map and third-party notices.

---

## Citation

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

## Documentation

- Device assembly, acquisition, and operation: <https://airscope.org/devkit/>
- Processing pipelines, notebooks, and examples: <https://airscope-docxs.readthedocs.io/en/latest/>

## Contact

Questions, bug reports, and suggestions can be submitted via the issue tracker or sent to the corresponding authors listed in the manuscript.
