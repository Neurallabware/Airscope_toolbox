# 🔬 Airscope   


### A *1-gram wireless mesoscope* for brain-wide, single-cell resolution imaging with truly unrestricted behavior


<p align="justify">

**Abstract:** Revealing how distributed cortical circuits coordinate behavior in freely moving animals remains a central challenge in systems neuroscience, constrained by limited fields of view, low throughput, and the tethering required by current miniaturized microscopes. We present **Airscope**, a wireless mesoscope delivering 4 µm lateral resolution across a 6 mm field of view at 10 Hz, weighing 1 gram. Integrating aspheric optics and a tailored wireless system, Airscope can record over 8,000 neurons per animal during freely behaving sessions, a 20-fold gain. Airscope enables previously inaccessible multi-animal imaging in room-scale enriched arenas, swimming assays, and multi-organ implantation. Initial investigations reveal dorsal cortex representations of social hierarchy supporting reliable decoding of competition outcomes, a reward discovery state during maze exploration, and spinal-to-cortex causal flow during fear conditioning.

</p>



<p align="center">
  <img src="./assets/1.jpg" alt="Airscope Demo" width="400"/>
</p> 


## ✨ Key Features

### 🔬 Compact yet powerful
- 6 mm × 6 mm field of view  
- 4 µm single-cell resolution  
- < 1 cm total height, ~1 gram weight  
- 1600 × 1200 pixels @ 10 fps 

### 📡 Fully wireless
- Real-time data streaming via Wi-Fi  
- On-board high-speed data logging  
- Supports free behavior in complex environments with multiple animals  

### 🎛️ Easy to use
- Intuitive data acquisition software  
- Zero-shot behavior segmentation with **SAM2Mice**  


## Wiki

A comprehensive wiki is maintained by the Airscope team.

To build and use the Airscope device and the associated DAQ software, please visit:  
https://airscope.org/devkit/

For data processing pipelines and example notebooks, please visit:  
https://airscope-docxs.readthedocs.io/en/latest/

## ✨ Key Features

### 🔬 Compact yet powerful
- 6 mm × 6 mm field of view  
- 4 µm single-cell resolution  
- < 1 cm total height and ~1 g weight  
- 1600 × 1200 pixels at 10 fps  

### 📡 Fully wireless
- Real-time data streaming via Wi-Fi  
- On-board high-speed data logging  
- Supports free behavior in complex environments with multiple animals  

### 🎛️ Easy to use
- Intuitive data acquisition software  
- Modular data processing pipelines  
- Zero-shot behavior segmentation with **SAM2Mice**  

## 📂 Contents

- [**📐 Structure**](./Structure)  
  Mechanical structure design of the Airscope.

- [**⚙️ Zemax**](./Zemax)  
  Zemax design files for the aspherical lens module.

- [**🛠️ Hardware**](./Hardware)  
  Electronic designs for the miniscope and auxiliary supporting PCBs.

- [**⚙️ Firmware**](./Firmware)  
  Embedded control code running on the Airscope device for wireless image acquisition.

- [**🎛️ DAQ Software**](./Software)  
  PC-side DAQ software for wireless monitoring, control, preview, and recording.

- [**💻 Data Processing Software**](./Software)  
  Data processing pipelines, including the following submodules:

  | Component | Description |
  |----------|-------------|
  | [Airscope_ca_processing](./Software/Airscope_ca_processing) | Calcium imaging processing pipeline for Airscope recordings |
  | [Neuron_BERT](./Software/Neuron_BERT) | Classification network for identifying winners and losers in tube tests |
  | [SAM2Mice](./Software/SAM2Mice) | Zero-shot behavior segmentation for freely moving mice |

- [**📊 Data Release**](./Data_release)  
  Supplementary datasets and visualization notebooks.

## 🙏 Citation

We encourage the community to **use, test, modify, and further develop this toolbox**.

If you have any questions or suggestions, or if you find any bugs in the code, please **contact us** or **submit an issue**.

If you use the code or data, please cite us:

```bibtex
@article{airscope2025,
  title   = {Deciphering cortex-wide neural dynamics of naturally
             behaving mice by 1-gram wireless mesoscope},
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

## Happy imaging! 🎥

## Happy imaging! 🎥

