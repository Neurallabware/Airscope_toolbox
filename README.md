# 🔬 Airscope   


### A *1-gram wireless mesoscope* for brain-wide, single-cell resolution imaging with truly unrestricted behavior

---

## 📢 Code Availability (IMPORTANT)

The code is hosted on GitHub at:  
**https://github.com/Neurallabware/Airscope_toolbox**  
The repository is currently **private** but will be made **publicly accessible upon publication**.

In the meantime, the full codebase can be accessed through Google Drive:  
**https://drive.google.com/drive/folders/1PhjPooL4QZGT8XVAjhhabJj6SmX-H24e**

---



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


## 📂 Contents
 

- [**📐 Structure**](./Structure)  
  System architecture, design documentation, and implementation details  (**CAD**)

<p align="center">
  <img src="./assets/2.jpg" alt="Airscope Demo" width="400"/>
</p> 


- [**⚙️ Firmware**](./Firmware)  
   Embedded control code in **C++** for wireless data transmission and device management

- [**🛠️ Hardware**](./Hardware)  

  | Component       | Description                         |
  |-----------------|-------------------------------------|
  | [Auxiliary](./Hardware/Auxiliary)       | Additional supporting components    |
  | [Extension PCB](./Hardware/Extension%20PCB)   | Optional expansion board             |
  | [LED FPC](./Hardware/LED%20FPC)         | Flexible PCB for LED control         |
  | [Main PCB](./Hardware/Main%20PCB)        | Core circuit board                   |
  | [Power FPC](./Hardware/Power%20FPC)      | Flexible PCB for power delivery      |


- [**💻 Software**](./Software)  
  
  Includes the following submodules:  

  | Component       | Description                         |
  |-----------------|-------------------------------------|
  | [Neuron_BERT](./Software/Neuron_BERT) | Classification network to determine winners and losers in tube tests |
  | [SAM2Mice](./Software/SAM2Mice)      | Zero-shot behavior segmentation in freely moving mice |


## 🙏 Credits & Usage

**Authors:** Yuanlong Zhang, Angran Li, Lekang Yuan and Mingrui Wang  

**Title of associated work:** *Deciphering cortex-wide neural dynamics of naturally behaving mice by 1-gram wireless mesoscope* (manuscript in preparation)  

We encourage people to **use, test, modify, and further develop this toolbox**.  
If you have any questions or suggestions, or find any bugs in the code, please **contact us** or **submit an issue**.  
If you use the code or data, please **cite us**!  

## Happy imaging! 🎥

