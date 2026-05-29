# Supplementary data: `5m2 natural habitat`


Release data is available in the public Airscope data release [here][release-data].

[release-data]: https://drive.google.com/drive/folders/1z9ibX8Ob2NnCdjQDHI7Z4tdhdspn647B?usp=drive_link


------------------------------------------------------------------------

## Contents of `data_aligned.mat`

-   **`timestamps`**\
    *Type:* cell array of strings (`1 × T`)\
    *Description:* Timestamps of calcium signals aligned with behavior capture.\
    *Format:* `dd-MMM-yy/HH-MM-SS-fff` (e.g.,
    `22-Jan-25/20-57-13-123`).
    
-   **`calcium_traces`**\
    *Type:* double matrix (`num_neurons × T`)\
    *Description:* Interpolated ΔF/F calcium traces, aligned to the behavior timestamps. Each column corresponds to one behavioral frame.

-   **`neuron_cortical_labels`**\
    *Type:* cell array of strings (`1 × num_neurons`)\
    *Description:* Cortical region identity of each neuron, derived from cortical alignment.

-   **`neuron_centroid_coordinates`**\
    *Type:* double matrix (`num_neurons × 2`)\
    *Description:* XY centroid coordinates of each segmented neuron in the calcium imaging field of view.\
    *Units:* pixels of Airscope calcium video.\
    *Columns:* Column 1 = X coordinate, Column 2 = Y coordinate.

------------------------------------------------------------------------
## Associated Behavioral Videos 

To complement the neural recordings, synchronized multi-view behavioral videos are provided in the `/video` directory:

-   **`CD00362AAK00005.mp4`** — Overhead (top-down) camera view of the naturalistic arena.  
-   **`AL00099B001.mp4, AL00099B003.mp4, BD03405A003.mp4, BD03405A005.mp4`** — Four lateral perspectives captured by side-mounted cameras, providing complementary information on body posture and social interaction.  

**Video specifications:** Resolution = 4096 × 3000 pixels; Frame rate = 30 fps; Format = MP4.  
All videos are time-locked to the calcium imaging data.

## Associated Annotation Files

-   **`detection_box.json`**\
    Mouse detection bounding boxes for behavior-video frames.

-   **`CD00362AAK00005_0000.json`**\
    LabelMe annotation file for the top-down camera arena regions.


## Example Usage

**MATLAB**

``` matlab
data = load('data_aligned.mat');
C = data.calcium_traces;                                
timestamps = data.time_stamps; 
regions = data.neuron_cortical_labels; 
coords = data.neuron_centroid_coordinates; 
```

**Python**

``` python
import scipy.io as sio
import os

DATA_DIR = "/path/to/natural_habitat"

ALIGNED_DATA_PATH = os.path.join(DATA_DIR, "data_aligned.mat")
BOX_PATH  = os.path.join(DATA_DIR, "detection_box.json")
LABEL_ME_REGION_PATH = os.path.join(DATA_DIR, "CD00362AAK00005_0000.json")

data = sio.loadmat(ALIGNED_DATA_PATH)

C = data["calcium_traces"]                               
timestamps = data["time_stamps"] 
regions = data["neuron_cortical_labels"] 
coords = data["neuron_centroid_coordinates"] 
```

------------------------------------------------------------------------







