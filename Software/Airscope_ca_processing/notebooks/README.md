## Demo Data

Download the demo dataset (JPEG frame sequence) from Google Drive:

**[Download demo data (frames.zip)](https://drive.google.com/file/d/1P-wrqrqEMpr30H5Tcj7h8rFYt6ORBHGP/view?usp=drive_link)**

After downloading, extract the archive to a directory of your choice:

```bash
unzip frames.zip -d /your/data/directory/
# The extracted folder should be named "frames"
# e.g. /your/data/directory/frames/
```

Then update the `DATA_PATH` variable at the top of **each notebook** to point to the extracted `frames` folder:

```python
DATA_PATH = "/your/data/directory/frames"
```

> `OUT_PATH` is derived automatically as a sibling `Analysis/` folder and does not need to be changed.

## Stage Map

The notebooks are organized as independent, resumable stage reports:

| Notebook | Stage | Principal QC |
|---|---|---|
| 01 | raw loading + motion correction | raw/corrected frame montages, projections, before/after frames, Suite2p diagnostics |
| 02 | preprocessing | corrected vs preprocessed movie, vessel mask overlay, preprocessing output inventory |
| 03 | background removal | neuron-enhanced frames, residual maps, projection statistics |
| 04 | neural signal extraction | ROI center map, ROI area distribution, summed ROI mask |
| 05 | final export | filtered masks, vessel-exclusion result, ROI-boundary overlay |

## Final Output Structure

All notebooks use:

```python
OUT_PATH = DATA_PATH.replace("frames", "Analysis")
```

After a complete run, the analysis directory is expected to contain the following
stage outputs. Optional debug videos are only written when `save_debug_video`
is enabled in the composed Hydra config.
