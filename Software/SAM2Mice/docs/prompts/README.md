# SAM2-Mice Manual Prompt Guide

SAM2Mice manual prompts are SAM2 input prompts, not training annotations. They are stored as LabelMe-compatible `.json` files next to extracted video frames. The segmentation code reads these JSON files when `prompt_source="manual"`.

You can create the JSON files with either:

- LabelMe desktop GUI
- SAM2-Mice browser prompt tool, exposed by `launch_annotator`

Both tools write one prompt JSON file per prompted frame, for example:

```text
frames/
├── 00000.jpg
├── 00000.json
├── 00001.jpg
├── 00002.jpg
└── ...
```

For bootstrapping inference, add prompts to the relevant batch folder, usually `batch_1`:

```text
bootstrap_frames/
├── batch_1/
│   ├── 00000.jpg
│   ├── 00000.json
│   └── ...
├── batch_2/
└── ...
```

## Option A: LabelMe Desktop GUI

Use LabelMe when you have a local desktop environment or X11 forwarding.

```bash
labelme /path/to/frames
```

For bootstrapping:

```bash
labelme /path/to/bootstrap_frames/batch_1
```

In LabelMe:

1. Open a frame.
2. Choose polygon mode and draw one polygon prompt around each mouse.
3. Set `label` to `mouse`.
4. Set `group_id` to the mouse ID. Use different `group_id` values for different mice, and keep the same `group_id` for the same mouse across prompted frames.
5. Save the prompt JSON. LabelMe writes `<frame_stem>.json` next to the image.

See the LabelMe usage example video: [`assets/labelme_usage_example.mp4`](../../assets/labelme_usage_example.mp4).

Example polygon prompt:

```json
{
  "shapes": [
    {
      "label": "mouse",
      "points": [
        [424.51, 1120.64],
        [311.61, 1212.58],
        [208.38, 1254.51],
        [239.03, 1204.51]
      ],
      "group_id": 1,
      "shape_type": "polygon",
      "flags": {}
    }
  ]
}
```

## Option B: SAM2-Mice Browser Prompt Tool

Use `launch_annotator` when running on a remote server or headless workstation. It starts a Gradio browser UI and saves LabelMe-compatible JSON files.

```python
from SAM2_Mice.utils import launch_annotator

launch_annotator(frames_dir="/path/to/frames", port=7860)
```

For bootstrapping:

```python
from SAM2_Mice.utils import launch_annotator

launch_annotator(frames_dir="/path/to/bootstrap_frames/batch_1", port=7860)
```

If the server is remote, forward the port from your local machine:

```bash
ssh -L 7860:localhost:7860 user@server
```

Then open:

```text
http://localhost:7860
```

The browser prompt tool can resume from existing JSON files in the same frame directory.

## Use Saved Manual Prompts

After saving prompt JSON files, run SAM2Mice with `prompt_source="manual"`.

Basic video predictor:

```python
predictor.run(
    video_path=None,
    frames_dir="/path/to/frames",
    prompt_source="manual",
    prompt_type="box",
    save_dir="results/manual",
    fps=20,
)
```

Bootstrapping predictor:

```python
predictor.run_bootstrapping(
    video_path=None,
    frames_dir="/path/to/bootstrap_frames",
    frame_interval=300,
    extract_frame=False,
    prompt_source="manual",
    prompt_type="point",
    save_dir="results/manual_bootstrap",
    fps=20,
)
```

## Choosing a Tool

Use LabelMe if you already have a graphical desktop or prefer the standard LabelMe workflow.

Use `launch_annotator` if you work on a remote GPU server, need browser access through SSH port forwarding, or want to resume and edit SAM2Mice manual prompts from a notebook.

The example notebooks expose this choice with:

```python
ANNOTATION_BACKEND = "launch_annotator"  # "labelme" or "launch_annotator"
```
