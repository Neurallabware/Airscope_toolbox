"""
Interactive frame annotator for SAM2-Mice.

Launches a Gradio web UI so you can label frames on a headless server.
Access it locally via SSH port forwarding:

    ssh -L 7860:localhost:7860 user@server

Then open http://localhost:7860 in your browser.

Output format
-------------
One ``<stem>.json`` per frame in standard labelme format. ``shapes`` may
contain a mix of two ``shape_type`` values:

  * ``"polygon"``    — ``points`` = ``[[x1, y1], [x2, y2], ...]`` (vertices or prompt points)
  * ``"rectangle"``  — ``points`` = ``[[x1, y1], [x2, y2]]`` (two opposite corners)

``group_id`` carries the mouse / object id; ``label`` is ``mouse<id>``.

Re-opening the same ``frames_dir`` auto-loads everything previously saved so
you can resume work without re-clicking.

Usage
-----
    from SAM2_Mice.utils.annotator import launch_annotator
    launch_annotator(frames_dir="/path/to/batch_1", port=7860)

Or from the command line:
    python -m SAM2_Mice.utils.annotator --frames_dir /path/to/batch_1
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError:
    gr = None  # deferred; error raised inside launch_annotator


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
_PALETTE = [
    (220, 60,  60),   # red
    (60,  200, 60),   # green
    (60,  100, 220),  # blue
    (220, 180, 0),    # yellow
    (0,   200, 200),  # cyan
    (200, 0,   200),  # magenta
    (255, 140, 0),    # orange
    (120, 80,  220),  # purple
    (180, 120, 60),   # brown
    (80,  180, 120),  # teal
]


def _color(obj_id: int) -> Tuple[int, int, int]:
    """Return the display color assigned to an object ID."""
    return _PALETTE[(int(obj_id) - 1) % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Labelme JSON I/O
# ---------------------------------------------------------------------------

def _encode_image_b64(image_path: str) -> str:
    """Read an image file and encode it as base64 for LabelMe JSON."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_hw(image_path: str) -> Tuple[int, int]:
    """Return image height and width, or zeros if the image cannot be read."""
    img = cv2.imread(image_path)
    return (img.shape[:2] if img is not None else (0, 0))


def write_labelme_json(image_path: str, frame_entries: List[dict], out_path: str) -> None:
    """Write all annotations for one frame to a standard labelme JSON.

    Each entry in ``frame_entries`` is:
        {
            "obj_id": int,
            "points": [[x,y], ...],
            "polygons": [[[x,y], ...], ...],
            "boxes": [[x1,y1,x2,y2], ...],
        }

    Point prompts are saved as one ``shape_type="polygon"`` shape per object
    with ``flags.sam2mice_prompt == "point"``. Each drawn polygon becomes one
    ``shape_type="polygon"`` shape; each box becomes one
    ``shape_type="rectangle"`` shape.
    """
    h, w = _image_hw(image_path)
    shapes: List[dict] = []
    for ann in frame_entries:
        oid = int(ann["obj_id"])
        label = f"mouse{oid}"
        points = ann.get("points", []) or []
        if points:
            shapes.append({
                "label": label,
                "group_id": oid,
                "points": [[float(pt[0]), float(pt[1])] for pt in points],
                "shape_type": "polygon",
                "flags": {"sam2mice_prompt": "point"},
            })
        for poly in ann.get("polygons", []) or []:
            if len(poly) < 3:
                continue
            shapes.append({
                "label": label,
                "group_id": oid,
                "points": [[float(p[0]), float(p[1])] for p in poly],
                "shape_type": "polygon",
                "flags": {},
            })
        for box in ann.get("boxes", []) or []:
            x1, y1, x2, y2 = [float(v) for v in box]
            shapes.append({
                "label": label,
                "group_id": oid,
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle",
                "flags": {},
            })

    data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageHeight": h,
        "imageWidth": w,
        "imageData": _encode_image_b64(image_path),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def _load_labelme_json(json_path: Path) -> List[dict]:
    """Parse one frame's labelme JSON into the in-memory annotation list."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    by_obj: Dict[int, dict] = {}
    h = int(data.get("imageHeight") or 0)
    w = int(data.get("imageWidth") or 0)
    for shape in data.get("shapes", []):
        oid = shape.get("group_id")
        if oid is None:
            lbl = shape.get("label", "")
            if lbl.startswith("mouse"):
                try:
                    oid = int(lbl[5:])
                except ValueError:
                    continue
            else:
                continue
        oid = int(oid)
        entry = by_obj.setdefault(
            oid, {"obj_id": oid, "points": [], "polygons": [], "boxes": [], "height": h, "width": w}
        )
        st = shape.get("shape_type")
        pts = shape.get("points", [])
        flags = shape.get("flags") or {}
        if st == "point" and len(pts) >= 1:
            entry["points"].append([float(pts[0][0]), float(pts[0][1])])
        elif st == "polygon" and flags.get("sam2mice_prompt") == "point":
            entry["points"].extend([[float(p[0]), float(p[1])] for p in pts])
        elif st == "polygon" and len(pts) >= 3:
            entry["polygons"].append([[float(p[0]), float(p[1])] for p in pts])
        elif st == "rectangle" and len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x1, x2 = sorted([float(x1), float(x2)])
            y1, y2 = sorted([float(y1), float(y2)])
            entry["boxes"].append([x1, y1, x2, y2])

    return list(by_obj.values())


def load_existing_annotations(frames_dir: Path, frame_names: List[str]) -> Dict[str, List[dict]]:
    """Scan ``frames_dir`` and rebuild the per-frame annotation state."""
    out: Dict[str, List[dict]] = {}
    stem_to_name = {Path(n).stem: n for n in frame_names}
    for json_path in frames_dir.glob("*.json"):
        stem = json_path.stem
        fname = stem_to_name.get(stem)
        if fname is None:
            continue
        loaded = _load_labelme_json(json_path)
        if loaded:
            out[fname] = loaded
    return out


# ---------------------------------------------------------------------------
# Overlay renderer
# ---------------------------------------------------------------------------

def _render(
    image_path: str,
    annotations: dict,
    frame_name: str,
    pending_polygon: Optional[dict] = None,
    pending_box_start: Optional[Tuple[int, int]] = None,
    pending_box_oid: Optional[int] = None,
) -> np.ndarray:
    """Draw all annotations for ``frame_name`` on top of the image."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for ann in annotations.get(frame_name, []):
        oid = int(ann["obj_id"])
        r, g, b = _color(oid)

        # Points
        for idx, pt in enumerate(ann.get("points", []) or [], start=1):
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img, (x, y), 5, (r, g, b), -1)
            cv2.circle(img, (x, y), 7, (255, 255, 255), 2)
            cv2.putText(img, str(idx), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (r, g, b), 1)

        # Polygons
        for poly in ann.get("polygons", []) or []:
            if len(poly) < 3:
                continue
            pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in poly], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (r, g, b))
            img = cv2.addWeighted(img, 0.78, overlay, 0.22, 0)
            cv2.polylines(img, [pts], isClosed=True, color=(r, g, b), thickness=2)

            ax, ay = int(pts[0][0]), int(pts[0][1])
            label = f"mouse{oid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (ax, max(0, ay - th - 8)), (ax + tw + 8, ay), (r, g, b), -1)
            cv2.putText(img, label, (ax + 4, max(th, ay - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Boxes
        for box in ann.get("boxes", []) or []:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (r, g, b), -1)
            img = cv2.addWeighted(img, 0.78, overlay, 0.22, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), (r, g, b), 2)

            label = f"mouse{oid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (r, g, b), -1)
            cv2.putText(img, label, (x1 + 4, max(th, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Pending polygon-in-progress
    if pending_polygon is not None and pending_polygon.get("frame") == frame_name:
        oid = int(pending_polygon.get("obj_id", 1))
        r, g, b = _color(oid)
        verts = pending_polygon.get("verts", [])
        pts = [(int(round(v[0])), int(round(v[1]))) for v in verts]
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (x, y), 5, (r, g, b), -1)
            cv2.circle(img, (x, y), 6, (255, 255, 255), 2)
            cv2.putText(img, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (r, g, b), 1)
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (r, g, b), 2)
        if len(pts) >= 2:
            cv2.line(img, pts[-1], pts[0], (r, g, b), 1, lineType=cv2.LINE_AA)
        if pts:
            x, y = pts[-1]
            cv2.putText(img, "click to add vertex · Close to finish",
                        (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (r, g, b), 1)

    # Pending box first corner indicator
    if pending_box_start is not None:
        r, g, b = _color(pending_box_oid or 1)
        x, y = pending_box_start
        cv2.drawMarker(img, (x, y), (r, g, b), markerType=cv2.MARKER_CROSS,
                       markerSize=18, thickness=2)
        cv2.putText(img, "click 2nd corner", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (r, g, b), 2)

    return img


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def _id_badge(oid: int) -> str:
    """Build the colored HTML badge shown for the active object ID."""
    r, g, b = _color(int(oid))
    return (
        f'<div style="background:rgb({r},{g},{b});color:white;'
        f'border-radius:8px;padding:6px 14px;display:inline-block;'
        f'font-weight:600;font-size:1.05em;letter-spacing:0.02em;">'
        f'mouse{int(oid)}</div>'
    )


def _build_app(frames_dir: str):
    """Construct and return the Gradio app for one frame directory."""
    if gr is None:
        raise ImportError("pip install gradio>=4.0")

    frame_dir = Path(frames_dir)
    valid_ext = {".jpg", ".jpeg", ".png"}
    frame_files = sorted(
        [p for p in frame_dir.iterdir() if p.suffix.lower() in valid_ext],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not frame_files:
        raise FileNotFoundError(f"No image files found in {frames_dir}")

    frame_names = [p.name for p in frame_files]
    frame_paths = {p.name: str(p) for p in frame_files}

    initial_annotations = load_existing_annotations(frame_dir, frame_names)
    n_loaded_frames = len(initial_annotations)
    n_loaded_shapes = sum(
        len(a.get("points", [])) + len(a.get("polygons", [])) + len(a.get("boxes", []))
        for anns in initial_annotations.values() for a in anns
    )

    def _json_path(fname: str) -> Path:
        """Return the LabelMe JSON path for a frame file name."""
        return frame_dir / f"{Path(fname).stem}.json"

    def _saved_marker(fname: str) -> str:
        """Return the selector marker that indicates whether JSON exists on disk."""
        return "✓" if _json_path(fname).exists() else " "

    def _ann_summary(frame_name, annotations) -> str:
        """Format the annotation summary shown beside the canvas."""
        anns = annotations.get(frame_name, [])
        saved = "✅ saved on disk" if _json_path(frame_name).exists() else "⬜ unsaved"
        if not anns:
            body = "No annotations yet."
        else:
            lines = []
            for a in sorted(anns, key=lambda x: int(x["obj_id"])):
                npt = len(a.get("points", []) or [])
                npg = len(a.get("polygons", []) or [])
                nbx = len(a.get("boxes", []) or [])
                lines.append(
                    f"  • mouse{int(a['obj_id'])}:  {npt} point(s) · {nbx} box · {npg} polygon(s)"
                )
            body = "\n".join(lines)
        return f"{saved}\n{body}"

    def _frame_choice(fname: str) -> str:
        """Format one frame selector choice with saved status and index."""
        idx = frame_names.index(fname) + 1
        return f"[{_saved_marker(fname)}] {idx:>4d}  {fname}"

    def _frame_choices(annotations) -> List[str]:
        """Return all frame selector choices for the current state."""
        return [_frame_choice(n) for n in frame_names]

    def _name_from_choice(choice: str) -> str:
        """Extract the frame file name from a selector choice string."""
        return choice.split()[-1]

    def _frame_counter_md(fname):
        """Format the current frame counter markdown."""
        idx = frame_names.index(fname) + 1
        return f"**Frame {idx} / {len(frame_names)}**   ·   `{fname}`"

    def _ensure_entry(annotations, frame_name, oid, h, w):
        """Ensure the annotation list contains an entry for the object ID."""
        anns = annotations.setdefault(frame_name, [])
        entry = next((a for a in anns if int(a["obj_id"]) == oid), None)
        if entry is None:
            entry = {"obj_id": oid, "points": [], "polygons": [], "boxes": [], "height": h, "width": w}
            anns.append(entry)
        entry.setdefault("points", [])
        entry.setdefault("polygons", [])
        entry.setdefault("boxes", [])
        return entry

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def on_canvas_click(frame_name, obj_id, mode, pending_box, annotations, evt: gr.SelectData):
        """Single entry point for clicks; behaviour depends on Mode toggle."""
        x, y = int(evt.index[0]), int(evt.index[1])
        oid = int(obj_id)
        h, w = _image_hw(frame_paths[frame_name])

        status_msg = ""
        if mode == "Point":
            entry = _ensure_entry(annotations, frame_name, oid, h, w)
            entry["points"].append([float(x), float(y)])
            status_msg = f"Added point ({x}, {y}) to mouse{oid}."
            pending_box = None
        else:  # Box: two-click
            if (pending_box is None
                or pending_box.get("frame") != frame_name
                or int(pending_box.get("obj_id", -1)) != oid):
                pending_box = {"frame": frame_name, "obj_id": oid, "x": x, "y": y}
                status_msg = f"Box corner 1 at ({x}, {y}) for mouse{oid}. Click second corner."
            else:
                x1, y1 = pending_box["x"], pending_box["y"]
                x2, y2 = x, y
                if x1 == x2 or y1 == y2:
                    status_msg = "⚠️  Box has zero area; pick a different second corner."
                else:
                    bx1, bx2 = sorted([x1, x2])
                    by1, by2 = sorted([y1, y2])
                    entry = _ensure_entry(annotations, frame_name, oid, h, w)
                    entry["boxes"].append([bx1, by1, bx2, by2])
                    status_msg = f"Added box ({bx1},{by1})→({bx2},{by2}) to mouse{oid}."
                    pending_box = None

        pending_xy = (pending_box["x"], pending_box["y"]) if pending_box else None
        pending_oid = pending_box["obj_id"] if pending_box else None
        img = _render(frame_paths[frame_name], annotations, frame_name,
                      pending_box_start=pending_xy, pending_box_oid=pending_oid)
        return (img, _ann_summary(frame_name, annotations) + f"\n{status_msg}",
                annotations, pending_box)

    def on_undo(frame_name, obj_id, annotations, pending_box):
        """Undo: clear an unfinished box corner; else drop the last shape."""
        oid = int(obj_id)
        msg = "Nothing to undo."
        if (pending_box is not None
            and pending_box.get("frame") == frame_name
            and int(pending_box.get("obj_id", -1)) == oid):
            msg = f"Removed pending box corner [{pending_box['x']}, {pending_box['y']}] from mouse{oid}."
            pending_box = None
        else:
            anns = annotations.get(frame_name, [])
            entry = next((a for a in anns if int(a["obj_id"]) == oid), None)
            if entry:
                if entry.get("points"):
                    pt = entry["points"].pop()
                    msg = f"Removed last point {pt} from mouse{oid}."
                elif entry.get("boxes"):
                    box = entry["boxes"].pop()
                    msg = f"Removed last box {box} from mouse{oid}."
                elif entry.get("polygons"):
                    poly = entry["polygons"].pop()
                    msg = f"Removed last polygon ({len(poly)} verts) from mouse{oid}."
                if not entry.get("points") and not entry.get("boxes") and not entry.get("polygons"):
                    annotations[frame_name] = [a for a in anns if int(a["obj_id"]) != oid]
        img = _render(frame_paths[frame_name], annotations, frame_name)
        return (img, _ann_summary(frame_name, annotations) + f"\n{msg}",
                annotations, pending_box)

    def on_clear_obj(frame_name, obj_id, annotations, pending_box):
        """Clear all prompts for the selected object in the current frame."""
        oid = int(obj_id)
        anns = annotations.get(frame_name, [])
        annotations[frame_name] = [a for a in anns if int(a["obj_id"]) != oid]
        if pending_box and int(pending_box.get("obj_id", -1)) == oid:
            pending_box = None
        img = _render(frame_paths[frame_name], annotations, frame_name)
        return (img, _ann_summary(frame_name, annotations) + f"\nCleared mouse{oid}.",
                annotations, pending_box)

    def on_clear_frame(frame_name, annotations, pending_box):
        """Clear all prompts from the current frame."""
        annotations[frame_name] = []
        pending_box = None
        img = _render(frame_paths[frame_name], annotations, frame_name)
        return (img, _ann_summary(frame_name, annotations) + "\nCleared frame.",
                annotations, pending_box)

    def _do_save(frame_name, annotations) -> str:
        """Write or remove the LabelMe JSON for one frame based on prompt state."""
        anns = annotations.get(frame_name, [])
        anns_with_shapes = [
            a for a in anns
            if (a.get("points") or []) or (a.get("polygons") or []) or (a.get("boxes") or [])
        ]
        out_path = _json_path(frame_name)
        if not anns_with_shapes:
            if out_path.exists():
                out_path.unlink()
                return f"🗑  Removed {out_path.name} (no annotations left)."
            return "⚠️  Nothing to save — add annotations first."
        write_labelme_json(frame_paths[frame_name], anns_with_shapes, str(out_path))
        return f"💾  Saved {out_path.name}"

    def on_save(frame_name, annotations):
        """Save prompts for the current frame and refresh selector state."""
        msg = _do_save(frame_name, annotations)
        return (
            _ann_summary(frame_name, annotations) + f"\n{msg}",
            gr.update(choices=_frame_choices(annotations), value=_frame_choice(frame_name)),
        )

    def on_save_all(frame_name, annotations):
        """Save prompts for every frame that has in-memory prompt state."""
        results = []
        for fname in frame_names:
            if annotations.get(fname):
                msg = _do_save(fname, annotations)
                results.append(f"[{fname}] {msg}")
        body = "\n".join(results) if results else "⚠️  No annotations in any frame."
        return (
            _ann_summary(frame_name, annotations) + "\n" + body,
            gr.update(choices=_frame_choices(annotations), value=_frame_choice(frame_name)),
        )

    def on_prev(frame_choice, annotations):
        """Move the frame selector to the previous frame."""
        fname = _name_from_choice(frame_choice)
        idx = frame_names.index(fname)
        new = frame_names[max(0, idx - 1)]
        return gr.update(value=_frame_choice(new))

    def on_next(frame_choice, annotations):
        """Move the frame selector to the next frame."""
        fname = _name_from_choice(frame_choice)
        idx = frame_names.index(fname)
        new = frame_names[min(len(frame_names) - 1, idx + 1)]
        return gr.update(value=_frame_choice(new))

    def on_frame_selector_change(frame_choice, annotations):
        """Render the newly selected frame and reset pending box state."""
        fname = _name_from_choice(frame_choice)
        img = _render(frame_paths[fname], annotations, fname)
        return img, _ann_summary(fname, annotations), _frame_counter_md(fname), None, fname

    def on_jump_unannotated(frame_choice, annotations):
        """Jump to the next frame without saved or in-memory prompts."""
        fname = _name_from_choice(frame_choice)
        idx = frame_names.index(fname)
        for i in range(idx + 1, len(frame_names)):
            n = frame_names[i]
            if not _json_path(n).exists() and not annotations.get(n):
                return gr.update(value=_frame_choice(n))
        for i in range(0, idx):
            n = frame_names[i]
            if not _json_path(n).exists() and not annotations.get(n):
                return gr.update(value=_frame_choice(n))
        return gr.update()

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------
    css = """
    .gradio-container { max-width: 1600px !important; }
    #title { text-align: center; margin: 0 0 4px 0; font-size: 1.6em; }
    #subtitle { text-align: center; color: #888; font-size: 0.88em; margin-bottom: 14px; }
    .panel-card {
        border: 1px solid #e2e6ea !important;
        border-radius: 10px !important;
        padding: 12px !important;
        background: #fafbfc !important;
        margin-bottom: 8px !important;
    }
    .panel-card h3, .panel-card .markdown {
        margin-top: 0 !important;
        margin-bottom: 8px !important;
        font-size: 0.95em !important;
    }
    .save-btn { background: #2563eb !important; color: white !important; font-weight: 600; }
    .save-all-btn { background: #1e40af !important; color: white !important; }
    .warn-btn { background: #ef4444 !important; color: white !important; }
    .close-btn { background: #10b981 !important; color: white !important; }
    .mode-radio label { font-weight: 600; }
    #canvas-col { background: #1f2937; border-radius: 12px; padding: 8px; }
    .frame-selector select { font-family: ui-monospace, SFMono-Regular, monospace !important; }
    """

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------
    with gr.Blocks(title="SAM2-Mice Annotator") as demo:
        demo._sam2_css = css
        annotations_state = gr.State(initial_annotations)
        pending_box_state = gr.State(None)      # {"frame", "obj_id", "x", "y"} | None
        current_frame_state = gr.State(frame_names[0])

        gr.HTML(
            '<div id="title">🐭 SAM2-Mice Frame Annotator (labelme format)</div>'
            '<div id="subtitle">'
            'Switch <b>Mode</b>, then click the image. '
            'Point mode saves one prompt point per click. '
            'Box mode needs two clicks (two opposite corners).'
            '</div>'
        )

        if n_loaded_frames > 0:
            gr.Markdown(
                f"✨ **Auto-loaded {n_loaded_shapes} shapes across {n_loaded_frames} frame(s)** "
                f"from existing labelme JSON files in `{frame_dir}`."
            )

        with gr.Row():
            # ── Left control panel ──────────────────────────────────────
            with gr.Column(scale=1, min_width=300):

                with gr.Group(elem_classes="panel-card"):
                    gr.Markdown("### 🎞 Frame")
                    frame_selector = gr.Dropdown(
                        choices=_frame_choices(initial_annotations),
                        value=_frame_choice(frame_names[0]),
                        label="Pick a frame  ([✓] = saved)",
                        interactive=True,
                        elem_classes="frame-selector",
                    )
                    with gr.Row():
                        btn_prev = gr.Button("◀ Prev", size="sm")
                        btn_next = gr.Button("Next ▶", size="sm")
                    btn_jump = gr.Button("⏭  Next unannotated", size="sm")
                    frame_counter = gr.Markdown(_frame_counter_md(frame_names[0]))

                with gr.Group(elem_classes="panel-card"):
                    gr.Markdown("### 🐭 Mouse ID")
                    obj_id = gr.Slider(
                        minimum=1, maximum=10, value=1, step=1,
                        label="Mouse ID (obj_id)", interactive=True,
                    )
                    id_preview = gr.HTML(_id_badge(1))

                with gr.Group(elem_classes="panel-card"):
                    gr.Markdown("### 🖱 Click mode")
                    mode_radio = gr.Radio(
                        choices=["Point", "Box"], value="Point",
                        label="What does a click do?", elem_classes="mode-radio",
                    )
                    gr.Markdown(
                        "<small>"
                        "• <b>Point</b>: each click saves one positive prompt point immediately.<br>"
                        "• <b>Box</b>: first click = corner 1, second click = corner 2."
                        "</small>"
                    )

                with gr.Group(elem_classes="panel-card"):
                    gr.Markdown("### 🛠 Edit")
                    with gr.Row():
                        btn_undo = gr.Button("↩ Undo last", size="sm")
                        btn_clear_obj = gr.Button("Clear mouse", size="sm", elem_classes="warn-btn")
                    btn_clear_frame = gr.Button("Clear entire frame", size="sm", elem_classes="warn-btn")

                with gr.Group(elem_classes="panel-card"):
                    gr.Markdown("### 💾 Save")
                    btn_save = gr.Button("Save this frame", elem_classes="save-btn")
                    btn_save_all = gr.Button("Save ALL frames", elem_classes="save-all-btn")

                status = gr.Textbox(
                    label="Status / annotations",
                    lines=8, interactive=False,
                    value=_ann_summary(frame_names[0], initial_annotations),
                )

            # ── Canvas ──────────────────────────────────────────────────
            with gr.Column(scale=3, elem_id="canvas-col"):
                canvas = gr.Image(
                    label=None, show_label=False,
                    interactive=True, height=720,
                    value=_render(frame_paths[frame_names[0]], initial_annotations, frame_names[0]),
                )

        # ── Events ──────────────────────────────────────────────────────

        frame_selector.change(
            on_frame_selector_change,
            inputs=[frame_selector, annotations_state],
            outputs=[canvas, status, frame_counter,
                     pending_box_state, current_frame_state],
        )

        obj_id.change(
            lambda oid: _id_badge(oid),
            inputs=[obj_id], outputs=[id_preview],
        )

        btn_prev.click(on_prev, inputs=[frame_selector, annotations_state], outputs=[frame_selector])
        btn_next.click(on_next, inputs=[frame_selector, annotations_state], outputs=[frame_selector])
        btn_jump.click(on_jump_unannotated, inputs=[frame_selector, annotations_state], outputs=[frame_selector])

        canvas.select(
            on_canvas_click,
            inputs=[current_frame_state, obj_id, mode_radio,
                    pending_box_state, annotations_state],
            outputs=[canvas, status, annotations_state,
                     pending_box_state],
        )

        btn_undo.click(
            on_undo,
            inputs=[current_frame_state, obj_id, annotations_state, pending_box_state],
            outputs=[canvas, status, annotations_state,
                     pending_box_state],
        )
        btn_clear_obj.click(
            on_clear_obj,
            inputs=[current_frame_state, obj_id, annotations_state, pending_box_state],
            outputs=[canvas, status, annotations_state,
                     pending_box_state],
        )
        btn_clear_frame.click(
            on_clear_frame,
            inputs=[current_frame_state, annotations_state, pending_box_state],
            outputs=[canvas, status, annotations_state,
                     pending_box_state],
        )

        btn_save.click(
            on_save,
            inputs=[current_frame_state, annotations_state],
            outputs=[status, frame_selector],
        )
        btn_save_all.click(
            on_save_all,
            inputs=[current_frame_state, annotations_state],
            outputs=[status, frame_selector],
        )

    return demo


def launch_annotator(frames_dir: str, port: int = 7860, share: bool = False):
    """Launch the Gradio annotation UI.

    Args:
        frames_dir: Directory containing extracted frames (.jpg / .png).
            Existing ``<stem>.json`` files in this directory are auto-loaded
            so previous work is restored.
        port:  Preferred local port; if busy, the next free port in
               [port, port+20] is used automatically.
        share: If True, create a public Gradio share link (requires internet).
    """
    app = _build_app(frames_dir)

    actual_port = port
    for candidate in range(port, port + 21):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", candidate)) != 0:
                actual_port = candidate
                break

    print(
        f"\n{'='*60}\n"
        f"  SAM2-Mice Annotator (labelme format)\n"
        f"  frames_dir : {frames_dir}\n"
        f"  local URL  : http://localhost:{actual_port}\n"
        f"  SSH tunnel : ssh -L {actual_port}:localhost:{actual_port} <user>@<server>\n"
        f"{'='*60}\n"
    )

    launch_kwargs = dict(
        server_name="0.0.0.0", server_port=actual_port, share=share, inbrowser=False,
    )
    css = getattr(app, "_sam2_css", None)
    if css is not None:
        try:
            app.launch(css=css, theme=gr.themes.Soft(), **launch_kwargs)
            return
        except TypeError:
            pass
    app.launch(**launch_kwargs)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM2-Mice interactive frame annotator")
    parser.add_argument("--frames_dir", required=True, help="Directory with extracted frames")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    launch_annotator(args.frames_dir, port=args.port, share=args.share)
