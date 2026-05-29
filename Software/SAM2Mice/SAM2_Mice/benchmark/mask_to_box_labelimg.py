import os
import cv2
import gzip
import pickle
import argparse
import numpy as np
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
from typing import List, Tuple, Dict, Iterable

from tqdm import tqdm


# ===============
# I/O utilities
# ===============
def load_pickle_maybe_gzip(path: str):
    """Load pickle file; try gzip first, fall back to plain pickle."""
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):  # not gzipped
        with open(path, 'rb') as f:
            return pickle.load(f)


def ensure_dir(p: Path) -> None:
    """Create a directory and all parents if needed."""
    p.mkdir(parents=True, exist_ok=True)


# ================================
# Mask -> boxes (union per object)
# ================================
def mask_to_xyxy_union(mask: np.ndarray, min_area: int = 900) -> List[Tuple[int, int, int, int]]:
    """
    From a binary mask, collect contours and return a single merged bounding box
    around all contours with area >= min_area. Returns [] if nothing valid.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            rects.append((x, y, w, h))
    if not rects:
        return []
    xs = [r[0] for r in rects]
    ys = [r[1] for r in rects]
    xws = [r[0] + r[2] for r in rects]
    yhs = [r[1] + r[3] for r in rects]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xws), max(yhs)
    return [(x1, y1, x2, y2)]


def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert absolute xyxy box coordinates to normalized YOLO cxcywh."""
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


# ==============================
# Writers: YOLO and Pascal VOC
# ==============================
def write_yolo_labels(label_path: Path, objs: List[Tuple[int, Tuple[int, int, int, int]]], img_w: int, img_h: int) -> None:
    """objs: list of (class_id, (x1,y1,x2,y2))"""
    lines: List[str] = []
    for cid, (x1, y1, x2, y2) in objs:
        cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(label_path, 'w') as f:
        f.write("\n".join(lines))


def write_classes_txt(labels_dir: Path, class_names: List[str]) -> None:
    """Write YOLO class names to classes.txt."""
    with open(labels_dir / 'classes.txt', 'w') as f:
        for n in class_names:
            f.write(n + '\n')


def write_voc_xml(xml_path: Path, image_filename: str, img_w: int, img_h: int,
                  objects: List[Tuple[str, Tuple[int, int, int, int]]]) -> None:
    """objects: list of (name, (x1,y1,x2,y2))"""
    ann = Element('annotation')
    SubElement(ann, 'folder').text = xml_path.parent.name
    SubElement(ann, 'filename').text = image_filename
    size = SubElement(ann, 'size')
    SubElement(size, 'width').text = str(img_w)
    SubElement(size, 'height').text = str(img_h)
    SubElement(size, 'depth').text = '3'
    SubElement(ann, 'segmented').text = '0'
    for name, (x1, y1, x2, y2) in objects:
        obj = SubElement(ann, 'object')
        SubElement(obj, 'name').text = name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        bnd = SubElement(obj, 'bndbox')
        SubElement(bnd, 'xmin').text = str(int(x1))
        SubElement(bnd, 'ymin').text = str(int(y1))
        SubElement(bnd, 'xmax').text = str(int(x2))
        SubElement(bnd, 'ymax').text = str(int(y2))
    ElementTree(ann).write(xml_path, encoding='utf-8', xml_declaration=True)


# ==============================
# Readers for visualization
# ==============================
def read_yolo_label_file(path: str, img_w: int, img_h: int) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    """Return list of (class_id, (x1,y1,x2,y2))"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1 and data.size > 0:
        data = data.reshape(1, -1)
    out = []
    for row in data:
        cid, cx, cy, w, h = row[:5]
        cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
        out.append((int(cid), (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))))
    return out


def read_voc_xml_file(path: str) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Read Pascal VOC XML boxes as class-name and xyxy tuples."""
    import xml.etree.ElementTree as ET
    if not os.path.exists(path):
        return []
    tree = ET.parse(path)
    root = tree.getroot()
    out = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        b = obj.find('bndbox')
        x1 = int(float(b.find('xmin').text))
        y1 = int(float(b.find('ymin').text))
        x2 = int(float(b.find('xmax').text))
        y2 = int(float(b.find('ymax').text))
        out.append((name, (x1, y1, x2, y2)))
    return out


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate visually distinct BGR colors for drawing labels."""
    colors: List[Tuple[int, int, int]] = []
    for i in range(max(1, n)):
        hue = int(180 * i / max(1, n))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    return colors


# ==============================
# Conversion from masks to labels
# ==============================
def convert_masks(
    pickle_path: str,
    out_dir: str,
    img_h: int,
    img_w: int,
    fmt: str = 'yolo',
    min_area: int = 900,
    num_classes: int | None = None,
    class_names: Iterable[str] | None = None,
) -> None:
    """
    processed_segments: list of dict per frame {obj_id -> packed mask bits}
    Writes either YOLO .txt + classes.txt or Pascal VOC .xml files per frame.
    """
    segments = load_pickle_maybe_gzip(pickle_path)
    out_path = Path(out_dir)
    ensure_dir(out_path)

    # Build class names if provided/needed
    if fmt == 'yolo':
        if class_names is not None:
            names = list(class_names)
        else:
            if num_classes is None:
                # infer from max obj_id across frames
                max_id = -1
                for seg in segments:
                    if len(seg) == 0:
                        continue
                    max_id = max(max_id, max(seg.keys()))
                num_classes = max(1, (max_id + 1))
            names = [f"mouse_{i}" for i in range(num_classes)]
        write_classes_txt(out_path, names)

    shape = (img_h, img_w)

    for idx in tqdm(range(len(segments)), desc='Converting masks'):
        seg: Dict[int, np.ndarray] = segments[idx]

        # gather objects per frame
        objs_xyxy: List[Tuple[int, Tuple[int, int, int, int]]] = []
        objs_voc: List[Tuple[str, Tuple[int, int, int, int]]] = []

        for obj_id, mask_bits in seg.items():
            mask = np.unpackbits(mask_bits).astype(np.uint8).reshape(shape)
            boxes = mask_to_xyxy_union(mask, min_area=min_area)  # returns 0 or 1 box here
            for (x1, y1, x2, y2) in boxes:
                if fmt == 'yolo':
                    objs_xyxy.append((int(obj_id), (x1, y1, x2, y2)))
                else:
                    name = f"mouse_{int(obj_id)}" if class_names is None else list(class_names)[int(obj_id)]
                    objs_voc.append((name, (x1, y1, x2, y2)))

        stem = f"{idx:05d}"
        if fmt == 'yolo':
            write_yolo_labels(out_path / f"{stem}.txt", objs_xyxy, img_w, img_h)
        else:
            # Pascal VOC also needs image filename; we keep a conventional name
            write_voc_xml(out_path / f"{stem}.xml", f"{stem}.jpg", img_w, img_h, objs_voc)


# ==============================
# Visualization
# ==============================
def visualize(
    images_dir: str,
    labels_dir: str,
    out_dir: str,
    fmt: str,
    conf_thr: float = 0.0,
) -> None:
    """Draw YOLO or VOC labels on images and write visualization frames."""
    ensure_dir(Path(out_dir))

    # list frames by image files
    img_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])
    # try read classes for color count (YOLO). Otherwise guess 10.
    class_txt = Path(labels_dir) / 'classes.txt'
    if class_txt.exists():
        with open(class_txt, 'r') as f:
            class_names = [l.strip() for l in f if l.strip()]
        colors = generate_colors(len(class_names))
    else:
        class_names = None
        colors = generate_colors(10)

    for fn in tqdm(img_files, desc='Visualizing'):
        img_path = os.path.join(images_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(fn)[0]

        if fmt == 'yolo':
            label_path = os.path.join(labels_dir, stem + '.txt')
            objs = read_yolo_label_file(label_path, w, h)  # list of (cid, (x1,y1,x2,y2))
            for cid, (x1, y1, x2, y2) in objs:
                color = colors[cid % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{cid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), color, cv2.FILLED)
                cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:  # voc
            label_path = os.path.join(labels_dir, stem + '.xml')
            objs = read_voc_xml_file(label_path)  # list of (name,(x1,y1,x2,y2))
            # build a color map for names
            name_to_color: Dict[str, Tuple[int, int, int]] = {}
            for i, (name, (x1, y1, x2, y2)) in enumerate(objs):
                if name not in name_to_color:
                    name_to_color[name] = colors[len(name_to_color) % len(colors)]
                color = name_to_color[name]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = name
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), color, cv2.FILLED)
                cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out_path = os.path.join(out_dir, fn)
        cv2.imwrite(out_path, img)


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser for conversion and visualization."""
    p = argparse.ArgumentParser(description='Convert packed masks to LabelImg annotations and visualize.')
    sub = p.add_subparsers(dest='cmd', required=True)

    pc = sub.add_parser('convert', help='Convert packed masks to annotations')
    pc.add_argument('--pickle', required=True, help='Path to pickle (optionally gz) with processed_segments')
    pc.add_argument('--out', required=True, help='Output labels directory')
    pc.add_argument('--height', type=int, required=True, help='Image height')
    pc.add_argument('--width', type=int, required=True, help='Image width')
    pc.add_argument('--format', choices=['yolo', 'voc'], default='yolo', help='Annotation format for LabelImg')
    pc.add_argument('--min-area', type=int, default=900, help='Min rect area to keep from mask')
    pc.add_argument('--num-classes', type=int, default=None, help='Num classes (YOLO). Inferred from obj ids if omitted')
    pc.add_argument('--classes', type=str, nargs='*', default=None, help='Class names ordered by id (YOLO/VOC)')

    pv = sub.add_parser('vis', help='Visualize annotations over images')
    pv.add_argument('--images', required=True, help='Images directory')
    pv.add_argument('--labels', required=True, help='Labels directory (txt for YOLO or xml for VOC)')
    pv.add_argument('--out', required=True, help='Output directory for visualizations')
    pv.add_argument('--format', choices=['yolo', 'voc'], default='yolo', help='Annotation format to read')
    pv.add_argument('--conf', type=float, default=0.0, help='Confidence threshold (YOLO; if present)')

    return p


def main():
    """Parse command-line arguments and dispatch the selected subcommand."""
    args = build_argparser().parse_args()
    if args.cmd == 'convert':
        convert_masks(
            pickle_path=args.pickle,
            out_dir=args.out,
            img_h=args.height,
            img_w=args.width,
            fmt=args.format,
            min_area=args.min_area,
            num_classes=args.num_classes,
            class_names=args.classes,
        )
    elif args.cmd == 'vis':
        visualize(
            images_dir=args.images,
            labels_dir=args.labels,
            out_dir=args.out,
            fmt=args.format,
            conf_thr=args.conf,
        )


if __name__ == '__main__':
    main()
