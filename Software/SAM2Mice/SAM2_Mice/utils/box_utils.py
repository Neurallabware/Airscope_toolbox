from typing import List


def calculate_overlap_percentage(box1, box2) -> float:
    """Return what fraction of box1's area is covered by box2."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)

    if box1_area == 0:
        return 0.0

    return intersection_area / float(box1_area)


def calculate_iou(box1, box2) -> float:
    """Return IoU between two [x1,y1,x2,y2] boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = box1_area + box2_area - intersection

    return intersection / float(union) if union > 0 else 0.0


def filter_overlapping_boxes(boxes: List, overlap_threshold: float = 0.9) -> List:
    """Remove boxes that are mostly contained within a larger box."""
    if len(boxes) <= 1:
        return boxes

    box_areas = [(i, (b[2] - b[0]) * (b[3] - b[1])) for i, b in enumerate(boxes)]
    box_areas.sort(key=lambda x: x[1], reverse=True)

    keep_indices = []
    for i, (idx1, _) in enumerate(box_areas):
        keep = True
        for j, (idx2, _) in enumerate(box_areas):
            if i == j:
                continue
            if calculate_overlap_percentage(boxes[idx1], boxes[idx2]) > overlap_threshold:
                keep = False
                break
        if keep:
            keep_indices.append(idx1)

    return [boxes[i] for i in keep_indices]
