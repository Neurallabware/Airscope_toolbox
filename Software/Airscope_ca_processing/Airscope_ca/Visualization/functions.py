import cv2
import numpy as np
from tqdm import tqdm

import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
from scipy.sparse import csc_matrix
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# from Angran Li. Plot for trace visualization
def plot_trace(A_path, C_path, output_dir, frame_len = 1000, neuron_step = 100):
    # neuron_image_data_path = r"/mnt/nas/lk/pico/Experiments/0729_multi_animal_n3/exp01_m20/29-Jul-24/23-24-56-719/analysis/seg_results_thresh_pmap_1/seg_results_filtered.mat"
    neuron_image_data = loadmat(A_path)

    # neuron_intensity_data_path = r"/mnt/nas/lk/pico/Experiments/0729_multi_animal_n3/exp01_m20/29-Jul-24/23-24-56-719/analysis/seg_results_thresh_pmap_1/infer_results_filtered.mat"
    C = loadmat(C_path)["C"]
    T = C.shape[1]
    # output_dir = "/mnt/nas/lk/pico/Experiments/0729_multi_animal_n3/exp01_m20/29-Jul-24/23-24-56-719/Neuron_trace/"  # Replace with the actual path
    os.makedirs(output_dir, exist_ok=True)

    names = [name for name in neuron_image_data.keys() if "frame" in name]
    # Extract frames and convert sparse matrices to dense arrays
    frame_shape = neuron_image_data[names[0]].shape
    summed_frames = np.zeros(frame_shape, dtype=np.int8)

    for frame_name in names:
        frame = np.array(neuron_image_data[frame_name].todense(), dtype=np.int8)
        summed_frames += frame

    # Normalize the summed frames to binary image (0 and 255)
    binary_image = (summed_frames > 0).astype(np.uint8) * 255
    image = Image.fromarray(binary_image, mode='L')
    image.save(os.path.join(output_dir,"whole_neurons.png"))
    print("Whole neurons image saved successfully!")

    # Extract specific frames and neurons for the right plot
    if T < frame_len:
        frame_start = 0
        frame_end = T
    else:
        frame_start = T // 2 - frame_len // 2
        frame_end = frame_start + frame_len
    
    neuron_total = C.shape[0]
    # neuron_step = 100
    # neuron_start, neuron_end=100, 200
    draw_y = True
    for neuron_start in tqdm(range(0, neuron_total, neuron_step)):

        neuron_end = min(neuron_start + neuron_step, neuron_total)
        if neuron_end == neuron_total:
            draw_y = False

        print(f"Process neurons from {neuron_start+1} to {neuron_end}.")

        neuron_indices = np.arange(neuron_start,neuron_end)

        # plasma
        cmap = plt.get_cmap('turbo', neuron_end-neuron_start)  # You can change 'viridis' to any other colormap
        norm = plt.Normalize(vmin=0, vmax=neuron_end-neuron_start)

        # Create a figure and axis for the plots
        fig, axs = plt.subplots(1, 3, figsize=(35, 10))

        temp_image = np.array(image.convert("RGB"))
        for i in neuron_indices:
            frame = np.array(neuron_image_data[names[i]].todense(), dtype=np.uint8)
            color=cmap(norm(i - neuron_start))
            temp_image[frame == 1] = (np.array(color[:3]) * 255).astype(np.uint8)
        temp_image = Image.fromarray(temp_image)

        # Left plot: Image
        axs[0].imshow(temp_image, cmap='gray')
        axs[0].axis('off')

        # Center plot: Neuron intensity heatmap
        cax = axs[1].imshow(C, aspect='auto', cmap='jet', interpolation='nearest')
        fig.colorbar(cax, ax=axs[1])
        axs[1].set_xlabel('Time (frames)')
        axs[1].set_ylabel('Neurons')
        # Add a rectangle to the center plot to highlight the selected area
        rect = Rectangle((frame_start, neuron_indices[0]),  # (x,y) position of the rectangle
                        frame_end - frame_start,  # width of the rectangle
                        len(neuron_indices),  # height of the rectangle
                        linewidth=1.5, edgecolor='red', facecolor='none', linestyle='-')
        axs[1].add_patch(rect)

        # # Right plot: Intensity curves for selected neurons

        offset = 20  # Offset to separate the curves visually
        for i, idx in enumerate(neuron_indices):
            axs[2].plot(np.arange(frame_start, frame_end), C[idx, frame_start:frame_end] * 80 + i * offset, linewidth=0.7, color = cmap(norm(i)))
        axs[2].set_xlabel('Time (frames)')
        axs[2].set_ylabel('Neuron number')

        # Set y-ticks to the actual neuron indices
        if draw_y:
            y_ticks = np.arange(0, len(neuron_indices) * offset + 1, step=len(neuron_indices) * offset // 10)
            y_tick_labels = np.arange(neuron_start, neuron_end + 1, step=(neuron_end - neuron_start) // 10) + 1
            axs[2].set_yticks(y_ticks)
            axs[2].set_yticklabels(y_tick_labels)
        else:
            axs[2].set_yticks([])

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{neuron_start+1}_to_{neuron_end}.png')
        plt.savefig(save_path, dpi=100)

# save mc_video to avi format
def save_video(video, fr, outpath, quality=97):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(outpath, fourcc, fr, (video[0].shape[1], video[0].shape[0]), isColor=False)

    # Set the quality (1-100, higher means better quality)
    out.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)

    for frame in video:
        # Ensure each frame is grayscale and uint8 
        # Check if the frame has 3 channels (BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
        else:
            gray_frame = frame  # Frame is already grayscale
        # Continue processing gray_frame
        gray_frame = gray_frame.astype(np.uint8) 

        out.write(gray_frame)

    out.release()

   
# def filter_masks_by_roundness(masks, max_axis_ratio=3, min_occupancy=0.5):
#     """
#     Filters a list of binary masks based on their roundness using elliptical fitting.

#     Args:
#         masks: A list of N d1 x d2 binary masks (numpy arrays).
#         max_axis_ratio: The maximum allowed ratio between the long and short axes of the fitted ellipse.
#         min_occupancy: The minimum required ratio of cell pixels within the fitted ellipse.

#     Returns:
#         A list of invalid ids
#     """

#     invalid_id = []

#     for idx, mask in tqdm(enumerate(masks)):
#         # Find contours in the mask
#         contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Ensure there's at least one contour (representing the cell)
#         if len(contours) > 0:
#             # Fit an ellipse to the largest contour
#             contour = max(contours, key=cv2.contourArea)
#             if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
#                 ellipse = cv2.fitEllipse(contour)

#                 # Calculate axis ratio
#                 (center, axes, orientation) = ellipse
#                 major_axis_length = max(axes)
#                 # print('major_axis_length:', major_axis_length)
#                 minor_axis_length = min(axes)
#                 if minor_axis_length == 0:
#                     invalid_id.append(idx)
#                     continue
#                 # print('minor_axis_length:', minor_axis_length)
#                 axis_ratio = major_axis_length / minor_axis_length
#                 # print('axis_ratio:', axis_ratio)
                

#                 # Calculate occupancy
#                 ellipse_mask = np.zeros_like(mask).astype(np.uint8)
#                 cv2.ellipse(ellipse_mask, ellipse, 1, -1)  # Fill the ellipse
#                 occupancy = np.sum(mask & ellipse_mask) / np.sum(ellipse_mask) # sum of the intersection of mask and ellipse mask divided by the sum of ellipse mask
#                 # print('occupancy:', occupancy)

#                 # Check if the mask meets the criteria
#                 if axis_ratio > max_axis_ratio or occupancy < min_occupancy:
#                     invalid_id.append(idx)
#             else:
#                 invalid_id.append(idx)
#         else:
#             invalid_id.append(idx)

#     return invalid_id


def filter_masks_by_roundness(masks, max_axis_ratio=3, min_occupancy=0.5):
    """
    Filters a list of binary masks based on their roundness using elliptical fitting.

    Args:
        masks: A list of N d1 x d2 binary masks (numpy arrays).
        max_axis_ratio: The maximum allowed ratio between the long and short axes of the fitted ellipse.
        min_occupancy: The minimum required ratio of cell pixels within the fitted ellipse.

    Returns:
        A list of invalid ids
    """

    invalid_id = []

    for idx, mask in tqdm(enumerate(masks)):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ensure there's at least one contour (representing the cell)
        if len(contours) > 0:
            # Fit an ellipse to the largest contour
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis_length = max(axes)
                minor_axis_length = min(axes)

                # 💥 Robust check before using axes
                if (not np.isfinite(major_axis_length) or not np.isfinite(minor_axis_length) or
                    major_axis_length <= 0 or minor_axis_length <= 0):
                    invalid_id.append(idx)
                    continue

                axis_ratio = major_axis_length / minor_axis_length

                # Calculate occupancy only if ellipse is valid
                ellipse_mask = np.zeros_like(mask, dtype=np.uint8)
                try:
                    cv2.ellipse(ellipse_mask, ellipse, 1, -1)  # Fill the ellipse
                except cv2.error as e:
                    print(f"[Warning] Failed to draw ellipse at idx {idx}: {e}")
                    invalid_id.append(idx)
                    continue

                occupancy = np.sum(mask & ellipse_mask) / np.sum(ellipse_mask)

                # Check if the mask meets the criteria
                if axis_ratio > max_axis_ratio or occupancy < min_occupancy:
                    invalid_id.append(idx)
            else:
                invalid_id.append(idx)
        else:
            invalid_id.append(idx)

    return invalid_id
