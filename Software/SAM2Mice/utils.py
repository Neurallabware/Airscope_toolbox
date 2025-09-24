import gzip
import json
import os
import pickle
import shutil
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def generate_mask(mask, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


def save_video_segments(file_name, video_segments):
    processed_segments = []

    for segment in video_segments:
        if segment == None:
            processed_segments.append(None)
            continue
        processed_segment = {}
        for obj_id, mask in segment.items():
            mask = np.squeeze(mask, axis=0)
            mask = np.where(mask > 0, 1, 0).astype(np.int8)
            # Convert mask to binary and then to bytes
            shape = mask.shape
            binary_mask = np.packbits(mask, axis=None)
            processed_segment[obj_id] = binary_mask
        processed_segments.append(processed_segment)
    # Compress with gzip
    with gzip.open(file_name, 'wb') as f:
        pickle.dump(processed_segments, f)
    print(f"{file_name} saved successfully!")
    return shape

def load_video_segments(file_name, shape):
    with gzip.open(file_name, 'rb') as f:
        processed_segments = pickle.load(f)

    video_segments = []
    for segment in processed_segments:
        if segment == None:
            video_segments.append(None)
            continue
        decompressed_segment = {}
        for obj_id, binary_mask in segment.items():
            # Convert bytes back to mask
            mask = np.unpackbits(binary_mask).astype(np.int8)
            mask = mask.reshape(shape)
            mask = np.expand_dims(mask, axis=0)  #  h*w to 1*h*w
            decompressed_segment[obj_id] = mask
        video_segments.append(decompressed_segment)

    print(f"Successfully loaded {file_name}!")
    return video_segments


# 稀疏矩阵存储，可选
# # 存储 batched_video_segments
# def save_video_segments(file_name, video_segments):
#     processed_segments = []
#
#     for segment in video_segments:
#         processed_segment = {}
#         for obj_id, mask in segment.items():
#             # 将 mask 中大于0的值置为1，并转换为 int8 类型
#             processed_mask = np.where(mask > 0, 1, 0).astype(np.int8)
#
#             processed_mask = sp.csr_matrix(processed_mask)
#             # processed_mask = mask
#             processed_segment[obj_id] = processed_mask
#         processed_segments.append(processed_segment)
#
#     # 使用 gzip 进行压缩存储
#     with gzip.open(file_name, 'wb') as f:
#         pickle.dump(processed_segments, f)
#     print(f"{file_name} saved successfully !"  )
#
#
# # 加载 batched_video_segments
# def load_video_segments(file_name):
#     with gzip.open(file_name, 'rb') as f:
#         processed_segments = pickle.load(f)
#
#     video_segments = []
#     for segment in processed_segments:
#         decompressed_segment = {}
#         for obj_id, processed_mask in segment.items():
#             decompressed_segment[obj_id] = processed_mask.toarray()
#         video_segments.append(decompressed_segment)
#
#     print(f"successfully load {file_name}!")
#     return video_segments


def generate_whole_masked_video(frame_paths, video_segments, output_video_path, fps=10):
    """
    Generates a video from masked frames using OpenCV.

    Args:
        frame_paths (list): List of paths to input frames.
        video_segments (list): List of dictionaries containing mask information for each frame.
        output_video_path (str): Path to save the generated video.
        fps (int): Frames per second for the output video.
    """
    masked_frames = []  # numpy format masked image list
    height, width = None, None

    # Create masked frames
    for out_frame_idx, frame_path in enumerate(tqdm(frame_paths)):
        img = Image.open(frame_path)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask_image = generate_mask(out_mask, obj_id=out_obj_id)

            masked_img_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
            masked_img = Image.alpha_composite(img.convert('RGBA'), masked_img_pil.convert('RGBA'))
            img = masked_img.convert('RGB')

        masked_img_np = np.array(img)  # Convert PIL image to NumPy array
        if height is None or width is None:
            height, width, _ = masked_img_np.shape  # Get dimensions from the first frame
        masked_frames.append(masked_img_np)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame in masked_frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    video_writer.release()  # Release the writer

    print(f"Video saved to {output_video_path}")


# if the number of images are so large, the ram will not be enough,
# so we need to store each mask firstly and then generate video
# def generate_masked_video_and_image(frame_paths, video_segments, output_video_path, fps, temp_folder, save_masks=True):
#     if not os.path.exists(temp_folder):
#         os.makedirs(temp_folder)
#
#     masked_image_paths = []
#     frame_size = None
#
#     print("Generating masks...")
#     for out_frame_idx, frame_path in tqdm(enumerate(frame_paths), total=len(frame_paths), desc="Mask Generation"):
#         if video_segments[out_frame_idx] is None:
#             masked_img_path = os.path.join(temp_folder, f'{out_frame_idx:05d}.jpg')
#             masked_image_paths.append(masked_img_path)
#             shutil.copy(frame_path, masked_img_path)
#             if frame_size is None:
#                 img = cv2.imread(frame_path)
#                 frame_size = (img.shape[1], img.shape[0])  # Width, Height
#             continue
#
#         img = Image.open(frame_path)
#         for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#             mask_image = generate_mask(out_mask, obj_id=out_obj_id)
#             masked_img_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
#             masked_img = Image.alpha_composite(img.convert('RGBA'), masked_img_pil.convert('RGBA'))
#             img = masked_img.convert('RGB')
#
#         masked_img = img.convert('RGB')
#         # Save the masked image
#         masked_img_path = os.path.join(temp_folder, f'{out_frame_idx:05d}.jpg')
#         masked_img.save(masked_img_path)
#         masked_image_paths.append(masked_img_path)
#
#         # Update frame size
#         if frame_size is None:
#             frame_size = masked_img.size  # (width, height)
#
#     # Initialize VideoWriterMJPG  mp4v
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for .mp4 format
#     video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
#
#     print("Writing frames to video...")
#     for img_path in tqdm(masked_image_paths, desc="Video Writing"):
#         frame = cv2.imread(img_path)
#         video_writer.write(frame)
#
#     # Release the video writer
#     video_writer.release()
#
#     # Optionally, clean up the temporary folder after video is created
#     if not save_masks:
#         for path in masked_image_paths:
#             os.remove(path)
#         shutil.rmtree(temp_folder)


import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image


def generate_masked_video_and_image(frame_paths, video_segments, output_video_path, fps, temp_folder, save_masks=True):
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    masked_image_paths = []
    frame_size = None

    def process_frame(out_frame_idx, frame_path):
        nonlocal frame_size
        masked_img_path = os.path.join(temp_folder, f'{out_frame_idx:05d}.jpg')

        if video_segments[out_frame_idx] is None:
            shutil.copy(frame_path, masked_img_path)
            if frame_size is None:
                img = cv2.imread(frame_path)
                frame_size = (img.shape[1], img.shape[0])  # Width, Height
            return masked_img_path

        img = Image.open(frame_path)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask_image = generate_mask(out_mask, obj_id=out_obj_id)
            masked_img_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
            masked_img = Image.alpha_composite(img.convert('RGBA'), masked_img_pil.convert('RGBA'))
            img = masked_img.convert('RGB')

        masked_img = img.convert('RGB')
        masked_img.save(masked_img_path)

        if frame_size is None:
            frame_size = masked_img.size  # (width, height)
        return masked_img_path

    print("Generating masks in parallel...")
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, idx, path) for idx, path in enumerate(frame_paths)]
        for future in tqdm(futures, total=len(frame_paths), desc="Mask Generation"):
            masked_image_paths.append(future.result())

    # Initialize VideoWriterMJPG  mp4v
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for .mp4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    print("Writing frames to video...")
    for img_path in tqdm(masked_image_paths, desc="Video Writing"):
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    # Optionally, clean up the temporary folder after video is created
    if not save_masks:
        for path in masked_image_paths:
            os.remove(path)
        shutil.rmtree(temp_folder)


def save_frames(queue, output_dirs, batch_size):
    """Worker function to save frames from the queue to disk."""
    while True:
        item = queue.get()
        if item is None:  # Stop signal
            break
        frame, frame_count, current_batch_dir = item
        frame_filename = os.path.join(current_batch_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        queue.task_done()



import threading
from queue import Queue

def extract_frames_multithreaded(video_path, batch_size=1000, output_dir="", num_workers=100):
    """Extract frames from a video and save them in batches using multithreading."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting frames from {video_path}")

    # Create a directory to store batches
    output_dirs = []
    batch_count = 0
    frame_count = 0
    current_batch_dir = os.path.join(output_dir, f"batch_{batch_count + 1}")
    os.makedirs(current_batch_dir, exist_ok=True)
    output_dirs.append(current_batch_dir)

    # Queue for multithreading
    frame_queue = Queue(maxsize=batch_size * 2)  # Buffer for frames

    # Start worker threads
    workers = []
    for _ in range(num_workers):
        thread = threading.Thread(target=save_frames, args=(frame_queue, output_dirs, batch_size))
        thread.daemon = True
        thread.start()
        workers.append(thread)

    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save a frame to the current batch directory
            if frame_count > 0 and frame_count % batch_size == 0:
                # Copy the first frame of the next batch to the current batch
                first_frame_next_batch = frame
                first_frame_filename = os.path.join(current_batch_dir, f"{frame_count:05d}.jpg")
                frame_queue.put((first_frame_next_batch, frame_count, current_batch_dir))

                batch_count += 1
                frame_count = 0
                current_batch_dir = os.path.join(output_dir, f"batch_{batch_count + 1}")
                os.makedirs(current_batch_dir, exist_ok=True)
                output_dirs.append(current_batch_dir)

            frame_queue.put((frame, frame_count, current_batch_dir))
            frame_count += 1
            pbar.update(1)

    # Stop all workers
    for _ in range(num_workers):
        frame_queue.put(None)
    for thread in workers:
        thread.join()

    cap.release()
    return output_dirs













