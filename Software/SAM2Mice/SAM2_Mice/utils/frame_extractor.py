import os
import threading
from tqdm import tqdm
import cv2
from queue import Queue

class VideoFrameExtractor:
    """Class for extracting frames from videos."""

    @staticmethod
    def extract_frames(video_path, output_dir=""):
        """Extract every frame from a video into sequential JPEG files."""

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for frame_idx in tqdm(range(total_frames), desc="Extracting frames", ncols=100):
            ret, frame = cap.read()
            if not ret:
                break
            filename = f"{frame_idx:05d}.jpg"
            output_path = os.path.join(output_dir, filename) if output_dir else filename

            cv2.imwrite(output_path, frame)

        cap.release()


    @staticmethod
    def extract_bootstrapping_frames(video_path, batch_size=1000, output_dir=""):
        """
        Extract frames from a video file.

        Args:
            video_path (str): Path to the video file.
            batch_size (int, optional): Number of frames per batch. Defaults to 1000.
            output_dir (str, optional): Directory to save extracted frames. Defaults to "".

        Returns:
            list: List of directories containing extracted frames.
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{video_path}'")
            return []

        print(f"Extract frames in {video_path}.")
        # Create a directory to store batches
        output_dirs = []
        batch_count = 0
        frame_count = 0
        current_batch_dir = os.path.join(output_dir, f"batch_{batch_count + 1}")
        os.makedirs(current_batch_dir, exist_ok=True)
        output_dirs.append(current_batch_dir)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame if it's time for a new batch
            if frame_count > 0 and frame_count % batch_size == 0:
                # Add the first frame of the next batch to the previous batch (for bootstrapping)
                frame_filename = f"{current_batch_dir}/{frame_count:05d}.jpg"
                cv2.imwrite(frame_filename, frame)

                batch_count += 1
                frame_count = 0
                current_batch_dir = os.path.join(output_dir, f"batch_{batch_count + 1}")
                os.makedirs(current_batch_dir, exist_ok=True)
                output_dirs.append(current_batch_dir)

            # Save frame as jpg
            frame_filename = f"{current_batch_dir}/{frame_count:05d}.jpg"
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

        cap.release()
        return output_dirs

    @staticmethod
    def extract_bootstrapping_frames_multithreaded(video_path, batch_size=1000, output_dir="", num_workers=4):
        """
        Extract frames from a video and save them in batches using multithreading.

        Args:
            video_path (str): Path to the video file.
            batch_size (int, optional): Number of frames per batch. Defaults to 1000.
            output_dir (str, optional): Directory to save extracted frames. Defaults to "".
            num_workers (int, optional): Number of worker threads. Defaults to 4.

        Returns:
            list: List of directories containing extracted frames.
        """

        def save_frames(queue, batch_size):
            """Worker function to save frames from the queue to disk."""
            while True:
                item = queue.get()
                if item is None:  # Stop signal
                    break
                frame, frame_count, current_batch_dir = item
                frame_filename = os.path.join(current_batch_dir, f"{frame_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                queue.task_done()

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{video_path}'")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Extracting {total_frames} frames from {video_path}")

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
            t = threading.Thread(target=save_frames, args=(frame_queue, batch_size), daemon=True)
            t.start()
            workers.append(t)

        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save a frame to the current batch directory
                if frame_count > 0 and frame_count % batch_size == 0:
                    # Copy the first frame of the next batch to the current batch
                    first_frame_next_batch = frame
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
        for t in workers:
            t.join()

        cap.release()
        return output_dirs
