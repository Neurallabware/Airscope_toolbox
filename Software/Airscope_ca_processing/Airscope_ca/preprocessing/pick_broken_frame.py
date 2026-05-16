import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

# Define the function to detect broken frames
# @Mingrui Wang, 06132024
def detect_broken_frame(img, threshold_scale=0.06, column_ratio=0.4): # This is too slow for per-frerame processing.
    # Perform edge detection using Prewitt filter
    edgeI_tmp = filters.prewitt(img)

    # Convert the edge detection result to binary (logical 0 and 1 values)
    threshold = np.max(np.array(edgeI_tmp)) * threshold_scale  # You can adjust this threshold as needed 0.05-0.08
    edgeI = np.array(edgeI_tmp) > threshold

    # Resize edgeI to keep only the first 40% columns
    cols = int(edgeI.shape[1] * column_ratio)
    edgeI = edgeI[:, :cols]

    # Compute the mean of edge detection result along columns
    I_mean = np.mean(edgeI, axis=1)

    # Check the condition
    flag = np.sum(I_mean > 0.5) > 0


    return flag


# replace bad frames with closest good frames
# @Yuanlong Zhang, 06212024
def replace_array(flag_array):
    """Replaces bad frames with the nearest good frames.
    
    Args:
        file_list_array: List of file paths for each frame.
        flag_array: List of boolean values indicating if a frame is good (True) or bad (False).

    Returns:
        List of pared (frame id needs to be replace, replaced frame id).
    """
    replace_frame_array = []
    bad_frame_indices = [i for i, flag in enumerate(flag_array) if flag] # note how to deal with flag
    good_frame_indices = [i for i, flag in enumerate(flag_array) if not flag]

    for bad_idx in bad_frame_indices:
        closest_good_idx = None
        min_distance = float("inf")

        # Search backward for nearest good frame
        for i in range(bad_idx - 1, -1, -1):
            if i in good_frame_indices: # good frame
                distance = bad_idx - i
                if distance < min_distance:
                    closest_good_idx = i
                    min_distance = distance
                break  # Stop once the first good frame is found

        # Search forward for nearest good frame (if closer than the backward one)
        for i in range(bad_idx + 1, len(flag_array)):
            if i in good_frame_indices:
                distance = i - bad_idx
                if distance < min_distance:
                    closest_good_idx = i
                break

        # Replace bad frame with nearest good frame (if found)
        if closest_good_idx is not None:
            replace_frame_array.append([bad_idx, closest_good_idx])

    return replace_frame_array


def main():
    # Define file paths
    videopath = 'data/sequences'
    frame_num = 100
    # Read and process the original image
    
    file_list_array = []
    flag_array = []
    for i in range(0, frame_num+1):
        path = os.path.join(videopath, 'frame_'+str(i)+'.jpg')
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flag = detect_broken_frame(img)
        
        file_list_array.append(path)
        flag_array.append(flag)
        # print(flag)
        if flag:
            print(f'Broken frame detected at frame_{str(i)}.jpg')
    # now print correct reading list
    closest_good_idx = replace_array(file_list_array, flag_array)
    for item in closest_good_idx:
        print(item)

if __name__ == '__main__':
    main()