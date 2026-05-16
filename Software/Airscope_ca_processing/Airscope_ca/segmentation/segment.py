# %%
import numpy as np
import time
import os
import glob
import tifffile
from skimage import io
import matplotlib.pyplot as plt

import numpy as np
from scipy import sparse
from scipy.io import savemat, loadmat
from scipy.sparse import lil_matrix
import pickle

# load modules

from .combine import segs_results, unique_neurons2_simp, \
    group_neurons, piece_neurons_IOU, piece_neurons_consume
from .complete_post import complete_segment
from .par3 import fastthreshold

def convert_mask(mask):
    row_sum = np.sum(mask, axis = 0)
    col_sum = np.sum(mask, axis = 1)
    row_mask = np.where(row_sum > 0)
    col_mask = np.where(col_sum > 0)
    roi = [np.min(col_mask), np.max(col_mask)+1, np.min(row_mask), np.max(row_mask)+1] # notice +1!
    mask_valid = mask[roi[0]:roi[1],roi[2]:roi[3]]
    return mask_valid, roi

# %%
def convert_to_sparse(frames):
    """
    Convert N x H x W matrix where most elements are 0 to a list of sparse matrices.

    Parameters:
    - frames : ndarray
        3D numpy array of shape (N, H, W)

    Returns:
    - sparse_frames : list
        List of `lil_matrix` objects, each corresponding to a frame.

        dummy_data = np.zeros((100, 1000, 1000), dtype=int)
        # Simulate some non-zero areas randomly
        for i in range(100):
            x, y = np.random.randint(0, 990, size=2)
            dummy_data[i, x:x+10, y:y+10] = 1

        # Convert to sparse format
        sparse_frames = convert_to_sparse(dummy_data)
    """
    N, H, W = frames.shape
    sparse_frames = []
    for i in range(N):
        # Create a lil_matrix for each frame
        frame_sparse = lil_matrix((H, W), dtype=int)
        # Find indices where the frame is not zero
        non_zero_indices = np.nonzero(frames[i])
        # Set these indices in the sparse matrix
        frame_sparse[non_zero_indices] = frames[i][non_zero_indices]
        sparse_frames.append(frame_sparse)
    return sparse_frames

def save_sparse_frames_to_mat(sparse_frames, filename='sparse_data.mat'):
    """
    Save a list of sparse matrices to a .mat file.

    Parameters:
    - sparse_frames : list of lil_matrix
        List of sparse matrices.
    - filename : str
        The name of the .mat file to save.

    # Example usage
        save_sparse_frames_to_mat(sparse_data)
    """
    data_dict = {f'frame_{i}': frame for i, frame in enumerate(sparse_frames)}
    savemat(filename, data_dict)

def load_sparse_frames_from_mat(filename):
    """
    Loads Boolean frames from a MAT file and returns them as a list.

    Args:
        filename: The path to the MAT file.

    Returns:
        A list of Boolean NumPy arrays representing the frames.
    """

    data = loadmat(filename)
    frames = []

    # Iterate over the keys in the data dictionary
    for key in data:
        # Check if the key matches the expected pattern ('frame_...')
        if key.startswith('frame_'):
            # Extract the frame data and convert it to Boolean
            frame = data[key].astype(bool)
            frames.append(frame.toarray())

    return frames



def neuron_segmentation(prob_map,
                        output_folder,
                        pixel_size = 2, # in um
                        minArea = 10, # min nueron size
                        avgArea = 30, # avg neuron size
                        thresh_pmap = 10, # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1) . SENSITIVE
                        thresh_mask = 0.5,  # values higher than "thresh_mask" times the maximum value of the mask are set to one
                        thresh_COM0 = 4, # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels)
                        thresh_COM = 8, # maximum COM distance of two masks to be considered the same neuron (unit: pixels)
                        cons = 3):

    display = True
    start = time.time()
    
    os.makedirs(output_folder, exist_ok=True)

    prob_map = prob_map.astype(np.float32)
    prob_map = (prob_map-prob_map.min())/ 255.0 # as the input is uint8, normalize to 0-1
    nframes = prob_map.shape[0]
    Lx =  prob_map.shape[1]
    Ly =  prob_map.shape[2]

    # set parameters for post-processing
    # Optimization_Info = loadmat(os.path.join('.', 'Optimization_Info_{}.mat'.format(0)))
    # Params_post_mat = Optimization_Info['Params'][0]
    Params_post={
            # minimum area of a neuron (unit: pixels).
            'minArea': minArea, 
            # average area of a typical neuron (unit: pixels) 
            'avgArea': avgArea,
            # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
            'thresh_pmap':thresh_pmap,  # for PICO, this is ultra low
            # values higher than "thresh_mask" times the maximum value of the mask are set to one.
            'thresh_mask': thresh_mask, 
            # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels)
            'thresh_COM0': thresh_COM0 / pixel_size, 
            # maximum COM distance of two masks to be considered the same neuron (unit: pixels)
            'thresh_COM': thresh_COM / pixel_size, 
            # minimum IoU of two masks to be considered the same neuron
            'thresh_IOU': 0.5, 
            # minimum consume ratio of two masks to be considered the same neuron
            'thresh_consume': 0.75, 
            # minimum consecutive number of frames of active neurons
            'cons':cons}

    # post-processing
    print(Params_post)
    Params_post_copy = Params_post.copy()
    Params_post_copy['thresh_pmap'] = None # Avoid repeated thresholding in postprocessing

    pmaps_b = np.zeros(prob_map.shape, dtype='uint8')
    thresh_pmap_float = (Params_post['thresh_pmap']+1)/256 # for published version

    # threshold the probability map to binary activity
    print(prob_map.dtype)
    print(pmaps_b.dtype)
    fastthreshold(prob_map, pmaps_b, thresh_pmap_float)
    # io.imsave(im_folder + '\\' + im_name + '_seg' + 'threshold.tiff', pmaps_b, check_contrast=False)

    # the rest of post-processing. The result is a 2D sparse matrix of the segmented neurons
    p = None
    useWT = False
    useMP = False
    Masks_2 = complete_segment(pmaps_b, Params_post_copy, useMP=useMP, p=p, useWT=useWT, display=display)
    if display:
        finish = time.time()
        time_post = finish-start
        time_frame_post = time_post/nframes*1000
        print('Processing Done: {:6f} s, {:6f} ms/frame'.format(time_post, time_frame_post))

    # convert to a 3D array of the segmented neurons
    Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')

    # save in smart way, utilizing sparse matrix
    sparse_frames = convert_to_sparse(Masks)
    save_sparse_frames_to_mat(sparse_frames, output_folder + '/seg_results.mat') # note can be empty


    # save summary image
    result_name = output_folder + '/SEG_SUM.png'
    if np.sum(Masks, axis = 0).max() == 0:
        mask_sum = np.zeros(prob_map.shape[1:], dtype='uint8')
        io.imsave(result_name, mask_sum)
        
        Masks = []
        C = []
        return Masks, C
    else:
        mask_sum = np.sum(Masks, axis = 0).astype('uint8')
        mask_sum = mask_sum * int(255/np.max(mask_sum))
        io.imsave(result_name, mask_sum)

    if display:
        finish = time.time()
        time_post = finish-start
        print('Mask saved: {:6f} s'.format(time_post))
    


# %% further show the temporal information
    N = Masks.shape[0] # component number
    print('Final Neuron number: ', N)

    C = np.zeros((N, nframes), dtype = 'float32')
    for i in range(N):
        # read the target positions
        mask_valid, roi = convert_mask(Masks[i])
        curr_temporal_signal = np.squeeze(np.sum(prob_map[:,roi[0]:roi[1],roi[2]:roi[3]] * mask_valid.astype(np.float32), axis=(1, 2))) # broadcast here.
        C[i] = curr_temporal_signal

    if display:
        finish = time.time()
        time_post = finish-start
        print('Temporal signal extracted: {:6f} s'.format(time_post))

    # save the time 
    savemat(output_folder + '/infer_results.mat', {'C': C})
    
    if display:
        finish = time.time()
        time_post = finish-start
        print('Temporal signal saved: {:6f} s'.format(time_post))
    
    return Masks, C
