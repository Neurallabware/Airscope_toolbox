# ### this is for a large file, see file in fnames = ['../demo_data/PICO.tif']. file is roughly 500 x  500 x 500
# stand alone version, for reading data
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div


import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
import tifffile as tif
import os
import warnings
# Set environment variables
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '8'
# disable warnings
warnings.filterwarnings("ignore", category=UserWarning)
from PIL import Image

try:
    cv2.setNumThreads(0)
except:
    pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)



import Airscope_ca.caiman as cm
from Airscope_ca.caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from Airscope_ca.caiman.utils.utils import download_demo
from Airscope_ca.caiman.base.movies import movie


def normcorre_function(video, # a numpy array of the video
                       fr,
                       max_shifts, 
                       strides, 
                       overlaps, 
                       max_deviation_rigid, 
                       pw_rigid,
                       shifts_opencv, 
                       border_nan, 
                       downsample_ratio,
                       outpath,
                       template = None,
                       save_movie = True,
                       save_diagnostics = False,
                       dview = None):
    # fr = 10 # frame rate
    # max_shifts = (50, 50)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    # strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
    # overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
    # max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
    # pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
    # shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    # border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
    # downsample_ratio = 0.1 # for displaying purpose

    m_orig = movie(video.astype(np.float32),
                     fr=fr,
                     start_time=0,
                     file_name='video',
                     meta_data=None)

    if dview is None:
        if 'dview' in locals():
            cm.stop_server(dview=dview)
            
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
        input_deview_flag = False
    else:
        input_deview_flag = True

    # create a motion correction object
    mc = MotionCorrect(video, dview=dview, max_shifts=max_shifts,
                    strides=strides, overlaps=overlaps,
                    max_deviation_rigid=max_deviation_rigid, 
                    shifts_opencv=shifts_opencv, nonneg_movie=True,
                    border_nan=border_nan)

    # correct for rigid motion correction and save the file (in memory mapped form)
    if template is not None:
        mc.motion_correct(save_movie=True, template=template)
    else:
        mc.motion_correct(save_movie=True)
    m_rig = cm.load(mc.mmap_file, 
                    fr=fr)
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int) # TODO save, this is for the border pixels
    # let's save this video in uint8
    if save_movie:
        tif.imsave(outpath + '/rigid.tif', m_rig.astype(np.uint8))

    if save_diagnostics:
        plt.figure(figsize = (20,10))
        plt.imshow(mc.total_template_rig, cmap = 'gray') # get the template
        plt.savefig(outpath + '/template.svg', format='svg')
        plt.savefig(outpath +'/template.png')

    if pw_rigid:
        mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
        mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional). We can use this feature to do a very large file compensation

        mc.motion_correct(save_movie=True, template=mc.total_template_rig) # note we do with template input
        out_template = mc.total_template_rig
        # load motion corrected movie, piecewise
        m_els = cm.load(mc.fname_tot_els,
                        fr=fr)
        m_out = m_els
    else:
        out_template = mc.total_template_rig
        m_els = None
        m_out = m_rig

    # do the downsampling and concatenate, for visualization
    if save_diagnostics:
        if pw_rigid:
            concat_movie = cm.concatenate([m_orig.resize(1, 1, downsample_ratio) - mc.min_mov*mc.nonneg_movie,
                            m_rig.resize(1, 1, downsample_ratio), m_els.resize(
                        1, 1, downsample_ratio)], axis=2)
        else:
            concat_movie = cm.concatenate([m_orig.resize(1, 1, downsample_ratio) - mc.min_mov*mc.nonneg_movie,
                            m_rig.resize(1, 1, downsample_ratio)], axis=2)
        tif.imsave(outpath + '/concat_movie.tif', concat_movie.astype(np.uint8))
    # let's save it
    m_nonrig = m_out
    # let's save this videoin uint8
    if save_movie and pw_rigid:
        tif.imsave(outpath + '/non_rigid.tif', m_nonrig.astype(np.uint8))

    if save_diagnostics:
        plt.close()
        plt.figure(figsize = (20,10))
        plt.plot(mc.shifts_rig)
        plt.legend(['x shifts','y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels')
        plt.savefig(outpath +'/rigid_shift.png')
        plt.savefig(outpath + '/rigid_shift.svg', format='svg')

    if save_diagnostics and pw_rigid:
        plt.close()
        plt.figure(figsize = (20,10))
        plt.subplot(2, 1, 1)
        plt.plot(mc.x_shifts_els)
        plt.ylabel('x shifts (pixels)')
        plt.subplot(2, 1, 2)
        plt.plot(mc.y_shifts_els)
        plt.ylabel('y_shifts (pixels)')
        plt.xlabel('frames')
        plt.savefig(outpath +'/non_rigid_shift.png')
        plt.savefig(outpath + '/non_rigid_shift.svg', format='svg')

    if save_diagnostics:
        plt.figure(figsize = (20,10))
        if pw_rigid:
            plt.subplot(1,3,1); plt.imshow(m_orig.local_correlations(eight_neighbours=True, swap_dim=False))
            plt.subplot(1,3,2); plt.imshow(m_rig.local_correlations(eight_neighbours=True, swap_dim=False))
            plt.subplot(1,3,3); plt.imshow(m_els.local_correlations(eight_neighbours=True, swap_dim=False))
        else:
            plt.subplot(1,2,1); plt.imshow(m_orig.local_correlations(eight_neighbours=True, swap_dim=False))
            plt.subplot(1,2,2); plt.imshow(m_rig.local_correlations(eight_neighbours=True, swap_dim=False))
        plt.savefig(outpath +'/local_correlation.png')
        plt.savefig(outpath + '/local_correlation.svg', format='svg')

    if pw_rigid:
        bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                        np.max(np.abs(mc.y_shifts_els)))).astype(int)
    else:
        bord_px_els = bord_px_rig

    # TODO save
    if input_deview_flag == False:
        cm.stop_server(dview=dview) # stop the server
    
    
    plt.close()
    return m_nonrig, bord_px_rig, bord_px_els, out_template

if __name__ == '__main__':
    # load from disk
    fr = 10
    max_shifts = (100, 100)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    strides =  (96, 96)  # create a new patch every x pixels for pw-rigid correction
    overlaps = (48, 48)  # overlap between pathes (size of patch strides+overlaps)
    num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
    max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
    pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
    shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
    downsample_ratio = 0.1 # for displaying purpose
    outpath = 'function_output2'
    os.makedirs(outpath, exist_ok = True)
    fnames = r'../../../Project ultra-compact miniscope\Data_processing\0730data'
    jpg_files = [f for f in os.listdir(fnames) if f.endswith('.jpg')]
    # Sort the files by their numeric names
    jpg_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    # Read each image file and convert it to a NumPy array
    image_arrays = []
    ind = 0
    for file in jpg_files:
        image_path = os.path.join(fnames, file)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_arrays.append(img)
        ind += 1
        if ind > 500:
            break

    # Stack all image arrays into a single NumPy array
    image_stack = np.stack(image_arrays, axis=0)

    # Print the shape of the resulting stack
    print(image_stack.shape)

    m_nonrig, bord_px_rig, bord_px_els, template = normcorre_function(video=image_stack, # a numpy array of the video
                       fr=fr,
                        max_shifts=max_shifts, 
                        strides=strides, 
                        overlaps=overlaps, 
                        max_deviation_rigid= max_deviation_rigid, 
                        pw_rigid=pw_rigid, 
                        shifts_opencv=shifts_opencv, 
                        border_nan=border_nan, 
                        downsample_ratio=downsample_ratio,
                        outpath=outpath,
                        template=None)
    

