import math
import numpy as np
import torch
from torch.nn import init
import cv2


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



def crop_patches(img, patch_size, stride_size):
    T, H, W = img.shape
    patch_size = list(patch_size)
    # bug fixed: if the patch size is larger than the image size, then the patch_num should be 1
    if T < patch_size[0]:
        patch_size[0] = T
    patch_num = (math.ceil((T - patch_size[0]) / stride_size[0]) + 1,
                 math.ceil((H - patch_size[1]) / stride_size[1]) + 1,
                 math.ceil((W - patch_size[2]) / stride_size[2]) + 1)

    patches = []
    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            for k in range(patch_num[2]):
                t_start = i * stride_size[0] if i < patch_num[0] - 1 else T - patch_size[0]
                h_start = j * stride_size[1] if j < patch_num[1] - 1 else H - patch_size[1]
                w_start = k * stride_size[2] if k < patch_num[2] - 1 else W - patch_size[2]
                patches.append(img[t_start:t_start + patch_size[0],
                               h_start:h_start + patch_size[1],
                               w_start:w_start + patch_size[2]])
    return patches, patch_num

# from patches to videos
def concat_patches(patches, origin_shape, patch_num, patch_size, stride_size, overlap_size):
    assert overlap_size[0] % 2 == 0 and overlap_size[1] % 2 == 0 and overlap_size[2] % 2 == 0

    out = np.zeros(origin_shape, dtype=np.float16)

    # bug fix: handle the case when the patch_num is 1
    if patch_num[0] == 1:
        patch_size = list(patch_size)
        patch_size[0] = origin_shape[0]
    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            for k in range(patch_num[2]):
                t_start = 0 if i == 0 else i * stride_size[0] + overlap_size[0] // 2
                h_start = 0 if j == 0 else j * stride_size[1] + overlap_size[1] // 2
                w_start = 0 if k == 0 else k * stride_size[2] + overlap_size[2] // 2

                t_end = (i + 1) * stride_size[0] + overlap_size[0] // 2 if i < patch_num[0] - 1 else origin_shape[0]
                h_end = (j + 1) * stride_size[1] + overlap_size[1] // 2 if j < patch_num[1] - 1 else origin_shape[1]
                w_end = (k + 1) * stride_size[2] + overlap_size[2] // 2 if k < patch_num[2] - 1 else origin_shape[2]

                if i == 0:
                    if patch_num[0] == 1:
                        pt_start = 0
                        pt_end = patch_size[0]
                    else:
                        pt_start = 0
                        pt_end = patch_size[0] - (overlap_size[0] // 2)
                elif i == patch_num[0] - 1:
                    pt_start = patch_size[0] - (t_end - t_start)
                    pt_end = patch_size[0]
                else:
                    pt_start = overlap_size[0] // 2
                    pt_end = patch_size[0] - (overlap_size[0] // 2)

                if j == 0:
                    ph_start = 0
                    ph_end = patch_size[1] - (overlap_size[1] // 2)
                elif j == patch_num[1] - 1:
                    ph_start = patch_size[1] - (h_end - h_start)
                    ph_end = patch_size[1]
                else:
                    ph_start = overlap_size[1] // 2
                    ph_end = patch_size[1] - (overlap_size[1] // 2)

                if k == 0:
                    pw_start = 0
                    pw_end = patch_size[2] - (overlap_size[2] // 2)
                elif k == patch_num[2] - 1:
                    pw_start = patch_size[2] - (w_end - w_start)
                    pw_end = patch_size[2]
                else:
                    pw_start = overlap_size[2] // 2
                    pw_end = patch_size[2] - (overlap_size[2] // 2)

                out[t_start:t_end, h_start:h_end, w_start:w_end] = patches[
                                                                       i * patch_num[1] * patch_num[2] + j * patch_num[
                                                                           2] + k] \
                    [pt_start:pt_end, ph_start:ph_end, pw_start:pw_end]
    return out


def collate_fn(batch):
    batch = list(zip(*batch))
    return torch.stack(batch[0]), torch.stack(batch[1])


