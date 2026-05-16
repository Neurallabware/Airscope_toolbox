import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
from scipy import interpolate
import time
import cv2
from scipy.ndimage import gaussian_filter
import tifffile

def process_image(I, sigma=60, power=0.995):
    """
    Process the input image by applying a Gaussian filter, computing a weight map,
    and adjusting the original image using this weight map.

    Parameters:
    - input_path: str, path to the input image
    - output_path: str, path to save the processed image
    - sigma: float, standard deviation for the Gaussian filter
    - power: float, power to raise the weight map

    Returns:
    - None
    """
    # Convert the image to double precision
    I = I.astype(np.float64)

    # Create and apply the Gaussian filter
    I_blur = gaussian_filter(I, sigma=sigma)

    # Compute the weight map
    weight_map = (1.0 / I_blur) * np.min(I_blur)
    weight_map = weight_map ** power

    # Smooth the weight map using the Gaussian filter
    weight_map = gaussian_filter(weight_map, sigma=sigma)

    # Apply the weight map to the original image
    I_weighted = I * weight_map

    # Normalize the result and convert it to an 8-bit image
    I_weighted = cv2.normalize(I_weighted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return I_weighted


def correct_image(img, error_XX_new, error_YY_new):
    """
    Correct an image by applying a deformation based on the provided error matrices.

    Parameters:
    - img: numpy array, the input image to be corrected.
    - error_XX_new: numpy array, the error matrix to adjust the X coordinates.
    - error_YY_new: numpy array, the error matrix to adjust the Y coordinates.

    Returns:
    - img_correction: numpy array, the corrected image.
    """
    # Generate meshgrid for the original coordinates
    X = np.arange(0, img.shape[1])
    Y = np.arange(0, img.shape[0])
    XX, YY = np.meshgrid(X, Y)

    # Apply the error matrices to the original coordinates
    XX_new = XX + error_XX_new
    YY_new = YY + error_YY_new

    # Perform cubic interpolation to correct the image
    # img_correction = interpolate.griddata((XX.ravel(), YY.ravel()), img.ravel(), (XX_new, YY_new), method='cubic')
    img_correction = cv2.remap(img,  XX_new.astype(np.float32), YY_new.astype(np.float32), cv2.INTER_CUBIC)

    return img_correction


def main():
    # load image
    filepath = 'data/single_frame'
    img = cv2.imread(filepath + '/frame_0int_corr.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = process_image(img)

    # load map
    Correction_Map = np.load('ckpt/CorrectionMap.npz')
    error_XX_new = Correction_Map['error_XX_new']
    error_YY_new = Correction_Map['error_YY_new']

    start_time = time.time()
    img_correction = correct_image(img, error_XX_new, error_YY_new)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time taken to correct the image: {elapsed_time} seconds")
    plt.imshow(img_correction, cmap='gray')
    plt.show()
    tifffile.imwrite(filepath + '/frame_0' + 'fov_corr' + '.jpg', img_correction)  # Save as TIFF file



if __name__ == '__main__':

    main()
