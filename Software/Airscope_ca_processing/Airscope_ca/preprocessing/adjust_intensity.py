import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tifffile

# writtent by Mingrui Wang, modified by Yuanlong Zhang
# 06212024


def adjust_intensity_image(img, sigma1=60, sigma2=60, power=1):
    """
    Process the image by applying Gaussian filters and a weight map, and display the result.

    Parameters:
    - I_test: numpy array, the grayscale input image.
    - sigma1: float, standard deviation for the first Gaussian filter.
    - sigma2: float, standard deviation for the second Gaussian filter.
    - power: float, power to which the weight map is raised.

    Returns:
    - weight_map: float, the maximum value in the processed image used for normalization.
    - img_change: numpy array, the processed image.
    """
    # Apply the first Gaussian filter to blur the image
    img_blur = gaussian_filter(img.astype(np.float64), sigma=sigma1)

    # Create the weight map based on the blurred image
    weight_map = (1.0 / img_blur) * np.min(img_blur)

    # Raise the weight map to the specified power
    weight_map = weight_map ** power

    # Apply the second Gaussian filter to the weight map
    weight_map = gaussian_filter(weight_map, sigma=sigma2)

    # Apply the weight map to the test image
    img_change = img.astype(np.float64) * weight_map

    return weight_map, img_change


def main():
    # Define paths
    filepath = 'data/single_frame'
    
    # Read and process the test image
    I_test = cv2.imread(filepath + '/frame_0.jpg')
    I_test = cv2.cvtColor(I_test, cv2.COLOR_BGR2GRAY)

    # Display the original image
    plt.subplot(121)
    plt.imshow(I_test, cmap='gray')

    # Process and display the image using the function
    weight_map, I_change = adjust_intensity_image(I_test)

    # Calculate the maximum value for normalization
    max_v = np.max(I_change) * 1.5
    
    I_change_frame = I_test.astype(np.float64) * weight_map
    I_change_frame = ((I_change_frame / max_v) * 255).astype(np.uint8)
    
    tifffile.imwrite(filepath + '/frame_0' + 'int_corr' + '.jpg', I_change_frame)  # Save as TIFF file

if __name__ == '__main__':
    main()
