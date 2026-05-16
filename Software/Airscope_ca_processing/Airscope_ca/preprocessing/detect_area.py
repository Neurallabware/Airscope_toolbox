import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def detect_calcium_center(img, sigma1=60):
    img_blur = gaussian_filter(img.astype(np.float64),sigma=sigma1)
    min_val = np.min(img)
    max_val = np.max(img)
    threshold = (min_val + max_val)/2
    Mask = img_blur > threshold
    XX, YY = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    centerX = np.sum(XX * Mask) // np.sum(Mask)
    centerY = np.sum(YY * Mask) // np.sum(Mask)
    return centerX, centerY


if __name__ == "__main__":
    img = cv2.imread("/mnt/nas/LAR/test/Analysis/mc/all/frame_0.tif", cv2.IMREAD_GRAYSCALE)
    CenterX, CenterY = detect_calcium_center(img, sigma1=60)
    plt.imshow(img)
    plt.scatter(CenterX, CenterY)
    plt.show()
    print()



