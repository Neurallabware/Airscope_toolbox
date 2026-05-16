import onnxruntime
from io import BytesIO
from PIL import Image

# Basic python and ML Libraries
import os
import random
import numpy as np

# We will be reading images using OpenCV
import cv2
import torch
from torchvision import transforms as torchtrans  
import torch.nn.functional as F

from PIL import Image
import numpy as np
import io
# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import segmentation_models_pytorch as smp

def load_model(model_name):

    # Assuming you're using a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    model = smp.Unet(
        encoder_name="efficientnet-b0",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b0
        encoder_depth = 3,
        decoder_channels = (256, 64, 16),
        encoder_weights="imagenet",       # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                # model output channels (number of classes in your dataset)
        activation='sigmoid'             # activation function, 'sigmoid' for binary segmentation
    ).to(device)

    # For model summary, you can use torchsummary
    # to device
    if torch.cuda.is_available():
        model.cuda()

    # load best model
    model.load_state_dict(torch.load(model_name))
    model.eval()
    return model, device

def post_process(probability, threshold):
    # don't remember where I saw it
    mask = (cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1])
    return mask

def visualize_img_and_mask(image, mask,  original_mask=None, save_path="", save_svg_path=""):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: "Fish", 1: "Flower", 2: "Gravel", 3: "Sugar"}

    if original_mask is None:
        f, ax = plt.subplots(1, 2, figsize=(24, 24))

        ax[0].imshow(image)
        ax[0].set_title("Image", fontsize=fontsize)
        ax[1].imshow(mask)
        ax[1].set_title("Mask", fontsize=fontsize)
    else:
        f, ax = plt.subplots(1, 3, figsize=(24, 12))

        ax[0].imshow(image)
        ax[0].set_title("Image", fontsize=fontsize)
        
        ax[1].imshow(mask)
        ax[1].set_title("Mask", fontsize=fontsize)
        
        ax[2].imshow(original_mask)
        ax[2].set_title("GT", fontsize=fontsize)
    if save_path:
        plt.savefig(save_path)
    if save_svg_path:
        plt.savefig(save_svg_path, format="svg")
    plt.show()

def get_vessel_mask(img, model_name, target_size = 512, best_threshold = 0.5):
    # load the onnx model
    model, device = load_model(model_name)

    # process the image
    img = torchtrans.ToTensor()(img)
    img = img.unsqueeze(0)
    image = F.interpolate(img, size=(target_size, target_size), mode='bilinear', align_corners=True)
    image_vis = np.squeeze(image.cpu().detach().numpy())

    prediction_out = model(image.to(device))  # Run the model with the input data

    probability = np.squeeze(prediction_out.cpu().detach().numpy())
    pr_mask = post_process(probability,best_threshold)
    return image_vis, pr_mask

if __name__ == "__main__":
    
    # get a single image
    img = Image.open('data/sequences/preprocess/frame_0.tif')
    img = np.array(img).astype(np.float32) / 255.0
    
    # get model name
    model_name = 'ckpt/vessel_model.pt'
    
    target_size = 512
    
    best_threshold = 0.5
    image, pr_mask = get_vessel_mask(img, model_name, target_size, best_threshold)
    
    # do visualization
    visualize_img_and_mask(image=image, 
                        mask=pr_mask
                    )
