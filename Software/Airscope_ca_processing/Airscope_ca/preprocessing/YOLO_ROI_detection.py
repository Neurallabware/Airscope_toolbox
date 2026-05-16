from ultralytics import YOLO

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from typing import Union
from Airscope_ca.preprocessing import detect_calcium_center


# use the center of the 1000*1000 tangle to crop ROI
def correct_crop_parameter(CenterX, CenterY, img, crop_parameter = [153, 303, 1000, 1000]):

    crop_parameter_correct = crop_parameter.copy()
    if CenterX + (crop_parameter[3] + 1) // 2 > img.shape[1]:
        crop_parameter_correct[1] = img.shape[1] - crop_parameter[3] + 1
    else:
        crop_parameter_correct[1] = max(0, CenterX - crop_parameter[3] // 2)

    if CenterY + (crop_parameter[2] + 1) // 2 > img.shape[0]:
        crop_parameter_correct[0] = img.shape[0] - crop_parameter[2] + 1
    else:
        crop_parameter_correct[0] = max(0, CenterY - crop_parameter[2] // 2)

    return crop_parameter_correct



def YOLO_center_detect(model,
                       img: np.ndarray,
                       crop_parameter_init:list=[153, 303, 1000, 1000],
                       output_dir: Union[str, os.PathLike]=""):

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    CenterX_1, CenterY_1 = detect_calcium_center(img)

    img= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_yolo_correct = img.copy()
    img_pre = img.copy()

    crop_parameter = crop_parameter_init.copy()
    crop_parameter_pre = correct_crop_parameter(CenterX_1, CenterY_1, img, crop_parameter = crop_parameter_init)

    print(f"crop parameter pre {crop_parameter_pre}")

    # RuiWang cropping
    x1_crop_pre = crop_parameter_pre[1]
    y1_crop_pre = crop_parameter_pre[0]
    x2_crop_pre = crop_parameter_pre[1] + crop_parameter_pre[2]
    y2_crop_pre = crop_parameter_pre[0] + crop_parameter_pre[3]
    cv2.rectangle(img_pre, (int(x1_crop_pre), int(y1_crop_pre)),
                (int(x2_crop_pre), int(y2_crop_pre)), (255, 255, 0), 2)

    # YOLO object detection
    results = model(img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Object class and confidence
            label = int(box.cls[0])
            confidence = box.conf[0]

            # Draw detection box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw label and confidence
            label_text = f"{dict[label]}: {confidence:.2f}"
            cv2.putText(img, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate center and cropping for 1000x1000 area
            CenterX = int((x1 + x2) / 2)
            CenterY = int((y1 + y2) / 2)

            # 1000x1000 cropping
            crop_parameter = correct_crop_parameter(CenterX, CenterY, img, crop_parameter_init)
            x1_crop = crop_parameter[1]
            y1_crop = crop_parameter[0]
            x2_crop = crop_parameter[1] + crop_parameter[2]
            y2_crop = crop_parameter[0] + crop_parameter[3]
            cv2.rectangle(img_yolo_correct, (int(x1_crop), int(y1_crop)),
                                  (int(x2_crop), int(y2_crop)), (0, 0, 255), 2)
            label_text = f"{dict[label]}: {confidence:.2f}"
            cv2.putText(img_yolo_correct, label_text, (int(x1_crop), int(y1_crop) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)

            print(f"crop parameter yolo: {crop_parameter}")


        # Create subplots to show the images side by side
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

        # Show original image
        axes[0].imshow(img)
        axes[0].set_title("Original YOLO")
        axes[0].axis('off')

        # Show RuiWang cropping image
        axes[1].imshow(img_pre)
        axes[1].set_title("Traditional cropping")
        axes[1].axis('off')

        # Show YOLO 1000x1000 cropping image
        axes[2].imshow(img_yolo_correct)
        axes[2].set_title("YOLO correct cropping")
        axes[2].axis('off')

        # Save the comparison figure
        output_path = os.path.join(output_dir, f"YOLO_ROI.png")
        plt.savefig(output_path, dpi=300)

        plt.close()

        # Display the plots
        # plt.show()

        return crop_parameter

if __name__ == "__main__":
    model = YOLO("ckpt/yolo_v8s.pt")

    # img_dir1 = "/mnt/nas/LAR/dataset/calcium_images_yolo/images/train"
    # img_list1 = [os.path.join(img_dir1, path) for path in os.listdir(img_dir1)]
    #
    # img_dir2 = "/mnt/nas/LAR/dataset/calcium_images_yolo/images/val"
    # img_list2 = [os.path.join(img_dir2, path) for path in os.listdir(img_dir2)]
    #
    # img_list1.extend(img_list2)

    img_path = "Z:/LAR/dataset/calcium_images_yolo/images/train/calcium_image_14.jpg"

    img = cv2.imread(img_path)

    output_dir = "Z:/LAR/test/"

    YOLO_center_detect(model, img, output_dir=output_dir)
