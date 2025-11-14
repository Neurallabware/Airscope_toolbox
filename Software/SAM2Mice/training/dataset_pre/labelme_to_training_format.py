import argparse
import base64
import json
import os
import os.path as osp
import imgviz
import PIL.Image
import yaml
from labelme import utils
from tqdm import tqdm

'''multiple json files'''
def process_json_files(json_dir, output_dir):
    # Create output directories
    annotations_dir = osp.join(output_dir, 'Annotations')
    jpegimages_dir = osp.join(output_dir, 'JPEGImages')

    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(jpegimages_dir, exist_ok=True)

    # Get the list of subdirectories
    subdirs = [osp.join(json_dir, d) for d in os.listdir(json_dir) if osp.isdir(osp.join(json_dir, d))]

    for i, subdir in enumerate(subdirs):
        files = os.listdir(subdir)
        for file in tqdm(files):
            if file.endswith('.json'):

                ann_save_dir = os.path.join(annotations_dir, os.path.basename(subdir))
                os.makedirs(ann_save_dir, exist_ok=True)
                jpeg_save_dir = os.path.join(jpegimages_dir, os.path.basename(subdir))
                os.makedirs(jpeg_save_dir, exist_ok=True)

                path = osp.join(subdir, file)  # Full path to each JSON file
                filename = file[:-5]  # Extract filename without the '.json'

                data = json.load(open(path))  # Load JSON file

                img = utils.image.img_b64_to_arr(data['imageData'])  # Decode image data

                label_name_to_value = {"_background_": 0}
                for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                    label_name = shape["label"]
                    if label_name not in label_name_to_value:
                        label_name_to_value[label_name] = len(label_name_to_value)

                lbl, _ = utils.shapes_to_label(
                    img.shape, data["shapes"], label_name_to_value
                )

                # Save original image in JPEGImages
                source_image_path = osp.join(jpeg_save_dir, f'{filename}.jpg')
                PIL.Image.fromarray(img).save(source_image_path)

                # Save label mask in Annotations
                label_image_path = osp.join(ann_save_dir, f'{filename}.png')
                utils.lblsave(label_image_path, lbl)

                print(f'Saved: {source_image_path} and {label_image_path}')

def main():
    parser = argparse.ArgumentParser(
        description="Convert a collection of LabelMe JSON files (organized in subdirectories)"
                    " into training format with 'Annotations' and 'JPEGImages' folders."
    )
    parser.add_argument(
        "json_dir",
        nargs="?",
        default="/mnt/nas/LAR/dataset/sam2_train_data/PICO_five_mouse_train",
        help="Directory containing subdirectories with LabelMe .json files (default: same as previous hardcoded path)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="/mnt/nas/LAR/dataset/sam2_train_data/PICO_five_mouse_dataset",
        help="Directory where output 'Annotations' and 'JPEGImages' will be created (default: same as previous hardcoded path)",
    )

    args = parser.parse_args()

    # Basic validation
    if not osp.exists(args.json_dir):
        parser.error(f"json_dir does not exist: {args.json_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    process_json_files(args.json_dir, args.output_dir)

if __name__ == "__main__":
    main()
