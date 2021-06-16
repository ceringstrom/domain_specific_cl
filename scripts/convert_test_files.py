import numpy as np
from skimage import io
import SimpleITK as sitk
import random
import os
import argparse

def convert_2d_segmentation_nifti_to_img(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert img.shape[0] == 1, "This function can only export 2D segmentations!"
    img = img[0]
    if transform is not None:
        img = transform(img)

    io.imsave(output_filename, img.astype(export_dtype), check_contrast=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGG csv to json converter")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.input_dir):
        for filename in files:
            if filename.split(".")[1] == "nii":
                prefix = filename.split(".")[0]
                png_name = prefix.replace("pred_seg_id_", "") + ".png"
                input_file = os.path.join(root, filename)
                output_file = os.path.join(args.output_dir, png_name)
                convert_2d_segmentation_nifti_to_img(input_file, output_file)