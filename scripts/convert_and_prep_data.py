from skimage import io
import SimpleITK as sitk
import numpy as np
import os 
import random
import json

unlabelled_dir = "/arc/project/st-rohling-1/data/kidney/cleaned_anonymized_kidney_images/included/"
img_dir = "/arc/project/st-rohling-1/data/kidney/cleaned_annotated_images/"
label_dir = "/arc/project/st-rohling-1/data/kidney/masks_jr/"
split_file = "/scratch/st-rohling-1/contrastive_learning/domain_specific_cl/split.json"
destination_dir = "/arc/project/st-rohling-1/data/kidney/dscl_data/"

def convert_2d_image_to_nifti(input_filename: str, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_grayscale: bool = False) -> None:
    img = io.imread(input_filename)

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_grayscale:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'
    else:
        if(img.shape[0]==1):
            img = np.array([img[0], img[0], img[0]])

    for j, i in enumerate(img):

        if is_grayscale:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_grayscale:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")

def convert_unlabelled_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    filenames = []

    for filename in os.listdir(input_dir):
        # print(filename)
        if filename[-3:] == "png":
            unique_name = filename[:-4]
            input_image_file = os.path.join(input_dir, filename)
            output_image_file = os.path.join(output_dir, unique_name)
            # convert_2d_image_to_nifti(input_image_file, output_image_file, is_grayscale=True)
            filenames.append(unique_name)
    return filenames

def convert_images_from_list(images, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    filenames = []
    for image in images:
        unique_name = image
        input_image_file = os.path.join(input_dir, image + ".png")
        output_image_file = os.path.join(output_dir, unique_name)
        if os.path.exists(input_image_file):
            convert_2d_image_to_nifti(input_image_file, output_image_file, is_grayscale=True)
            filenames.append(unique_name)
        else:
            print(input_image_file)
    return filenames


if __name__ == '__main__':

    unlabelled_output_dir = os.path.join(destination_dir, "unlabelled")

    with open(split_file, "r") as fp:
        all_cases = json.load(fp)
    print(all_cases.keys())
    training_cases = all_cases['1']["train"] + all_cases['1']["val"]
    testing_cases = all_cases['1']["test"]
    
    file_dict = {}

    imagesTr_cap = os.path.join(destination_dir, "Task001_KidneyCapsule", "imagesTr")
    imagesTr_reg = os.path.join(destination_dir, "Task002_KidneyRegions", "imagesTr")
    labelsTr_cap = os.path.join(destination_dir, "Task001_KidneyCapsule", "labelsTr")
    labelsTr_reg = os.path.join(destination_dir, "Task002_KidneyRegions", "labelsTr")
    for directory in [imagesTr_cap, imagesTr_reg, labelsTr_cap, labelsTr_reg]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # file_dict["pretrain"] = convert_unlabelled_images(unlabelled_dir, unlabelled_output_dir)
    print(training_cases)
    imagesTr = convert_images_from_list(training_cases, img_dir, imagesTr_cap)
    convert_images_from_list(training_cases, img_dir, imagesTr_reg)
    convert_images_from_list(training_cases, os.path.join(label_dir, "capsule"), labelsTr_cap)
    convert_images_from_list(training_cases, os.path.join(label_dir, "regions"), labelsTr_reg)

    file_dict["train"] = all_cases['1']["train"]
    file_dict["val"] = all_cases['1']["val"]


    imagesTs_cap = os.path.join(destination_dir, "Task001_KidneyCapsule", "imagesTs")
    imagesTs_reg = os.path.join(destination_dir, "Task002_KidneyRegions", "imagesTs")
    for directory in [imagesTs_cap, imagesTs_reg]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    file_dict["test"] = convert_images_from_list(testing_cases, img_dir, imagesTs_cap)
    # with open(os.path.join(destination_dir, "split.json"), 'w') as fp:
    #     json.dump(file_dict, fp)
