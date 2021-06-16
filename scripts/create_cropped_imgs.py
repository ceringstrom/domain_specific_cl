# script to crop the images into the target resolution and save them.
import numpy as np
import pathlib
import json

import nibabel as nib

import os.path
from os import path

import sys
sys.path.append("/scratch/st-rohling-1/contrastive_learning/domain_specific_cl/domain_specific_cl")

import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc','prostate_md','kidney_ptr','kidney_tr'])

parse_config = parser.parse_args()
#parse_config = parser.parse_args(args=[])

selection_file = "/scratch/st-rohling-1/contrastive_learning/domain_specific_cl/data/split.json"

if parse_config.dataset == 'acdc':
    print('load acdc configs')
    import experiment_init.init_acdc as cfg
    import experiment_init.data_cfg_acdc as data_list
elif parse_config.dataset == 'prostate_md':
    print('load prostate_md configs')
    import experiment_init.init_prostate_md as cfg
    import experiment_init.data_cfg_prostate_md as data_list
elif parse_config.dataset == 'kidney_ptr' or parse_config.dataset == 'kidney_tr':
    print('load kidney configs')
    import experiment_init.init_kidney_regions as cfg
    import experiment_init.data_cfg_kidney as data_list
else:
    raise ValueError(parse_config.dataset)

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)
print(parse_config.dataset)
if parse_config.dataset == 'acdc' :
    #print('set acdc orig img dataloader handle')
    orig_img_dt=dt.load_acdc_imgs
    start_id,end_id=1,101
elif parse_config.dataset == 'prostate_md':
    #print('set prostate_md orig img dataloader handle')
    orig_img_dt=dt.load_prostate_imgs_md
    start_id,end_id=0,48
elif parse_config.dataset == 'kidney_ptr':
    orig_img_dt = dt.load_kidney_imgs
    with open(selection_file) as fp:
        selection = json.load(fp)
    labeled_id_list=selection['pretrain']
    start_id, end_id = 0, len(labeled_id_list)
elif parse_config.dataset == 'kidney_tr':
    orig_img_dt = dt.load_kidney_imgs
    with open(selection_file) as fp:
        selection = json.load(fp)
    labeled_id_list=selection['train'] + selection["validation"]
    start_id, end_id = 0, len(labeled_id_list)

# For loop to go over all available images
for index in range(start_id,end_id):
    if(index<10):
        test_id='00'+str(index)
    elif(index<100):
        test_id='0'+str(index)
    else:
        test_id=str(index)
    test_id_l=[test_id]
    
    if parse_config.dataset == 'acdc' :
        file_path=str(cfg.data_path_tr)+str(test_id)+'/patient'+str(test_id)+'_frame01.nii.gz'
        mask_path=str(cfg.data_path_tr)+str(test_id)+'/patient'+str(test_id)+'_frame01_gt.nii.gz'
    elif parse_config.dataset == 'prostate_md':
        file_path=str(cfg.data_path_tr)+str(test_id)+'/img.nii.gz'
        mask_path=str(cfg.data_path_tr)+str(test_id)+'/mask.nii.gz'
    elif parse_config.dataset == 'kidney_ptr':
        file_path = os.path.join(cfg.data_path_pretr, labeled_id_list[index] + ".nii.gz")
        mask_path = ""
        test_id_l = [labeled_id_list[index]]
    elif parse_config.dataset == 'kidney_tr':
        file_path = os.path.join(cfg.data_path_tr, "imagesTr", labeled_id_list[index] + ".nii.gz")
        mask_path = os.path.join(cfg.data_path_tr, "labelsTr", labeled_id_list[index] + ".nii.gz")
        test_id_l = [labeled_id_list[index]]
    print(file_path)
    #check if image file exists
    if(path.exists(file_path)):
        print('crop',test_id)
    else:
        print('continue',test_id)
        continue
    
    #check if mask exists
    if(path.exists(mask_path)):
        print("hi")
        # Load the image &/mask
        img_sys,label_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1,label_present=1,prtr=False)
        # Crop the loaded image &/mask to target resolution
        cropped_img_sys,cropped_mask_sys = dt.preprocess_data(img_sys, label_sys, pixel_size)
    else:
        # Load the image &/mask
        img_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1,label_present=1)
        #dummy mask with zeros
        label_sys=np.zeros_like(img_sys)
        # Crop the loaded image &/mask to target resolution
        cropped_img_sys = dt.preprocess_data(img_sys, label_sys, pixel_size, label_present=0)
    
    #output directory to save cropped image &/mask
    if not(parse_config.dataset == 'kidney_ptr' or parse_config.dataset == 'kidney_tr'):
        save_dir_tmp=str(cfg.data_path_tr_cropped)+str(test_id)+'/'
        pathlib.Path(save_dir_tmp).mkdir(parents=True, exist_ok=True)

    if (parse_config.dataset == 'acdc') :             
        affine_tst[0,0]=-cfg.target_resolution[0]
        affine_tst[1,1]=-cfg.target_resolution[1]
    elif (parse_config.dataset == 'prostate_md') :   
        affine_tst[0,0]=cfg.target_resolution[0]
        affine_tst[1,1]=cfg.target_resolution[1]
    elif (parse_config.dataset == 'kidney_ptr') :   
        affine_tst[0,0]=cfg.target_resolution[0]
        affine_tst[1,1]=cfg.target_resolution[1]
    elif (parse_config.dataset == 'kidney_tr') :   
        affine_tst[0,0]=cfg.target_resolution[0]
        affine_tst[1,1]=cfg.target_resolution[1]

    if (parse_config.dataset == 'kidney_ptr'):
        array_img = nib.Nifti1Image(cropped_img_sys, affine_tst)
        pred_filename = os.path.join(str(cfg.data_path_pretr_cropped), labeled_id_list[index]+'.nii.gz')
        nib.save(array_img, pred_filename)
        print(pred_filename)
    elif(parse_config.dataset == 'kidney_tr'):
        array_img = nib.Nifti1Image(cropped_img_sys, affine_tst)
        pred_filename = os.path.join(str(cfg.data_path_tr_cropped), "imagesTr", labeled_id_list[index]+'.nii.gz')
        nib.save(array_img, pred_filename)
        if(path.exists(mask_path)):
            array_mask = nib.Nifti1Image(cropped_mask_sys.astype(np.int16), affine_tst)
            pred_filename = os.path.join(str(cfg.data_path_tr_cropped), "labelsTr", labeled_id_list[index]+'.nii.gz')
            nib.save(array_mask, pred_filename)
    else:
        #Save the cropped image &/mask
        array_img = nib.Nifti1Image(cropped_img_sys, affine_tst)
        pred_filename = str(save_dir_tmp)+'img_cropped.nii.gz'
        nib.save(array_img, pred_filename)
        if(path.exists(mask_path)):
            array_mask = nib.Nifti1Image(cropped_mask_sys.astype(np.int16), affine_tst)
            pred_filename = str(save_dir_tmp)+'mask_cropped.nii.gz'
            nib.save(array_mask, pred_filename)

