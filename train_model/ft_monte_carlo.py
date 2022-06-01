import os

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

import matplotlib
matplotlib.use('Agg')

import numpy as np
#to make directories
import pathlib

import sys
sys.path.append('../')

from utils import *

# import wandb
# wandb.init(project="dscl-kidney-ftn-cap", config=tf.flags.FLAGS)

import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc','prostate_md','mmwhs','kidney_cap','kidney_reg'])
#no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr1', choices=['tr1', 'tr2', 'tr8','tr20','ftn'])
#combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1')
#learning rate of seg unet
parser.add_argument('--lr_seg', type=float, default=0.001)

#data aug - 0 - disabled, 1 - enabled
parser.add_argument('--data_aug', type=int, default=1, choices=[0,1])
#version of run
parser.add_argument('--ver', type=int, default=0)

# Pre-training configs
#no of training images
parser.add_argument('--pretr_no_of_tr_imgs', type=str, default='tr52', choices=['tr52','tr22','tr10','ptr'])
#combination of training images
parser.add_argument('--pretr_comb_tr_imgs', type=str, default='c1')
#version of run
parser.add_argument('--pretr_ver', type=int, default=0)
#no of iterations to run
parser.add_argument('--pretr_n_iter', type=int, default=5000)
#data augmentation used in pre-training
parser.add_argument('--pretr_data_aug', type=int, default=0)
# bounding box dim - dimension of the cropped image. Ex. if bbox_dim=100, then 100 x 100 region is randomly cropped from original image of size W x W & then re-sized to W x W.
# Later, these re-sized images are used for pre-training using global contrastive loss.
parser.add_argument('--pretr_cont_bbox_dim', type=int, default=192)
# temperature_scaling factor
parser.add_argument('--temp_fac', type=float, default=0.1)
#learning rate of seg unet
parser.add_argument('--lr_reg', type=float, default=0.001)

# type of global_loss_exp_no for global contrastive loss - used to pre-train the Encoder (e)
# 0 - G^{R}  - default loss formulation as in simCLR (sample images in a batch from all volumes)
# 1 - G^{D-} - prevent negatives to be contrasted for images coming from corresponding partitions from other volumes for a given positive image.
# 2 - G^{D}  - as in (1) + additionally match positive image to corresponding slice from similar partition in another volume
parser.add_argument('--global_loss_exp_no', type=int, default=0)
# no_of_partitions per volume
parser.add_argument('--n_parts', type=int, default=4)

# type of local_loss_exp_no for Local contrastive loss
# 0 - default loss formulation. Sample local regions from two images. these 2 images are intensity transformed version of same image.
# 1 - (0) + sample local regions to match from 2 differnt images that are from 2 different volumes but they belong to corresponding local regions of similar partitions.
parser.add_argument('--local_loss_exp_no', type=int, default=0)

# segmentation loss used for optimization
# 0 for weighted cross entropy, 1 for dice loss w/o background label, 2 for dice loss with background label (default)
parser.add_argument('--dsc_loss', type=int, default=2)

#random deformations - arguments
#enable random deformations
parser.add_argument('--rd_en', type=int, default=1)
#sigma of gaussian distribution used to sample random deformations 3x3 grid values
parser.add_argument('--sigma', type=float, default=5)
#enable random contrasts
parser.add_argument('--ri_en', type=int, default=1)
#enable 1-hot encoding of the labels 
parser.add_argument('--en_1hot', type=int, default=1)
#controls the ratio of deformed images to normal images used in each mini-batch of the training
parser.add_argument('--rd_ni', type=int, default=1)

# no. of local regions to consider in the feature map for local contrastive loss computation
parser.add_argument('--no_of_local_regions', type=int, default=13)

#no. of decoder blocks used. Here, 1 means 1 decoder block used, 2 is for 2 blocks,..., 5 is for all blocks aka full decoder.
parser.add_argument('--no_of_decoder_blocks', type=int, default=1)

#local_reg_size - 1 for 3x3 local region size in the feature map. <local_reg> -> flat -> w*flat -> 128 bit z vector matching;
#               - 0 for 1x1 local region size in the feature map
parser.add_argument('--local_reg_size', type=int, default=1)
#wgt_en - 1 for having extra weight layer on top of 'z' vector from local region.
#      - 0 for not having any weight layer.
parser.add_argument('--wgt_en', type=int, default=1)


#no. of neighbouring local regions sampled from the feature maps to act as negative samples in local contrastive loss
# for a given positive local region - currently 5 local regions are chosen from each feature map (due to memory issues).
parser.add_argument('--no_of_neg_local_regions', type=int, default=5)

#overide the no. of negative (-ve) local neighbouring regions chosen for local loss computation- 4 for L^{D} (local_loss_exp_no=1) - due to memory issues
parser.add_argument('--no_of_neg_regs_override', type=int, default=4)

#no of iterations to run
parser.add_argument('--n_iter', type=int, default=5001)

parser.add_argument('--split', type=int, default=100)

parser.add_argument('--mc_iterations', type=int, default=500)

parser.add_argument('--output_dir', type=str, default="/scratcg/st-rohling-1/test")

parse_config = parser.parse_args()
# wandb.config.update(parse_config)
#parse_config = parser.parse_args(args=[])

if parse_config.dataset == 'acdc':
    print('load acdc configs')
    import experiment_init.init_acdc as cfg
    import experiment_init.data_cfg_acdc as data_list
elif parse_config.dataset == 'mmwhs':
    print('load mmwhs configs')
    import experiment_init.init_mmwhs as cfg
    import experiment_init.data_cfg_mmwhs as data_list
elif parse_config.dataset == 'prostate_md':
    print('load prostate_md configs')
    import experiment_init.init_prostate_md as cfg
    import experiment_init.data_cfg_prostate_md as data_list
elif parse_config.dataset == 'kidney_cap':
    print('load kidney_cap configs')
    import experiment_init.init_kidney_capsule as cfg
    import experiment_init.data_cfg_kidney as data_list
elif parse_config.dataset == 'kidney_reg':
    print('load kidney_reg configs')
    import experiment_init.init_kidney_regions as cfg
    import experiment_init.data_cfg_kidney as data_list
else:
    raise ValueError(parse_config.dataset)

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg,False)

if parse_config.dataset == 'acdc':
    print('set acdc orig img dataloader handle')
    orig_img_dt=dt.load_acdc_imgs
elif parse_config.dataset == 'mmwhs':
    print('set mmwhs orig img dataloader handle')
    orig_img_dt=dt.load_mmwhs_imgs
elif parse_config.dataset == 'prostate_md':
    print('set prostate_md orig img dataloader handle')
    orig_img_dt=dt.load_prostate_imgs_md
elif parse_config.dataset == 'kidney_cap' or parse_config.dataset == 'kidney_reg':
    print('set kidney_cap orig img dataloader handle')
    orig_img_dt=dt.load_kidney_imgs
    if parse_config.dataset == 'kidney_cap':
        sub_folder = "capsule"
    else:
        sub_folder = "regions"

#  load model object
from models import modelObj
model = modelObj(cfg)
#  load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

if(parse_config.rd_en==1):
    parse_config.en_1hot=1
else:
    parse_config.en_1hot=0

struct_name=cfg.struct_name
val_step_update=cfg.val_step_update

######################################
# Define final U-net model & directory to save - for segmentation task
#######################################
#define directory to save fine-tuned model
save_dir=str(cfg.srt_dir)+'/augmodelsfull/'+'split_'+str(parse_config.split)+'/'+str(parse_config.dataset)+'/trained_models/fine_tune_on_pretrained_encoder_and_decoder_net/'

if(parse_config.pretr_data_aug==0):
    save_dir=str(save_dir)+'/no_data_aug/'
elif(parse_config.pretr_data_aug==2):
    save_dir=str(save_dir)+'/with_data_aug/'
elif(parse_config.pretr_data_aug==3):
    save_dir=str(save_dir)+'/with_data_aug_sha/'
elif(parse_config.pretr_data_aug==4):
    save_dir=str(save_dir)+'/with_data_aug_nak/'
elif(parse_config.pretr_data_aug==5):
    save_dir=str(save_dir)+'/with_data_aug_all/'
elif(parse_config.pretr_data_aug==6):
    save_dir=str(save_dir)+'/with_data_aug_dep/'
elif(parse_config.pretr_data_aug==7):
    save_dir=str(save_dir)+'/with_data_aug_tgc/'

if(parse_config.rd_en==1 and parse_config.ri_en==1):
    save_dir=str(save_dir)+'rand_deforms_and_ints_en/'
elif(parse_config.rd_en==1):
    save_dir=str(save_dir)+'rand_deforms_en/'
elif(parse_config.ri_en==1):
    save_dir=str(save_dir)+'rand_ints_en/'

save_dir=str(save_dir)+'global_loss_exp_no_'+str(parse_config.global_loss_exp_no)+'_local_loss_exp_no_'+str(parse_config.local_loss_exp_no) \
             +'_n_parts_'+str(parse_config.n_parts)+'/'

save_dir=str(save_dir)+'temp_fac_'+str(parse_config.temp_fac)+'/'
save_dir=str(save_dir)+'enc_bbox_dim_'+str(parse_config.pretr_cont_bbox_dim)+'/'

if(parse_config.local_reg_size==1):
    if(parse_config.wgt_en==1):
        save_dir=str(save_dir)+'local_reg_size_3x3_wgt_en/'
    else:
        save_dir=str(save_dir)+'local_reg_size_3x3_wgt_dis/'
else:
    save_dir=str(save_dir)+'local_reg_size_1x1_wgt_dis/'

save_dir=str(save_dir)+'no_of_decoder_blocks_'+str(parse_config.no_of_decoder_blocks)+'/'

save_dir=str(save_dir)+'no_of_local_regions_'+str(parse_config.no_of_local_regions)

if(parse_config.local_loss_exp_no==1):
    parse_config.no_of_neg_regs_override=4
    save_dir=save_dir+'_no_of_neg_regions_'+str(parse_config.no_of_neg_regs_override)+'/'
else:
    save_dir=save_dir+'_no_of_neg_regions_'+str(parse_config.no_of_neg_local_regions)+'/'

save_dir=str(save_dir)+'last_ep_model/'
    
save_dir=str(save_dir)+str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/unet_dsc_'+str(parse_config.dsc_loss)+'_n_iter_'+str(parse_config.n_iter)+'_lr_seg_'+str(parse_config.lr_seg)+'/'

print('save dir ',save_dir)
######################################

######################################
tf.reset_default_graph()
# Segmentation Network
ae = model.seg_unet(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,en_1hot=parse_config.en_1hot,mtask_en=0)

# define network/graph to apply random deformations on input images
ae_rd = model.deform_net(batch_size=cfg.mtask_bs)

# define network/graph to apply random contrast and brightness on input images
ae_rc = model.contrast_net(batch_size=cfg.mtask_bs)

# define graph to compute 1-hot encoding of segmentation mask
ae_1hot = model.conv_1hot()
######################################

######################################
# Define checkpoint file to save CNN network architecture and learnt hyperparameters
checkpoint_filename='fine_tune_trained_encoder_and_decoder_net_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)
######################################


# get test volumes id list
print('get test volumes list')
test_list = data_list.test_data()

######################################

######################################
# find best model checkpoint over all epochs and restore it
mp_best=get_max_chkpt_file(save_dir)
print('mp_best',mp_best)

saver = tf.train.Saver()
sess = tf.Session(config=config)
saver.restore(sess, mp_best)
print("Model restored")
#####################################

# infer predictions over test volumes from the best model saved during training
save_dir_tmp=(save_dir+'/test_set_predictions/uncertainty2/' + sub_folder).replace("kidney_reg", "kidney_cap")
print(save_dir_tmp)
f1_util.net_monte_carlo(test_list,sess,ae,dt,orig_img_dt,save_dir_tmp,parse_config.dataset,parse_config.mc_iterations)

sess.close()
tf.reset_default_graph()
######################################
