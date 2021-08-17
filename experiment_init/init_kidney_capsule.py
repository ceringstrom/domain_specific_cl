################################################################
# Definitions required for CNN graph
################################################################
#Filter size at different depth level of CNN in order
fs=3
#Interpolation type for upsampling layers in decoder
interp_val=1 # 0 - bilinear interpolation; 1- nearest neighbour interpolation
################################################################

################################################################
# data dimensions, num of classes and resolution
################################################################
#Name of dataset
dataset_name='kidney_capsule'
#Image Dimensions
img_size_x = 256
img_size_y = 192
# Images dimensions in one-dimensional array
img_size_flat = img_size_x * img_size_y
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1
# Number of label classes
num_classes=2
#Image dimensions in x and y directions
size=(img_size_x,img_size_y)
#target image resolution
target_resolution=(4,4)
#label class name
class_name='cap'

zoom=0.5
################################################################
#data paths
################################################################
#validation_update_step to save values
val_step_update=1
#base directory of the code
base_dir='/scratch/st-rohling-1/contrastive_learning/'
srt_dir='/scratch/st-rohling-1/contrastive_learning/domain_specific_cl/domain_specific_cl/'

#Path to data in original dimensions in default resolution
data_path_pretr='/arc/project/st-rohling-1/data/kidney/CL_data/dscl_data/unlabelled/'
data_path_tr='/arc/project/st-rohling-1/data/kidney/CL_data/dscl_data/Task001_KidneyCapsule/'


#Path to data in cropped dimensions in target resolution (saved apriori)
data_path_pretr_cropped='/arc/project/st-rohling-1/data/kidney/CL_data/dscl_data/unlabelled_cropped/'
data_path_tr_cropped='/arc/project/st-rohling-1/data/kidney/CL_data/dscl_data/Task001_KidneyCapsule_Cropped/'
################################################################

################################################################
#training hyper-parameters
################################################################
#learning rate for segmentation net
lr=0.001
#pre-training batch size
mtask_bs=20
#batch_size for fine-tuning on segmentation task
batch_size_ft=16
#foreground structures names to segment
struct_name=['Capsule']
