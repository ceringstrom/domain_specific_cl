import numpy as np
import scipy.ndimage.interpolation
from scipy.stats import halfnorm

from skimage import transform
from skimage.measure import label
from skimage.morphology import convex_hull_image
from scipy.stats import nakagami
import random

import os
import re
import cv2
import math


def augmentation_function(ip_list, dt, labels_present=1, en_1hot=0):
    '''
    To generate affine augmented image,label pairs.

    ip params:
        ip_list: list of 2D slices of images and its labels if labels are present
        dt: dataloader object
        labels_present: to indicate if labels are present or not
        en_1hot: to indicate labels are used in 1-hot encoding format
    returns:
        sampled_image_batch : augmented images generated
        sampled_label_batch : corresponding augmented labels
    '''

    if(len(ip_list)==2 and labels_present==1):
        images = ip_list[0]
        labels = ip_list[1]
    else:
        images=ip_list[0]

    if images.ndim > 4:
        raise AssertionError('Augmentation will only work with 2D images')

    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for index in range(num_images):

        img = np.squeeze(images[index,...])
        if(labels_present==1):
            lbl = np.squeeze(labels[index,...])

        do_rotations,do_scaleaug,do_fliplr,do_simple_rot=0,0,0,0
        #option 5 is to not perform any augmentation i.e, use the original image
        #randomly select the augmentation to apply of the options stated below.
        aug_select = np.random.randint(5)

        if(np.max(img)>0.001):
            if(aug_select==0):
                do_rotations=1
            elif(aug_select==1):
                do_scaleaug=1
            elif(aug_select==2):
                do_fliplr=1
            elif(aug_select==3):
                do_simple_rot=1

        # ROTATE between angle -15 to 15
        if do_rotations:
            angles = [-15,15]
            random_angle = np.random.uniform(angles[0], angles[1])
            img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1)
            if(labels_present==1):
                if(en_1hot==1):
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=1)
                else:
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        # RANDOM SCALE
        if do_scaleaug:
            n_x, n_y = img.shape
            #scale factor between 0.95 and 1.05
            scale_fact_min=0.95
            scale_fact_max=1.05
            scale_val = round(random.uniform(scale_fact_min,scale_fact_max), 2)
            slice_rescaled = transform.rescale(img, scale_val, order=1, preserve_range=True, mode = 'constant')
            img = dt.crop_or_pad_slice_to_size(slice_rescaled, n_x, n_y)
            if(labels_present==1):
                if(en_1hot==1):
                    slice_rescaled = transform.rescale(lbl, scale_val, order=1, preserve_range=True, mode = 'constant')
                    lbl = dt.crop_or_pad_slice_to_size_1hot(slice_rescaled, n_x, n_y)
                else:
                    slice_rescaled = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode = 'constant')
                    lbl = dt.crop_or_pad_slice_to_size(slice_rescaled, n_x, n_y)

        # RANDOM FLIP
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.fliplr(img)
                if(labels_present==1):
                    lbl = np.fliplr(lbl)

        # Simple rotations at angles of 45 degrees
        if do_simple_rot:
            fixed_angle = 45
            random_angle = np.random.randint(8)*fixed_angle

            img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1)
            if(labels_present==1):
                if(en_1hot==1):
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=1)
                else:
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        new_images.append(img[..., np.newaxis])
        if(labels_present==1):
            new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    if(labels_present==1):
        sampled_label_batch = np.asarray(new_labels)

    if(len(ip_list)==2 and labels_present==1):
        return sampled_image_batch, sampled_label_batch
    else:
        return sampled_image_batch

def calc_deform(cfg,batch_size, mu=0,sigma=10, order=3):
    '''
    To generate a batch of smooth deformation fields for the specified mean and standard deviation value.

    input params:
        cfg: experiment config parameter (contains image dimensions, batch_size, etc)
        mu: mean value for the normal distribution
        sigma: standard deviation value for the normal distribution
        order: order of interpolation; 3 = bicubic interpolation
    returns:
        flow_vec: batch of deformation fields generated
    '''

    flow_vec = np.zeros((batch_size,cfg.img_size_x,cfg.img_size_y,2))

    for i in range(batch_size):
        #mu, sigma = 0, 10 # mean and standard deviation
        dx = np.random.normal(mu, sigma, 9)
        dx_mat = np.reshape(dx,(3,3))
        dx_img = transform.resize(dx_mat, output_shape=(cfg.img_size_x,cfg.img_size_y), order=order,mode='reflect')

        dy = np.random.normal(mu, sigma, 9)
        dy_mat = np.reshape(dy,(3,3))
        dy_img = transform.resize(dy_mat, output_shape=(cfg.img_size_x,cfg.img_size_y), order=order,mode='reflect')


        flow_vec[i,:,:,0] = dx_img
        flow_vec[i,:,:,1] = dy_img

    return flow_vec

def shuffle_minibatch(ip_list, batch_size=20,num_channels=1,labels_present=1,axis=2):
    '''
    To sample a minibatch images of batch_size from all the available 3D volumes.

    input params:
        ip_list: llist of 3d volumes and its labels if present
        batch_size: number of 2D slices to consider for the training
        labels_present: to indicate labels are used in 1-hot encoding format
        num_channels : no of channels of the input image
        axis : the axis along which we want to sample the minibatch -> axis vals : 0 - for sagittal, 1 - for coronal, 2 - for axial
    returns:
        image_data_train_batch: concatenated 2D slices randomly chosen from the total input data
        label_data_train_batch: concatenated 2D slices of labels with indices corresponding to the input data selected.
    '''

    if(len(ip_list)==2 and labels_present==1):
        image_data_train = ip_list[0]
        label_data_train = ip_list[1]
    else:
        image_data_train=ip_list[0]

    img_size_x=image_data_train.shape[0]
    img_size_y=image_data_train.shape[1]
    img_size_z=image_data_train.shape[2]

    len_of_train_data=np.arange(image_data_train.shape[axis])

    #randomize=np.random.choice(len_of_train_data,size=len(len_of_train_data),replace=True)
    randomize=np.random.choice(len_of_train_data,size=batch_size,replace=True)

    count=0
    for index_no in randomize:
        if(axis==2):
            img_train_tmp=np.reshape(image_data_train[:,:,index_no],(1,img_size_x,img_size_y,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[:,:,index_no],(1,img_size_x,img_size_y))

        elif(axis==1):
            img_train_tmp=np.reshape(image_data_train[:,index_no,:,],(1,img_size_x,img_size_z,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[:,index_no,:],(1,img_size_x,img_size_z))

        else:
            img_train_tmp=np.reshape(image_data_train[index_no,:,:],(1,img_size_y,img_size_z,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[index_no,:,:],(1,img_size_y,img_size_z))


        if(count==0):
            image_data_train_batch=img_train_tmp
            if(labels_present==1):
                label_data_train_batch=label_train_tmp

        else:
            image_data_train_batch=np.concatenate((image_data_train_batch, img_train_tmp),axis=0)
            if(labels_present==1):
                label_data_train_batch=np.concatenate((label_data_train_batch, label_train_tmp),axis=0)

        count=count+1
        if(count==batch_size):
            break

    if(len(ip_list)==2 and labels_present==1):
        return image_data_train_batch, label_data_train_batch
    else:
        return image_data_train_batch

def shuffle_minibatch_mtask(ip_list, batch_size=20,num_channels=1,labels_present=1,axis=2):
    '''
        To sample a minibatch images of batch_size from all the available 3D volumes.

        input params:
            ip_list: list of 3d volumes and its labels if present
            batch_size: number of 2D slices to consider for the training
            labels_present: to indicate labels are used in 1-hot encoding format
            num_channels : no of channels of the input image
            axis : the axis along which we want to sample the minibatch -> axis vals : 0 - for sagittal, 1 - for coronal, 2 - for axial
        returns:
            image_data_train_batch: concatenated 2D slices randomly chosen from the total input data
            label_data_train_batch: concatenated 2D slices of labels with indices corresponding to the input data selected.
        '''
    is_speckle = False

    if(len(ip_list)==2 and labels_present==1):
        image_data_train = ip_list[0]
        label_data_train = ip_list[1]
    elif(len(ip_list)==1 and labels_present==0):
        image_data_train = ip_list[0]
    elif(len(ip_list)==3):
        image_data_train = ip_list[0]
        label_data_train = ip_list[1]
        speckle_data_train = ip_list[2]
        is_speckle = True

    if(num_channels==1):
        img_size_x=image_data_train.shape[0]
        img_size_y=image_data_train.shape[1]
    else:
        img_size_x=image_data_train.shape[1]
        img_size_y=image_data_train.shape[2]

    len_of_train_data=np.arange(image_data_train.shape[axis])
    #randomize=np.random.choice(len_of_train_data,size=len(len_of_train_data),replace=true)
    randomize=np.random.choice(len_of_train_data,size=batch_size,replace=True)

    count=0
    for index_no in randomize:
        if(num_channels==1):
            img_train_tmp=np.reshape(image_data_train[:,:,index_no],(1,img_size_x,img_size_y,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[:,:,index_no],(1,img_size_x,img_size_y))
        else:
            img_train_tmp = np.reshape(image_data_train[index_no], (1, img_size_x, img_size_y, num_channels))
            if(labels_present==1):
                label_train_tmp = np.reshape(label_data_train[index_no], (1, img_size_x, img_size_y))
        if(is_speckle):
            speckle_train_tmp = np.reshape(speckle_data_train[index_no], (1, 3, 2))

        if(count==0):
            image_data_train_batch=img_train_tmp
            if(labels_present==1):
                label_data_train_batch=label_train_tmp
            if(is_speckle):
                speckle_data_train_batch = speckle_train_tmp
        else:
            image_data_train_batch=np.concatenate((image_data_train_batch, img_train_tmp),axis=0)
            if(labels_present==1):
                label_data_train_batch=np.concatenate((label_data_train_batch, label_train_tmp),axis=0)
            if(is_speckle):
                speckle_data_train_batch = np.concatenate((speckle_data_train_batch, speckle_train_tmp), axis=0)

        count=count+1
        if(count==batch_size):
            break
    if(is_speckle):
        return image_data_train_batch, label_data_train_batch, speckle_data_train_batch
    elif(labels_present==1):
        return image_data_train_batch, label_data_train_batch
    else:
        return image_data_train_batch

def change_axis_img(ip_list, labels_present=1, def_axis_no=2, cat_axis=0):
    '''
    To swap the axes of 3D volumes as per the network input
    input params:
        ip_list: list of 3D volumes and its labels if labels are present
        labels_present: to indicate if labels are present or not
        def_axis_no: axis which needs to be swapped (default is axial direction here)
        cat_axis: axis along which the images need to concatenated
    returns:
        mergedlist_img: swapped axes 3D volumes
        mergedlist_labels: corresponding swapped 3D volumes
    '''
    # Swap axes of 3D volume according to the input of the network
    if(len(ip_list)==2 and labels_present==1):
        labeled_data_imgs = ip_list[0]
        labeled_data_labels = ip_list[1]
    else:
        labeled_data_imgs=ip_list[0]

    #can also define in an init file - base values
    img_size_x=labeled_data_imgs.shape[0]
    img_size_y=labeled_data_imgs.shape[1]

    total_slices = labeled_data_imgs.shape[def_axis_no]
    for slice_no in range(total_slices):

        img_test_slice = np.reshape(labeled_data_imgs[:, :, slice_no], (1, img_size_x, img_size_y, 1))
        if(labels_present==1):
            label_test_slice = np.reshape(labeled_data_labels[:, :, slice_no], (1, img_size_x, img_size_y))

        if (slice_no == 0):
            mergedlist_img = img_test_slice
            if(labels_present==1):
                mergedlist_labels = label_test_slice

        else:
            mergedlist_img = np.concatenate((mergedlist_img, img_test_slice), axis=cat_axis)
            if(labels_present==1):
                mergedlist_labels = np.concatenate((mergedlist_labels, label_test_slice), axis=cat_axis)

    if(len(ip_list)==2 and labels_present==1):
        return mergedlist_img,mergedlist_labels
    else:
        return mergedlist_img

def load_val_imgs(val_list,dt,orig_img_dt):
    '''
    To load validation ACDC/Prostate/MMWHS images and its labels, pixel resolution list
    input params:
        val_list: list of validation patient ids of the dataset
        dt: dataloader object
        orig_img_dt: dataloader for the image
    returns:
        val_label_orig: returns list of labels without any pre-processing applied
        val_img_re: returns list of images post pre-processing steps done
        val_label_re: returns list of labels post pre-processing steps done
        pixel_val_list: returns list of pixel resolution values of original images
    '''
    val_label_orig=[]
    val_img_list=[]
    val_label_list=[]
    pixel_val_list=[]

    for val_id in val_list:
        val_id_list=[val_id]
        val_img,val_label,pixel_size_val=orig_img_dt(val_id_list)
        #pre-process the image into chosen resolution and dimensions
        val_cropped_img,val_cropped_mask = dt.preprocess_data(val_img, val_label, pixel_size_val)

        #change axis for computation of dice score
        val_img_re,val_labels_re= change_axis_img([val_cropped_img,val_cropped_mask])

        val_label_orig.append(val_label)
        val_img_list.append(val_img_re)
        val_label_list.append(val_labels_re)
        pixel_val_list.append(pixel_size_val)

    return val_label_orig,val_img_list,val_label_list,pixel_val_list

def estimateNakagami(im):

    #making arrays to compute expectations as per nakagami estimates
    #careful for overflows, need to declare python int in case
    
    # im = im.astype(np.int64)
    e_x2 = 0
    e_x4 = 0
    rows,cols = im.shape
    N = rows*cols
    for x in range(0,rows):
        for y in range(0,cols):
            e_x2 = e_x2 + (im[x,y])**2
            e_x4 = e_x4 + (im[x,y])**4
    if N > 0:     
        e_x2 = e_x2 / N
        e_x4 = e_x4 / N
    else:
        e_x2 = 0
        e_x4 = 0
    
    nakScale = e_x2
    #using inverse normalized variance esimator for Nakagami
    if(( e_x4 - (e_x2**2)) == 0):
        print("Setting 0")
        nakShape = 0
    else:
        nakShape = e_x2**2 / ( e_x4 - (e_x2**2))
    
    return np.nan_to_num([nakShape, nakScale])

def compute_img_distributions(img_list, label_list):
    params = np.zeros((img_list.shape[-1], 3, 2))
    for i in range(img_list.shape[-1]):
        label = label_list[:, :, i]
        img = img_list[:, :, i]
        for j in range(1, 4):
            mask = (label == j).astype("int8")
            nonzero_indices = np.nonzero(mask)
            values = np.expand_dims(img[nonzero_indices], axis=0)
            if values.shape[1] != 0 :
                params[i, j-1] = estimateNakagami(values)
    return params

def get_max_chkpt_file(model_path,min_ep=10):
    '''
    To return the checkpoint file that yielded the best dsc value/lowest loss value on val images
    input params:
        model_path: directory of the experiment where the checkpoint files are stored
        min_ep: variable to ensure that the model selected has higher epoch no. than this no. (here its 10).
    returns:
        fin_chkpt_max: checkpoint file with best dsc value
    '''
    print(model_path)
    for dirName, subdirList, fileList in os.walk(model_path):
        fileList.sort()
        for filename in fileList:
            #print('1',filename)
            if ".meta" in filename.lower() and 'best_model' in filename:
                numbers = re.findall('\d+',filename)
                #print('0',filename,numbers,numbers[0],numbers[1],min_ep)
                if "_v2" in filename:
                    tmp_ep_no=int(numbers[1])
                else:
                    tmp_ep_no=int(numbers[0])
                if(tmp_ep_no>min_ep):
                    chkpt_max=os.path.join(dirName,filename)
                    min_ep=tmp_ep_no
    fin_chkpt_max = re.sub('\.meta$', '', chkpt_max)
    return fin_chkpt_max

def get_chkpt_file(model_path,match_name='',min_ep=10):
    '''
        To return the "last epoch number" over all epochs checkpoint file on val images
        input params:
            model_path: directory of the experiment where the checkpoint files are stored
            min_ep: variable to ensure that the model selected has higher epoch no. than this no. (here its 10).
        returns:
            fin_chkpt_max: checkpoint file with last epoch number over all epochs
        '''
    print(model_path)
    for dirName, subdirList, fileList in os.walk(model_path):
        fileList.sort()
        #min_ep=10
        print("HELLO")
        print(fileList)
        for filename in fileList:
            if ".meta" in filename.lower():
                numbers = re.findall('\d+',filename)
                print(filename)
                print(numbers)
                if numbers:
                    numbers = numbers
                else:
                    numbers = [0]
                #print('model_path',model_path,filename)
                #print('0',filename,numbers,numbers[0],min_ep)
                #print('match name',match_name)
                chkpt_max=os.path.join(dirName,filename)
                if(isNotEmpty(match_name)):
                    if(match_name in filename and '00000-of-00001' not in filename and int(numbers[0])>min_ep):
                        print('1')
                        chkpt_max=os.path.join(dirName,filename)
                elif(int(numbers[0])>min_ep):
                    print('2')
                    chkpt_max=os.path.join(dirName,filename)
                    min_ep=int(numbers[0])

    # print(chkpt_max)
    # chkpt_max=os.path.join(model_path,"train_decoder_wgts_kidney_cap_epochs_999.ckpt.meta")
    fin_chkpt_max = re.sub('\.meta$', '', chkpt_max)
    #print(fin_chkpt_max)
    return fin_chkpt_max

def isNotEmpty(s):
    '''
        To check if file exists in a directory
    '''
    return bool(s and s.strip())

def mixup_data_gen(x_train,y_train,alpha=0.1):
    '''
    # Generator for mixup data - to linearly combine 2 random image,label pairs from the batch of image,label pairs
    input params:
        x_train: batch of input images
        y_train: batch of input labels
        alpha: alpha value (mixing co-efficient value)
    returns:
        x_out: linearly combined resultant image
        y_out: linearly combined resultant label
    '''
    len_x_train = x_train.shape[0]
    x_out=np.zeros_like(x_train)
    y_out=np.zeros_like(y_train)

    for i in range(len_x_train):
        lam = np.random.beta(alpha, alpha)
        rand_idx1 = np.random.choice(len_x_train)
        rand_idx2 = np.random.choice(len_x_train)

        x_out[i] = lam * x_train[rand_idx1] + (1 - lam) * x_train[rand_idx2]
        y_out[i] = lam * y_train[rand_idx1] + (1 - lam) * y_train[rand_idx2]

    return x_out, y_out

def create_rotated_imgs(x_train,batch_size,cfg):
    '''
    # randomly apply rotation on (image,label) pairs - out of 4 values (0,90,180,270) degrees
    input params:
        x_train: batch of input images
        batch_size: batch size
    returns:
        image_data_train_batch: rotated images
        label_data_train_batch: rotated index/label out of 0 to 3 (0 for 0deg, 1 for 90deg, 2 for 180deg, 3 for 270deg)
    '''

    #randomize = np.random.choice(len_x_train, size=batch_size, replace=False)
    label_data_train_batch=np.zeros(batch_size)
    count=0
    rot_index_no=0
    for ind in range(0,batch_size):
        #index_no=randomize[count]
        index_no=count
        img_train_tmp=np.reshape(np.rot90(x_train[index_no,:,:],rot_index_no),(1,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))
        #label_train_tmp=np.reshape(np.asarray(rot_index_no),(1,1))
        #label_train_tmp=np.reshape(rot_index_no,(1,1))
        label_train_tmp=rot_index_no

        rot_index_no = rot_index_no + 1
        if(rot_index_no==4):
            rot_index_no=0
            count=count+1

        if(ind==0):
            image_data_train_batch=img_train_tmp
            label_data_train_batch[ind]=label_train_tmp
        else:
            image_data_train_batch=np.concatenate((image_data_train_batch, img_train_tmp),axis=0)
            #label_data_train_batch=np.concatenate((label_data_train_batch, label_train_tmp),axis=0)
            label_data_train_batch[ind]=label_train_tmp
        #count=count+1
        if(ind==batch_size-1):
            break

    return image_data_train_batch, label_data_train_batch

def stitch_two_crop_batches(ip_list,cfg,batch_size):
    '''
    # stitch 2 batches of (image,label) pairs with different augmentations applied on the same set of original (image,label) pair
    input params:
        ip_list: list of 2 set of (image,label) pairs with different augmentations applied
        cfg : contains config settings of the image
        batch_size: batch size of final stitched set
    returns:
        cat_img_batch: stitched set of 2 batches of images under different augmentations
        cat_lbl_batch: stitched set of 2 batches of labels under different augmentations
    '''

    if(len(ip_list)==4):
        img_batch1=ip_list[0]
        lbl_batch1=ip_list[1]
        img_batch2=ip_list[2]
        lbl_batch2=ip_list[3]
        cat_img_batch=np.zeros((2*batch_size,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))
        cat_lbl_batch=np.zeros((2*batch_size,cfg.img_size_x,cfg.img_size_y))
    else:
        img_batch1=ip_list[0]
        img_batch2=ip_list[1]
        cat_img_batch=np.zeros((2*batch_size,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))

    for index in range(0,2*batch_size,2):
        cat_img_batch[index]  =img_batch1[int(index/2)]
        cat_img_batch[index+1]=img_batch2[int(index/2)]
        #print(int(index/2),index,index+1)
        if(len(ip_list)==4):
            cat_lbl_batch[index]  =lbl_batch1[int(index/2)]
            cat_lbl_batch[index+1]=lbl_batch2[int(index/2)]

    if(len(ip_list)==4):
        return cat_img_batch,cat_lbl_batch
    else:
        return cat_img_batch

def crop_batch(ip_list,cfg,batch_size,box_dim=100,box_dim_y=100,low_val=10,high_val=70):
    '''
    To select a cropped part of the image and resize it to original dimensions
    input param:
        ip_list: input list of image, labels
        cfg: contains config settings of the image
        batch_size: batch size value
        box_dim_x,box_dim_y: co-ordinates of the cropped part of the image to be select and resized to original dimensions
        low_val : lowest co-ordinate value allowed as starting point of the cropped window
        low_val : highest co-ordinate value allowed as starting point of the cropped window
    return params:
        ld_img_re_bs: cropped images that are resized into original dimensions
        ld_lbl_re_bs: cropped masks that are resized into original dimensions

    '''
    #ld_label_batch = np.squeeze(np.zeros_like(ld_img_batch))
    #box_dim = 100  # 100*100
    if(len(ip_list)==2):
        ld_img_batch=ip_list[0]
        ld_label_batch=ip_list[1]
        ld_img_re_bs=np.zeros_like(ld_img_batch)
        ld_lbl_re_bs=np.zeros_like(ld_label_batch)
    else:
        ld_img_batch=ip_list[0]
        ld_img_re_bs=np.zeros_like(ld_img_batch)

    x_dim=cfg.img_size_x
    y_dim=cfg.img_size_y

    box_dim_arr_x=np.random.randint(low=low_val,high=high_val,size=batch_size)
    box_dim_arr_y=np.random.randint(low=low_val,high=high_val,size=batch_size)

    for index in range(0, batch_size):
        #inpaint = np.ones(ld_img_batch[index, :, :, 0].shape)
        #x = int(np.random.randint(cfg.img_size_x - box_dim_arr_x[index], size=1))
        #y = int(np.random.randint(cfg.img_size_y - box_dim_arr_y[index], size=1))
        #inpaint[x:x + box_dim, y:y + box_dim] = 0
        #ld_label_batch[index]=ld_img_batch[index, :, :, 0]
        #ld_img_batch[index, :, :, 0] = ld_img_batch[index, :, :, 0] * inpaint
        x,y=box_dim_arr_x[index],box_dim_arr_y[index]
        if(len(ip_list)==2):
            im_crop = ld_img_batch[index,x:x + box_dim, y:y + box_dim_y,0]
            ld_img_re_bs[index,:,:,0]=transform.resize(im_crop,(x_dim,y_dim),order=1)
            lbl_crop = ld_label_batch[index,x:x + box_dim, y:y + box_dim_y]
            ld_lbl_re_bs[index]=transform.resize(lbl_crop,(x_dim,y_dim),order=0)
        else:
            im_crop = ld_img_batch[index,x:x + box_dim, y:y + box_dim_y,0]
            ld_img_re_bs[index,:,:,0]=transform.resize(im_crop,(x_dim,y_dim),order=1)

    if(len(ip_list)==2):
        return ld_img_re_bs,ld_lbl_re_bs
    else:
        return ld_img_re_bs

def find_mask(img, threshold=1/800, index=-2):
    v = img
    # print(img.shape)
    diff_mean = (np.clip(v, 0, 1))
    # find connected components after threshold
    cc = label(diff_mean > threshold)
    v, c = np.unique(cc, return_counts=True)
    # find second largest connect component (largest is the background)
    second_largest_component = v[c.tolist().index(sorted(c)[index])]

    # take convex hull to remove small gaps
    # noinspection PyTypeChecker
    return convex_hull_image(np.where(cc == second_largest_component, 1, 0))

def zoom_batch(ip_list,cfg,batch_size):
    # print("ZOOM")
    zoom_scale = halfnorm.rvs(loc=1, scale=cfg.zoom, size=batch_size)
    zoomed_batch = np.zeros_like(ip_list)
    for i in range(batch_size):
        img = ip_list[i]
        scaled_img = img / np.max(img)
        scale = zoom_scale[i]
        img = np.squeeze(img)
        scaled_img = np.squeeze(scaled_img)
        mask = find_mask(scaled_img)
        total = np.product(mask.shape)
        covering = mask.sum() / total
        com = scipy.ndimage.measurements.center_of_mass(mask)

        if covering < .15:
            mask = find_mask(scaled_img, 1/800 / 2)
        if covering > .9 or com[0] < mask.shape[0]/4:
            mask = find_mask(scaled_img, 1/800, -1)
        # mask = mask_list[i]
        com = scipy.ndimage.measurements.center_of_mass(mask)
        img_shape = img.shape
        r_offset = min(max(0, int((scale * com[0] - img.shape[0]) / 2)), int((scale-1) * img.shape[0]))
        c_offset = min(max(0, int((scale * com[1] - img.shape[1]) / 2)), int((scale-1) * img.shape[1]))

        zoomed = scipy.ndimage.zoom(img, scale)

        new_image = mask * zoomed[r_offset:r_offset + img.shape[0], c_offset:c_offset + img.shape[1]]
        new_image = np.expand_dims(new_image, axis=2)

        zoomed_batch[i] = new_image

    return zoomed_batch

def get_scanlines(img):
    left_line = []
    right_line = []

    start_row = 0
    h, w = img.shape
    min_col = w
    for i in range(h):
        row = img[i, :]
        if(np.count_nonzero(row)) > 1:
            nonzero = row.nonzero()
            if start_row==0:
                start_row = i
            if nonzero[0][0] < min_col:
                min_col = nonzero[0][0]
            if nonzero[0][0] > min_col + 5:
                break
            left_line.append(nonzero[0][0])
            right_line.append(nonzero[0][-1])
    y = np.arange(start_row, start_row+len(left_line))
    left = np.polyfit(np.array(left_line),y,1)
    right = np.polyfit(np.array(right_line),y,1)
    xi = (left[1]-right[1])/(right[0]-left[0])
    yi = left[0] * xi + left[1]

    slopes = (left, right)
    
    radius = img.shape[0] - yi
    center = left_line[0] + (right_line[0] - left_line[0])/2

    return center, radius, -yi, slopes

def get_triangle_mask(img_shape, center, radius, buffer, slopes, width=(0, 0.2)):
    h, w = img_shape
    
    m1 = slopes[0][0]
    m2 = slopes[1][0]
    
    theta_low = min(np.arctan(1/m1), np.arctan(1/m2))
    theta_high = max(np.arctan(1/m1), np.arctan(1/m2))
    
    if width[1] > theta_high - theta_low:
        width[1] = theta_high - theta_low
    
    width = np.random.uniform(low=width[0], high=width[1], size=1)
    position = np.random.uniform(low=theta_low+width/2, high=theta_high-width/2, size=1)
    
    low_slope = 1/np.tan(position-width/2)
    high_slope = 1/np.tan(position+width/2)
    
    low_intercept = -buffer - low_slope * center
    high_intercept = -buffer - high_slope * center
    
    lines = [(low_slope[0], low_intercept[0]), (high_slope[0], high_intercept[0])]
    contours = []
    for (slope, intercept) in lines:
        bottom_intercept = (h-intercept) / slope
        if intercept >= 0 and intercept <= h:
            contours.append([0, intercept])
        elif(bottom_intercept >= 0 and bottom_intercept <= w):
            contours.append([bottom_intercept, h])
        else:
            contours.append([w, slope * w + intercept])
    contours.append([-lines[1][1] / lines[1][0], 0])
    contours.append([-lines[0][1] / lines[0][0], 0])
    mask = np.zeros((h, w))
    cv2.fillPoly(mask, pts = [np.array(contours, dtype=np.int32)], color=(1))
    return mask

def get_bounded_nakagami(length, shape, scale, min_val=0, max_val=255):
    vals = np.zeros(length)
    for i in range(length):
        val = -1
        while val < min_val or val > max_val:
            val = nakagami.rvs(shape, scale=scale, size=1)
        vals[i] = val
    return vals

def shadow_batch(ip_list,cfg,batch_size):
    shadowed_batch = np.zeros_like(ip_list)
    for i in range(batch_size):
        img = ip_list[i]
        img = np.squeeze(img)
        img = np.swapaxes(img, 0, 1)
        try:
            # cv2.imwrite("test_{}.png".format(i), (img*255/ np.max(img)).astype("int8"))
            center, radius, buffer, slopes = get_scanlines(img*255)
            top = (img[:, int(center)]!=0).argmax() + int(buffer)
            mask = get_triangle_mask(img.shape, center, radius, int(buffer), slopes)
            num_pix = int(mask.sum())
            nak = get_bounded_nakagami(num_pix, 0.202, 189.3, min_val=0, max_val=255) / 255 * np.max(img)
            masked_image = img.copy()
            masked_image[np.where(mask != 0)] = nak
            outer_mask = find_mask(img, index=-2)
            masked_image = masked_image * outer_mask
        except Exception as e:
            masked_image = img
        # cv2.imwrite("test2_{}.png".format(i), (masked_image*255/ np.max(img)).astype("int8"))
        masked_image = np.swapaxes(masked_image, 0, 1)
        masked_image = np.expand_dims(masked_image, axis=-1)
        shadowed_batch[i] = masked_image
    return shadowed_batch

def depth_batch(ip_list,cfg,batch_size):
    zoom_scale = halfnorm.rvs(loc=1, scale=cfg.zoom, size=batch_size)
    zoomed_batch = np.zeros_like(ip_list)
    for i in range(batch_size):
        img = ip_list[i]
        img = np.swapaxes(img, 0, 1)
        # print(img.shape)
        # cv2.imwrite("test_{}.png".format(i), (img*255/ np.max(img)).astype("int8"))
        scaled_img = img / np.max(img)
        scale = zoom_scale[i]
        img = np.squeeze(img)
        scaled_img = np.squeeze(scaled_img)
        mask = find_mask(scaled_img)
        total = np.product(mask.shape)
        covering = mask.sum() / total
        com = scipy.ndimage.measurements.center_of_mass(mask)

        if covering < .15:
            mask = find_mask(scaled_img, 1/800 / 2)
        if covering > .9 or com[0] < mask.shape[0]/4:
            mask = find_mask(scaled_img, 1/800, -1)
        # mask = mask_list[i]
        com = scipy.ndimage.measurements.center_of_mass(mask)
        img_shape = img.shape
        # r_offset = min(max(0, int(scale * com[0] - img.shape[0] / 2)), zoomed.shape[0] - img.shape[0])

        zoomed = scipy.ndimage.zoom(img, scale, order=5, mode="mirror")
        # print("com", com)
        img_top = np.nonzero(img[:, int(com[1])])[0][0]
        zoom_top = np.nonzero(zoomed[:, int(com[1]*scale)])[0][0]

        c_offset = min(max(0, int(scale * com[1] - img.shape[1] / 2)), zoomed.shape[1] - img.shape[1])
        r_offset = int((scale-1) *  img_top)

        # print("zoomed")
        # print(np.max(img), np.min(img))
        # print(np.max(zoomed), np.min(zoomed))
        # print(c_offset)
        # print(r_offset, zoom_top, img_top, scale)

        new_image = mask * zoomed[r_offset:r_offset + img.shape[0], c_offset:c_offset + img.shape[1]]
        new_image = np.expand_dims(new_image, axis=2)

        # cv2.imwrite("test2_{}.png".format(i), (new_image*255/ np.max(new_image)).astype("int8"))

        new_image = np.swapaxes(new_image, 0, 1)

        zoomed_batch[i] = new_image

    return zoomed_batch

def apply_tgc(usimg, center, radius, buffer, regions=2, gain_range=(0.5, 2)):
    img = usimg.copy()
    maxval = np.max(img)
    gains = np.random.uniform(gain_range[0], gain_range[1], regions)
    # print(gains)
    nz_indices = np.nonzero(img)
    # print(np.sum(img==0.0))
    # cv2.imwrite("tgain.png", (img==0).astype("uint8")*255)
    radii = np.sqrt(np.square(nz_indices[0] + buffer) + np.square(nz_indices[1] - center))
    lower_limit = np.min(radii)
    upper_limit = np.max(radii)
    region_size = (upper_limit-lower_limit) / regions
    gain_array = np.zeros(radii.shape)
    # print((gain_array.dtype))
    current_pos = lower_limit
    for i in range(regions):
        gain_array[np.where(radii > current_pos)] = gains[i]
        current_pos += region_size
    # print("gains", np.max(gain_array), np.min(gain_array))
    # print(img.dtype)
    # img[nz_indices] = gain_array
    # cv2.imwrite("gain.png", (img*255/np.max(gains)).astype("uint8"))
    # print("monmax")
    # print(np.unique((img*255/np.max(gains)).astype("uint8"))) #, np.min((img*255/np.max(gains)).astype("uint8")))
    img[nz_indices] = gain_array * img[nz_indices]
    # print(np.unique(img).shape)
    img[img > maxval] = maxval
    return img

def tgc_batch(ip_list,cfg,batch_size):
    tgc_batch = np.zeros_like(ip_list)
    for i in range(batch_size):
        # print(i)
        img = ip_list[i]
        img = np.squeeze(img)
        img = np.swapaxes(img, 0, 1)
        try:
            # cv2.imwrite("tdest_{}.png".format(i), (img*255/ np.max(img)).astype("uint8"))
            
            center, radius, buffer, slopes = get_scanlines(img*255)
            # print(center, radius, buffer, slopes)
            gain_image = apply_tgc(img, center, radius, buffer, regions=10)
        except Exception as e:
            gain_image = img
        # print("uni", np.unique(gain_image*255.0/ np.max(gain_image)).astype("uint8"))
        # cv2.imwrite("tdest2_{}.png".format(i), (gain_image*255.0/ np.max(gain_image)).astype("uint8"))
        # print(np.max(), np.min())
        gain_image = np.swapaxes(gain_image, 0, 1)
        gain_image = np.expand_dims(gain_image, axis=-1)
        tgc_batch[i] = gain_image
    return tgc_batch

def create_inpaint_box(ld_img_batch,cfg,batch_size,box_dim=100,only_center_box=0):
    '''
    To create bounding boxes with pixels values set to 0 in the image
    input param:
        ld_img_batch: input batch of images
        cfg: contains config settings of the image
        batch_size: batch size value
        box_dim: dimensions of the bounding box applied on the image that will be set to zero. Ex: 100 denotes - 100x100 box of pixels are set to 0.
        only_center_box: to create inpaint box only in the center (1) or not (0) or (2) variable dimensions of box and variable co-ordinates location.
    return params:
        ld_img_batch: images with inpainted boxes where pixel values are set to 0.
    '''
    #ld_label_batch = np.squeeze(np.zeros_like(ld_img_batch))
    #box_dim = 100  # 100*100
    box_dim_arr_x=np.random.randint(low=30,high=90,size=batch_size)
    box_dim_arr_y=np.random.randint(low=30,high=90,size=batch_size)
    for index in range(0, batch_size):
        inpaint = np.ones(ld_img_batch[index, :, :, 0].shape)
        if(only_center_box==0):
            x = int(np.random.randint(cfg.img_size_x - box_dim, size=1))
            y = int(np.random.randint(cfg.img_size_y - box_dim, size=1))
        elif(only_center_box==1):
            x=int(cfg.img_size_x/2)-int(box_dim/2)
            y=int(cfg.img_size_y/2)-int(box_dim/2)
        elif(only_center_box==2):
            x = int(np.random.randint(cfg.img_size_x - box_dim_arr_x[index], size=1))
            y = int(np.random.randint(cfg.img_size_y - box_dim_arr_y[index], size=1))
        inpaint[x:x + box_dim, y:y + box_dim] = 0
        #ld_label_batch[index]=ld_img_batch[index, :, :, 0]
        ld_img_batch[index, :, :, 0] = ld_img_batch[index, :, :, 0] * inpaint

    return ld_img_batch

def create_rand_augs(cfg,parse_config,sess,df_ae_rd,df_ae_ri,ld_img_batch,ld_label_batch):
    '''
        To create bounding boxes with pixels values set to 0 in the image
        input param:
            cfg: contains config settings of the image
            parse_config: input configs for the experiment
            sess : session with networks
            df_ae_rd : random deformation fields graph
            df_ae_ri : random intensity transformations graph
            ld_img_batch: input batch of images
            ld_label_batch: input batch of masks
        return params:
            ld_img_batch: images applied with deformation fields / intensity transformations / both
            ld_label_batch_1hot: masks with same deformation fields / intensity transformations / both as applied on respective images
    '''
    batch_size=ld_img_batch.shape[0]
    #calc random deformation fields
    rand_deform_v = calc_deform(cfg,batch_size,0,parse_config.sigma)

    ld_img_batch_tmp=np.copy(ld_img_batch)
    ld_label_batch_1hot = sess.run(df_ae_rd['y_tmp_1hot'],feed_dict={df_ae_rd['y_tmp']:ld_label_batch})
    #print('tmp label shape',ld_label_batch.shape)
    ld_label_batch_tmp=np.copy(ld_label_batch)
    ###########################
    # use deform model to get deformed images on application of the random deformation fields
    ##########################
    if(parse_config.rd_en==1):
        rd_img_batch = sess.run(df_ae_rd['deform_x'],feed_dict={df_ae_rd['x_tmp']:ld_img_batch_tmp,df_ae_rd['flow_v']:rand_deform_v})
        rd_label_batch=sess.run([df_ae_rd['deform_y_1hot']],feed_dict={df_ae_rd['y_tmp']:ld_label_batch_tmp,df_ae_rd['flow_v']:rand_deform_v})
        rd_label_batch=rd_label_batch[0]

    #add random contrast and brightness over random deformations
    if(parse_config.ri_en==1 and parse_config.rd_en==0):
        #apply random instensity augmentations over original images
        ri_img_batch,_=sess.run([df_ae_ri['rd_fin'],df_ae_ri['rd_cont']], feed_dict={df_ae_ri['x_tmp']: ld_img_batch_tmp})
    elif(parse_config.rd_en==1 and parse_config.ri_en==1):
        # apply random instensity augmentations over randomly deformed images
        rd_ri_img_batch,_=sess.run([df_ae_ri['rd_fin'],df_ae_ri['rd_cont']], feed_dict={df_ae_ri['x_tmp']: rd_img_batch})

    if(parse_config.rd_ni==1):
        max_no=int(cfg.mtask_bs)-1
        no_orig=np.random.randint(1, high=max_no)

        if(parse_config.rd_en==1):
            ld_img_batch[no_orig:] = rd_img_batch[no_orig:]
            ld_label_batch_1hot[no_orig:] = rd_label_batch[no_orig:]
        elif(parse_config.ri_en==1):
            ld_img_batch[no_orig:] = ri_img_batch[no_orig:]
            ld_label_batch_1hot[no_orig:] = ld_label_batch_1hot[no_orig:]
        elif(parse_config.rd_en==1 and parse_config.ri_en==1):
            ld_img_batch[no_orig:] = rd_ri_img_batch[no_orig:]
            ld_label_batch_1hot[no_orig:] = rd_ri_img_batch[no_orig:]

    return ld_img_batch,ld_label_batch_1hot



def context_restoration(ld_img_batch,cfg,batch_size=10,patch_dim=5,N=10):
    '''
    # To perform swapping of patches of pixels in the image & the task is to restore this as the learning task
     input param:
         ld_img_batch: input batch of 2D images
         cfg: config parameters
         batch_size: batch size
         box_dim: dimensions of the patch box to swap
         N: No. of iterations of swapping to perform on 1 2D image.
    :return:
         ld_img_batch_fin: swapped batch of 2D images.
    '''

    ld_img_batch_tmp=np.copy(ld_img_batch)
    ld_img_batch_fin=np.copy(ld_img_batch)

    for index in range(0,batch_size):
        count=0
        #print('index',index)
        #for each image do 'N' patch swaps
        for i in range(0,10*N):
            #sample 2 numbers for x & y to define the 2 patches to swap
            box_dim_arr_x=np.random.randint(low=patch_dim,high=cfg.img_size_x-patch_dim,size=2)
            box_dim_arr_y=np.random.randint(low=patch_dim,high=cfg.img_size_y-patch_dim,size=2)
            # these are start points of (x1,y1)+cbox_size & (x2,y2)+cbox_size
            x1min,x2min=box_dim_arr_x[0],box_dim_arr_x[1]
            y1min,y2min=box_dim_arr_y[0],box_dim_arr_y[1]

            x1max,x2max=x1min+patch_dim,x2min+patch_dim
            y1max,y2max=y1min+patch_dim,y2min+patch_dim

            isOverlapping = (x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max)

            #print('x1,y1',x1min,x1max,y1min,y1max)
            #print('x2,y2',x2min,x2max,y2min,y2max)
            #print('isoverlap',isOverlapping,index)
            # check if they overlap
            if (isOverlapping==1):
                # if yes - not good sample, so resample
                continue
            else:
                # else - ok sample, so swap the sampled cboxs
                count=count+1

            #swap patch1 box pixel values into patch2 box
            ld_img_batch_fin[index,x1min:x1max,y1min:y1max,0]=ld_img_batch_tmp[index,x2min:x2max,y2min:y2max,0]
            #swap patch2 box pixel values into patch1 box
            ld_img_batch_fin[index,x2min:x2max,y2min:y2max,0]=ld_img_batch_tmp[index,x1min:x1max,y1min:y1max,0]

            if(count>=N):
                # Repeat 'N' times for each sampled image; break after N times
                #print('count',count)
                break
    return ld_img_batch_fin

def unscaled_gaussian(x, spread):
    return math.exp(-x**2/(2 * spread**2))


def sample_minibatch_for_global_loss_opti_cine(img_list,cfg,batch_sz,n_parts):
    '''
    Create a batch with '(2 * batch_sz) * n_vols' no. of 2D images where n_vols is no. of 3D volumes and n_parts is no. of partitions per volume.
    input param:
         img_list: input batch of 3D cines
         cfg: config parameters
         batch_sz: final batch size
         n_cines: number of cines
         n_parts: size of partitions
    return:
         fin_batch: swapped batch of 2D images.
    '''

    #select indexes of 'm' cines out of total M.
    im_ns=random.sample(range(0, len(img_list)), batch_sz)
    fin_batch=np.zeros((2*batch_sz,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))
    #print(im_ns)
    count = 0
    for vol_index in im_ns:
        im_v=img_list[vol_index]
        i_sel=random.sample(range(n_parts, im_v.shape[2]-n_parts), 1)[0]
        pair_options = list(range(i_sel-n_parts, i_sel+n_parts+1))
        pair_options.remove(i_sel)
        i_pair_sel = random.sample(pair_options, 1)[0]
        fin_batch[count]=np.expand_dims(im_v[:,:,i_sel], axis=2)
        count = count+1
        fin_batch[count]=np.expand_dims(im_v[:,:,i_pair_sel], axis=2)
        count = count+1

    return fin_batch

def sample_minibatch_for_global_loss_opti_cine_hn(img_list,cfg,batch_sz,n_parts):
    '''
    Create a batch with '(2 * batch_sz) * n_vols' no. of 2D images where n_vols is no. of 3D volumes and n_parts is no. of partitions per volume.
    input param:
         img_list: input batch of 3D cines
         cfg: config parameters
         batch_sz: final batch size
         n_cines: number of cines
         n_parts: size of partitions
    return:
         fin_batch: swapped batch of 2D images.
    '''

    #select indexes of 'm' cines out of total M.
    print(len(img_list))
    print(range(0, len(img_list)))
    print(batch_sz)
    im_ns=random.sample(range(0, len(img_list)), int(batch_sz / 2))
    fin_batch=np.zeros((2*batch_sz,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))
    #print(im_ns)
    count = 0
    for vol_index in im_ns:
        im_v=img_list[vol_index]
        i_sel=random.sample(range(n_parts, im_v.shape[2]-n_parts), 1)[0]
        pair_options = list(range(i_sel-n_parts, i_sel+n_parts+1))
        pair_options.remove(i_sel)
        i_pair_sel = random.sample(pair_options, 1)[0]
        fin_batch[count]=np.expand_dims(im_v[:,:,i_sel], axis=2)
        count = count+1
        fin_batch[count]=np.expand_dims(im_v[:,:,i_pair_sel], axis=2)
        count = count+1
        possible_include = list(range(n_parts, im_v.shape[2]-n_parts))
        possible_exclude = list(range(max(0, i_sel-3*n_parts), min(im_v.shape[2]-n_parts, i_sel+3*n_parts+1)))
        possible_i = [x for x in possible_include if x not in possible_exclude]
        i_sel_2 = random.sample(possible_i, 1)[0]
        pair_options_2 = list(range(i_sel_2-n_parts, i_sel_2+n_parts+1))
        pair_options_2.remove(i_sel_2)
        i_pair_sel_2 = random.sample(pair_options_2, 1)[0]
        fin_batch[count]=np.expand_dims(im_v[:,:,i_sel_2], axis=2)
        count = count+1
        fin_batch[count]=np.expand_dims(im_v[:,:,i_pair_sel_2], axis=2)
        count = count+1
    return fin_batch

def sample_minibatch_for_contrastive_loss_opti(img_list,cfg,batch_sz,spread,softened=False):
    #other idea to only select from the one and just use the
    labels = np.random.randint(0, 2, size=batch_sz)
    labels = labels.astype("float")
    fin_batch=np.zeros((2*batch_sz,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))
    for i in range(batch_sz):
        label = labels[i]
        if label:
            vol_ind = random.sample(range(0, len(img_list)), 1)[0]
            vol = img_list[vol_ind]
            frame_idx_1 = random.sample(range(spread, vol.shape[2]-spread), 1)[0]
            frame_idx_2 = random.sample(range(frame_idx_1-spread, frame_idx_1+spread+1), 1)[0]
            if softened:
                labels[i] = 1 - abs(frame_idx_1-frame_idx_2) / spread
            fin_batch[i]=np.expand_dims(vol[:,:,frame_idx_1], axis=2)
            fin_batch[i+batch_sz]=np.expand_dims(vol[:,:,frame_idx_2], axis=2)
        else:
            vol_ind = random.sample(range(0, len(img_list)), 2)
            
            vol1 = img_list[vol_ind[0]]
            frame_idx = random.sample(range(0, vol1.shape[2]), 1)[0]
            fin_batch[i]=np.expand_dims(vol1[:,:,frame_idx], axis=2)

            vol2 = img_list[vol_ind[1]]
            frame_idx = random.sample(range(0, vol2.shape[2]), 1)[0]
            fin_batch[i+batch_sz]=np.expand_dims(vol2[:,:,frame_idx], axis=2)
    return fin_batch, labels
        
def sample_minibatch_for_contrastive_loss_opti_hn(img_list,cfg,batch_sz,spread,softened=False):
    #other idea to only select from the one and just use the
    labels = np.random.randint(0, 3, size=batch_sz)
    labels = labels.astype("float")
    fin_batch=np.zeros((2*batch_sz,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))
    for i in range(batch_sz):
        print(i)
        label = labels[i]
        print(label)
        if label==1:
            print("LABEL 1")
            vol_ind = random.sample(range(0, len(img_list)), 1)[0]
            print(vol_ind)
            vol = img_list[vol_ind]
            print(vol.shape)
            frame_idx_1 = random.sample(range(spread, vol.shape[2]-spread), 1)[0]
            frame_idx_2 = random.sample(range(frame_idx_1-spread, frame_idx_1+spread+1), 1)[0]
            print(frame_idx_1, frame_idx_2)
            if softened:
                labels[i] = 1 - abs(frame_idx_1-frame_idx_2) / spread
            print(labels[i])
            print(fin_batch.shape)
            fin_batch[i]=np.expand_dims(vol[:,:,frame_idx_1], axis=2)
            fin_batch[i+batch_sz]=np.expand_dims(vol[:,:,frame_idx_2], axis=2)
        elif label==2:
            print("LABEL 2")
            vol_ind = random.sample(range(0, len(img_list)), 2)
            
            vol1 = img_list[vol_ind[0]]
            frame_idx = random.sample(range(0, vol1.shape[2]), 1)[0]
            fin_batch[i]=np.expand_dims(vol1[:,:,frame_idx], axis=2)
            print(frame_idx)
            include = list(range(0, vol1.shape[2]))
            exclude = list(range(frame_idx-spread, frame_idx+spread+1))
            possible_idx = [x for x in include if x not in exclude]

            frame_idx = random.sample(possible_idx, 1)[0]
            print(frame_idx)
            fin_batch[i+batch_sz]=np.expand_dims(vol1[:,:,frame_idx], axis=2)
            labels[i] = 0
            print(labels[i] )
        else:
            print("LABEL 0")
            vol_ind = random.sample(range(0, len(img_list)), 2)
            print(vol_ind)
            vol1 = img_list[vol_ind[0]]
            frame_idx = random.sample(range(0, vol1.shape[2]), 1)[0]
            fin_batch[i]=np.expand_dims(vol1[:,:,frame_idx], axis=2)

            vol2 = img_list[vol_ind[1]]
            frame_idx = random.sample(range(0, vol2.shape[2]), 1)[0]
            fin_batch[i+batch_sz]=np.expand_dims(vol2[:,:,frame_idx], axis=2)
            print(labels[i])
    return fin_batch, labels

def sample_minibatch_for_global_loss_opti(img_list,cfg,batch_sz,n_vols,n_parts):
    '''
    Create a batch with 'n_parts * n_vols' no. of 2D images where n_vols is no. of 3D volumes and n_parts is no. of partitions per volume.
    input param:
         img_list: input batch of 3D volumes
         cfg: config parameters
         batch_sz: final batch size
         n_vols: number of 3D volumes
         n_parts: number of partitions per 3D volume
    return:
         fin_batch: swapped batch of 2D images.
    '''

    count=0
    #select indexes of 'm' volumes out of total M.
    im_ns=random.sample(range(0, len(img_list)), n_vols)
    fin_batch=np.zeros((batch_sz,cfg.img_size_x,cfg.img_size_x,cfg.num_channels))
    #print(im_ns)
    for vol_index in im_ns:
        #print('j',j)
        #if n_parts=4, then for each volume: create 4 partitions, pick 4 samples overall (1 from each partition randomly)
        im_v=img_list[vol_index]
        ind_l=[]
        #starting index of first partition of any chosen volume
        ind_l.append(0)

        #find the starting and last index of each partition in a volume based on input image size. shape[0] indicates total no. of slices in axial direction of the input image.
        for k in range(1,n_parts+1):
            ind_l.append(k*int(im_v.shape[0]/n_parts))
        #print('ind_l',ind_l)

        #Now sample 1 image from each partition randomly. Overall, n_parts images for each chosen volume id.
        for k in range(0,len(ind_l)-1):
            #print('k',k,ind_l[k],ind_l[k+1])
            if(k+count>=batch_sz):
                break
            #sample image from each partition randomly
            i_sel=random.sample(range(ind_l[k],ind_l[k+1]), 1)
            #print('k,i_sel',k+count, i_sel)
            fin_batch[k+count]=im_v[i_sel]
        count=count+n_parts
        if(count>=batch_sz):
            break

    return fin_batch

def stitch_batch_global_loss_gd(cfg,batch1,batch2,batch3,batch4,n_parts):
    '''
    Create a merged batch of input 4 batches of 2D images.
    input param:
         cfg: config parameters
         batch1: batch one - original image batch
         batch2: batch two - batch 1 with a set of random crop + intensity augmentations
         batch3: batch three - another different image batch to batch 1
         batch4: batch three - batch 3 with a set of random crop + intensity augmentations.
         n_parts: number of partitions per 3D volume
    return:
         fin_batch: merged batch of 3 input batches one, two and three
    '''
    if(n_parts==4):
        max_bz=4*cfg.batch_size_ft
    else:
        max_bz=5*cfg.batch_size_ft+4
    fin_batch=np.zeros((max_bz,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))
    c=0
    for i in range(0,max_bz,4*n_parts):
        #print(i,c)
        if(i+4*n_parts>=max_bz):
            break
        fin_batch[i:i+n_parts]=batch1[c:c+n_parts]
        fin_batch[i+n_parts:i+2*n_parts]=batch2[c:c+n_parts]
        fin_batch[i+2*n_parts:i+3*n_parts]=batch3[c:c+n_parts]
        fin_batch[i+3*n_parts:i+4*n_parts] = batch4[c:c+n_parts]
        c=c+n_parts
    #print(fin_batch.shape)
    return fin_batch


def stitch_batch_global_loss_gdnew(cfg,batch1,batch2,batch3,batch4,batch5,batch6,n_parts):
    '''
    Create a merged batch of input 3 batches of 2D images.
    input param:
         cfg: config parameters
         batch1: batch one - One set of original images batch
         batch2: batch two - batch one with one set of random crop + intensity augmentations
         batch3: batch three - batch one with another set of random crop + intensity augmentations. This is different to batch two.
         batch4: batch four - another set of different original images batch to batch 1
         batch5: batch five - batch two with one set of random crop + intensity augmentations
         batch6: batch six - batch two with another set of random crop + intensity augmentations. This is different to batch five.
         n_parts: number of partitions per 3D volume
    return:
         fin_batch: merged batch of 3 input batches one, two and three
    '''
    if(n_parts==4):
        max_bz=4*cfg.batch_size_ft
    else:
        max_bz=5*cfg.batch_size_ft+4
    fin_batch=np.zeros((max_bz,cfg.img_size_x,cfg.img_size_y,cfg.num_channels))
    c=0
    for i in range(0,max_bz,6*n_parts):
        #print(i,c)
        if(i+6*n_parts>=max_bz):
            break
        fin_batch[i:i+n_parts]=batch1[c:c+n_parts]
        fin_batch[i+n_parts:i+2*n_parts]=batch2[c:c+n_parts]
        fin_batch[i+2*n_parts:i+3*n_parts]=batch3[c:c+n_parts]
        fin_batch[i+3*n_parts:i+4*n_parts] = batch4[c:c+n_parts]
        fin_batch[i+4*n_parts:i+5*n_parts] = batch5[c:c+n_parts]
        fin_batch[i+5*n_parts:i+6*n_parts] = batch6[c:c+n_parts]
        c=c+n_parts
    #print(fin_batch.shape)
    return fin_batch

