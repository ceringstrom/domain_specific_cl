import sys
import json

selection_file = "/scratch/st-rohling-1/contrastive_learning/domain_specific_cl/data/split.json"

def train_data(no_of_tr_imgs,comb_of_tr_imgs):
    with open(selection_file) as fp:
        selection = json.load(fp)
    if(no_of_tr_imgs=='ptr' and comb_of_tr_imgs=='c1'):
        labeled_id_list=selection['pretrain']
    elif(no_of_tr_imgs=='ftn' and comb_of_tr_imgs=='c1'):
        labeled_id_list=selection['train']
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list

def val_data(no_of_tr_imgs,comb_of_tr_imgs):
    with open(selection_file) as fp:
        selection = json.load(fp)
    return selection['validation']

def test_data():
    with open(selection_file) as fp:
        selection = json.load(fp)
    return selection['test']
