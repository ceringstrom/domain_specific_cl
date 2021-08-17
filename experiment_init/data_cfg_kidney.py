import sys
import json

split_file = "/scratch/st-rohling-1/contrastive_learning/domain_specific_cl/split.json"

def train_data(no_of_tr_imgs,comb_of_tr_imgs):
    with open(split_file) as fp:
        splits = json.load(fp)
    if(no_of_tr_imgs=='ptr' and comb_of_tr_imgs=='c1'):
        labeled_id_list=splits['pretrain']
    elif(no_of_tr_imgs=='ftn'):
        split_num = str(int(comb_of_tr_imgs.replace('c', '')) / 100)
        # split_num = int(split_num) if split_num==1.0 else spl
        if split_num in splits:
            labeled_id_list=splits[split_num]['train']
        else:
            print('Error! Select valid combination of training images')
            sys.exit() 
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list

def val_data(no_of_tr_imgs,comb_of_tr_imgs):
    with open(split_file) as fp:
        splits = json.load(fp)
    if(no_of_tr_imgs=='ftn'):
        split_num = str(int(comb_of_tr_imgs.replace('c', '')) / 100)
        print(split_num)
        print(splits.keys())
        if split_num in splits.keys():
            labeled_id_list=splits[split_num]['val']
        else:
            print('Error! Select valid combination of training images')
            sys.exit() 
    else:
        print('Error! Select valid combination of training images')
        sys.exit() 
    return labeled_id_list

def test_data():
    with open(split_file) as fp:
        splits = json.load(fp)

    labeled_id_list=splits['1.0']['test']

    return labeled_id_list
