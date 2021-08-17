import json
import os

selection_file = "/scratch/st-rohling-1/contrastive_learning/domain_specific_cl/split.json"

with open(selection_file) as fp:
    selection = json.load(fp)

files = os.listdir("/arc/project/st-rohling-1/data/kidney/CL_data/dscl_data/unlabelled/")
identifiers = [x.replace(".nii.gz", "") for x in files]
selection["pretrain"] = identifiers
with open(selection_file, 'w') as fp:
    selection = json.dump(selection, fp)