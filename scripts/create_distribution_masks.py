# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:58:55 2021

@author: Ricky

Functions for QUS spekcle analysis
"""
import os
import sys
import time
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from scipy.stats import burr, gamma, lomax, nakagami, pareto, rayleigh, rice
import SimpleITK as sitk
import math
import pandas as pd
import os.path
from glob import glob
import nibabel as nib
import scipy.io
import multiprocessing as mp
sys.path.append("/scratch/st-rohling-1/contrastive_learning/domain_specific_cl/domain_specific_cl")
import experiment_init.init_kidney_regions as reg_cfg

imDir = reg_cfg.data_path_pretr_cropped
dirpath, _ = os.path.split(reg_cfg.data_path_pretr_cropped)
outDir = "/scratch/st-rohling-1/contrastive_learning/domain_specific_cl/unlabelled_stats"

print("out" + outDir)
# statsListColNames = ['fileName', 'maskNum',
#                      'burrC','burrD', 'burrLoc', 'burrScale',
#                      'gamShape', 'gamLoc', 'gamScale', 
#                      'lomaxShape','lomaxLoc','lomaxScale',
#                      'nakShape', 'nakLoc', 'nakScale',
#                      'paretoShape', 'paretoLoc','paretoScale',
#                      'rayleighLoc','rayleighScale',
#                      'ricShape','ricLoc','ricScale']
statsListColNames = ['fileName', 'maskNum',
                     'nakShape', 'nakScale']

numParams = len(statsListColNames) - 2
#Setting patch size, should be rectangular representing the resolution cell
patchSizeX = 20
patchSizeY = 20
#minimum nonzero pixels in patch, other wise ignore patch
minMaskPixels = -1

#Makes a patch of size (height, width) given an image and the (row, col) 
#for the center of that patch
def patchStats(im, patchSizeX, patchSizeY):
    #creating list of parameter maps
    im = im[:, :, 0]
    rows, cols = im.shape
    paramMaps = []
    for i in range(0, numParams):
        paramMaps.append(np.zeros((rows,cols)))
    
    #iterating through all rows and cols and making the patches
    
    halfPatchX = math.floor(patchSizeX/2)
    halfPatchY = math.floor(patchSizeY/2)
    tic = time.time()
    for row in range(0,rows):
        
        # tic = time.time()
        for col in range(0,cols):
            #have to define start and stop index of the patch

            
            #careful, python uses 0 indexing unlike matlab
            #NOTE: currently no padding is done on the edges (e.g. patches on
            #top edge have height of patchSize/2 instead of patchSize), assuming
            #edges don't really matter and we don't have to waste computation
            #to compute them
            if((row - halfPatchY)<0):
                patchStartY = 0
            else:
                patchStartY = (row - halfPatchY)
            #since it's 0 indexing, last row is (row-1)
            if((row + halfPatchY) >= rows):
                patchEndY = (rows - 1)
            else:
                patchEndY = (row + halfPatchY)
            if((col - halfPatchX)<0):
                patchStartX = 0
            else:
                patchStartX = (col - halfPatchY)
            #since it's 0 indexing, last row is (row-1)
            if((col + halfPatchX) >= cols):
                patchEndX = (cols - 1)
            else:
                patchEndX = (col + halfPatchX)

            imPatch = im[patchStartY:patchEndY, patchStartX:patchEndX]

            #print(Patch: (", row, col, "), ",
            #      patchStartY, patchEndY, patchStartX, patchEndX)
            
            #computing stats for this patch
            #statsList = fitDistributions(imPatch)
            statsList = estimateNakagami(imPatch)
            
            #filling the parameter maps with the distro params
            for i in range(0, numParams):
                paramMaps[i][row,col] = statsList[i]
        
    toc = time.time()
    print('Row: ', row, ' of ', rows, (str(toc-tic)), 's')

    return paramMaps


def fitDistributions(im):
    statsList = []
    #burrShape, burrLoc, burrScale = burr.fit(im)
    # print('Computing distributions')
    # burrC, burrD, burrLoc, burrScale = burr.fit(im)
    # gammaShape, gammaLoc, gammaScale = gamma.fit(im)
    # lomaxShape, lomaxLoc, lomaxScale = lomax.fit(im)
    nakShape, nakLoc, nakScale = nakagami.fit(im)
    # paretoShape, paretoLoc, paretoScale = pareto.fit(im)
    # rayleighLoc, rayleighScale = rayleigh.fit(im)
    # ricShape, ricLoc, ricScale = rice.fit(im)

    # statsList.append([burrC, burrD, burrLoc, burrScale,
    #                   gammaShape, gammaLoc, gammaScale, 
    #                   lomaxShape, lomaxLoc, lomaxScale,
    #                   nakShape, nakLoc, nakScale,
    #                   paretoShape, paretoLoc, paretoScale,
    #                   rayleighLoc, rayleighScale,
    #                   ricShape, ricLoc, ricScale])
    statsList.append([nakShape, nakLoc, nakScale])

    return np.nan_to_num(statsList)

def estimateNakagami(im):

    #making arrays to compute expectations as per nakagami estimates
    #careful for overflows, need to declare python int in case
    
    # im = im.astype(np.int64)
    e_x2 = 0
    e_x4 = 0
    # im = im[:, :, 0]
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
        nakShape = 0
    else:
        nakShape = e_x2**2 / ( e_x4 - (e_x2**2))
    
    return np.nan_to_num([nakShape, nakScale])

#creates an array of only the pixels corresponding to a certain binary mask
def getMaskedPixels(im, mask, maskVal):
    maskedPixels = []
    rows, cols = im.shape
    for x in range(0,rows):
        for y in range(0,cols):
            if(mask[x,y] == maskVal):
                maskedPixels.append(im[x,y])

    return maskedPixels

def processImage(pngFile):
    print("File Being Processed: " + pngFile)
    matFile = outDir + "/" + os.path.basename(pngFile.replace(".nii.gz", ".npz"))

    imArray = asarray(nib.load(pngFile).get_data())
    

    paramMaps = patchStats(imArray, patchSizeX, patchSizeY)
    # scipy.io.savemat(matFile, mdict={'data': paramMaps})
    print("outfile:" + matFile)
    np.savez(matFile, np.array(paramMaps))

#main script starts here
if not(os.path.exists(outDir)):
    os.mkdir(outDir)

# print(glob(os.path.join(imDir, "*.nii.gz")))
cpus = mp.cpu_count()  
pool = mp.Pool(processes=cpus)
print("cpus " + str(cpus))
#paramater maps
print(len(glob(os.path.join(imDir, "*.nii.gz"))))
print(os.path.join(imDir, "*.nii.gz"))
for pngFile in glob(os.path.join(imDir, "*.nii.gz")):
    # print(pngFile)
    # processImage(pngFile)
    result = pool.apply_async(processImage, args=(pngFile,))
pool.close()
pool.join()
