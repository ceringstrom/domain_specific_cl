# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:58:55 2021

@author: Ricky

Functions for QUS spekcle analysis
"""
import os
import sys
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

imDir = os.path.join(reg_cfg.data_path_tr_cropped, "imagesTr")
maskDir = os.path.join(reg_cfg.data_path_tr_cropped, "labelsTr")
outDir = os.path.join(reg_cfg.data_path_tr_cropped, "statsTr")
# statsListColNames = ['fileName', 'maskNum',
#                      'burrC','burrD', 'burrLoc', 'burrScale',
#                      'gamShape', 'gamLoc', 'gamScale', 
#                      'lomaxShape','lomaxLoc','lomaxScale',
#                      'nakShape', 'nakLoc', 'nakScale',
#                      'paretoShape', 'paretoLoc','paretoScale',
#                      'rayleighLoc','rayleighScale',
#                      'ricShape','ricLoc','ricScale']
statsListColNames = ['fileName', 'maskNum',
                     'nakShape', 'nakLoc', 'nakScale']
outFile = outDir + '\stats.csv'
numParams = len(statsListColNames) - 2
#Setting patch size, should be rectangular representing the resolution cell
patchSizeX = 20
patchSizeY = 20
#minimum nonzero pixels in patch, other wise ignore patch
minMaskPixels = -1

#Makes a patch of size (height, width) given an image and the (row, col) 
#for the center of that patch
def patchStats(im, mask):
    # print(type(im))
    # print(im.shape)
    # print(mask.shape)
    im = np.squeeze(im)
    #creating list of parameter maps
    rows, cols = im.shape
    paramMaps = []
    for i in range(0, numParams):
        paramMaps.append(np.zeros((rows,cols)))
    
    #iterating through all rows and cols and making the patches
    for row in range(0,rows):
        print(row)
        for col in range(0,cols):
            #have to define start and stop index of the patch
            halfPatchX = math.floor(patchSizeX/2)
            halfPatchY = math.floor(patchSizeY/2)
            
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

            #now we can get the patch from the image and mask
            imPatch = im[patchStartY:patchEndY, patchStartX:patchEndX]
            maskPatch = mask[patchStartY:patchEndY, patchStartX:patchEndX]
            totalMaskPixels = np.count_nonzero(maskPatch)
            
           #now these patches serve as the input images to the uusal fitDistribution fnc
           #if insufficient masked pixels, then save comp time by ignoring
            if(totalMaskPixels > minMaskPixels):
                # print("Mask Pixels:", totalMaskPixels, "Patch: (", row, col, "), ",
                # patchStartY, patchEndY, patchStartX, patchEndX)
                
                #computing stats for this patch
                statsList = fitDistributions(imPatch)[0]
                #filling the parameter maps with the distro params
                for i in range(0, numParams):
                    paramMaps[i][row,col] = statsList[i]
            else:
                print('Insufficient masked pixels: ', totalMaskPixels, "Patch: (", row, col, "), ",
                  patchStartY, patchEndY, patchStartX, patchEndX)

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
    maskFile = maskDir + "/" + os.path.basename(pngFile)
    matFile = outDir + "/" + os.path.basename(pngFile.replace(".nii.gz", ".npz"))

    imArray = asarray(nib.load(pngFile).get_data())
    maskArray = asarray(nib.load(maskFile).get_data())
    

    paramMaps = patchStats(imArray, maskArray)
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
for pngFile in glob(os.path.join(imDir, "*.nii.gz")):
    result = pool.apply_async(processImage, args=(pngFile,))
    break
pool.close()
pool.join()


#full image stats
#creating one big list, future dataframe for all the stats
# statsList = []
# for pngFile in glob(os.path.join(imDir, "*.png")):
#     print(pngFile)
    
#     maskFile = maskDir + "\\" + os.path.basename(pngFile)
#     matFile = matDir + "\\" + os.path.basename(pngFile)  + ".mat"
    
#     if(os.path.isfile(pngFile) and os.path.isfile(maskFile)):
#         imArray = asarray(Image.open(pngFile).convert('L'))
#         maskArray = asarray(Image.open(maskFile).convert('L'))
        
#         maskList = np.unique(maskArray)
#         maskedPixelsList = []
        
        
#         #looping through the masks
#         #storing masked pixels ast a list of sublists, one sublist for each mask
#         #careful - index 0 corresponds to mask value 1
        
#         for maskNum in maskList:
#             if maskNum != 0:
#                 print("Fitting dist on image, mask #: ", maskNum)
#                 maskedPixelsList = getMaskedPixels(imArray, maskArray, maskNum)
#                 #appending file name, then the stats list
#                 distributionParams = fitDistributions(maskedPixelsList)[0].tolist()
#                 statsList.append([os.path.basename(pngFile)] + [maskNum] + distributionParams)



# print('Saving: ', outFile)
# statsDF = pd.DataFrame(statsList, columns = statsListColNames)
# statsDF.to_csv(outFile, index=False)


#showing plots
# plt.figure()
# plt.imshow(mask) 
# plt.show() 
