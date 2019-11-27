'''
Pre-process the data to extract patches
Input: A csv file containing path to input files

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import math
import pdb
import numpy as np
import SimpleITK as sitk
import pandas as pd
import os.path   
import subprocess           
from sklearn.model_selection import KFold

def convert_to_isotropic(imageFilePath, outputFileName):
    #Read new image file
    try:
        inputVolume = sitk.ReadImage(imageFilePath)
        npInputVolume = sitk.GetArrayFromImage(inputVolume)
    except:
        print("Not able to read volume in sitk: " + str(imageFilePath))
        sys.exit()

    inputSpacing = inputVolume.GetSpacing()
    inputSize = inputVolume.GetSize()
    #Resample the images to make them iso-tropic
    resampleFilter = sitk.ResampleImageFilter()
    T = sitk.Transform()
    T.SetIdentity()
    resampleFilter.SetTransform(T)
    resampleFilter.SetInterpolator(sitk.sitkBSpline)
    resampleFilter.SetDefaultPixelValue( float(np.min(npInputVolume) ))
    isoSpacing = 1 #math.sqrt(inputSpacing[2] * inputSpacing[0])
    resampleFilter.SetOutputSpacing((isoSpacing,isoSpacing,isoSpacing))
    resampleFilter.SetOutputOrigin(inputVolume.GetOrigin())
    resampleFilter.SetOutputDirection(inputVolume.GetDirection())
    dx = int(inputSize[0] * inputSpacing[0] / isoSpacing)
    dy = int(inputSize[1] * inputSpacing[1] / isoSpacing)
    dz = int((inputSize[2] - 1 ) * inputSpacing[2] / isoSpacing)
    resampleFilter.SetSize((dx,dy,dz))
    try:
        resampleVolume = resampleFilter.Execute(inputVolume)
    except:
        print("Resample failed: " + str(imageFilePath) )
        sys.exit()
    sitk.WriteImage(resampleVolume, outputFileName)
    return resampleFilter    

def Image2Patch(inputImg, labelMaskImg, patchSize, finalPatchSize, acceptRate):
    """ This function converts image to patches. 
        Here is the input of the function:
          inputImg : input image. This should be simpleITK object
          labelMaskImg : label image containing mask of the lobes (values greater than 0)
          patchSize : size of the patch. It should be array of three scalar
          acceptRate : If portion of the patch inside of the mask exceeds value, it would be accepted 
        Here is the output of the function:
          patchImgData : It is a list containing the patches of the image
          patchLblData : Is is a list containing the patches of the label image
          
    """
    patchVol = patchSize[0]*patchSize[1]*patchSize[2]
    largePatchImgData = []

    for x in range(0,inputImg.GetSize()[0]-patchSize[0],patchSize[0]):
        for y in range(0,inputImg.GetSize()[1]-patchSize[1],patchSize[1]):
            for z in range(0,inputImg.GetSize()[2]-patchSize[2],patchSize[2]):
                patchLblImg = sitk.RegionOfInterest(labelMaskImg, size=patchSize, index=[x,y,z])
                npPatchLblImg = sitk.GetArrayFromImage(patchLblImg)
                if ((npPatchLblImg > 0).sum() > acceptRate*patchVol ):    # if the patch has more than 70%
                    #largePatchSize = [2*patchSize[0], 2*patchSize[1], 2*patchSize[2]]
                    largePatchSize = finalPatchSize
                    #largePatchIndex = [x-patchSize[0]/2, y-patchSize[1]/2, z-patchSize[2]/2]
                    shift_x = int((finalPatchSize[0] - patchSize[0])/2)
                    shift_y = int((finalPatchSize[1] - patchSize[1])/2)
                    shift_z = int((finalPatchSize[2] - patchSize[2])/2)
                    largePatchIndex = [x-shift_x, y-shift_y, z-shift_z]
                    try:
                        largePatchImg = sitk.RegionOfInterest(inputImg, size=largePatchSize, index=[x-6,y-6,z-6])
                        npLargePatchImg = sitk.GetArrayFromImage(largePatchImg)
                        largePatchImgData.append(npLargePatchImg.copy())

                    except:
                        print("Overlapping Patch outside the largest possible region...")
    

    largePatchImgData = np.asarray(largePatchImgData)
    
    return largePatchImgData

def extract_patch(isoRawImage_file, isoLabelImage_file, upperThreshold, lowerThreshold, patch_size, overlap_size,acceptRate, max_no_patches):
    #Read the input isotropic image volume
    isoRawImage = sitk.ReadImage(isoRawImage_file)
    npIsoRawImage = sitk.GetArrayFromImage(isoRawImage)
    #print(npIsoRawImage.shape)
   
    # Thresholding the isotropic raw image
    npIsoRawImage[npIsoRawImage > upperThreshold] = upperThreshold
    npIsoRawImage[npIsoRawImage < lowerThreshold] = lowerThreshold

    thresholdIsoRawImage = sitk.GetImageFromArray(npIsoRawImage)
    thresholdIsoRawImage.SetOrigin(isoRawImage.GetOrigin())
    thresholdIsoRawImage.SetSpacing(isoRawImage.GetSpacing())
    thresholdIsoRawImage.SetDirection(isoRawImage.GetDirection())
    print("Input Volume: ", thresholdIsoRawImage.GetSize())
    
    #Read the input isotropic label image
    isoLabelImage = sitk.ReadImage(isoLabelImage_file)
    #npIsoLabelImage = sitk.GetArrayFromImage(isoLabelImage)
    print("Lung Mask Volume: ", isoLabelImage.GetSize())
    
    #Generate binary label map
    thresholdFilter = sitk.BinaryThresholdImageFilter()
    binaryLabelImage = thresholdFilter.Execute(isoLabelImage, 1, 1024, 1, 0)

    #Extract Patches
    # Generate Patches of the masked Image
    while True:
        patchImgData = Image2Patch(thresholdIsoRawImage, binaryLabelImage, [overlap_size, overlap_size, overlap_size], \
                       [patch_size, patch_size, patch_size], acceptRate)
        if patchImgData.shape[0] < max_no_patches:
            break
        if acceptRate < 1:
            acceptRate += 0.09
            print("Trying again with acceptRate: ", acceptRate)
        else:
            return np.empty([0])
    
    return patchImgData                   

     
def main(argv):    
    #Parse the arguments
    parser = argparse.ArgumentParser(description='Subject2Vector Data Preprocessing')
    parser.add_argument('-i', '--input_csv', type=str, default='../Data/Subject2Vec_Input.csv', help = 'Input csv with sid, nifty , segmentation-nifty. segmentation-nifty is optional. Choose --segmentation=1 to perform segmentation using CIP.')
    parser.add_argument('-d', '--data_dir', type=str, default='/pghbio/dbmi/batmanlab/Data/COPDGene', help='Directory where input data is stored. There should be one folder for each patient id in input csv')
    parser.add_argument('-o', '--output_dir', type=str, default='../Output/COPD_Sub2Vec_Data',  help='Directory where files are saved. We will create one folder for each sid and save all the files for that sid there.')
    parser.add_argument('-s1', '--output_csv', type=str, default='../Output/Subject2Vec_Input_Updated.csv',  help='Output csv files with columns-sid, nifty, nifty-segmentation, nifty-ISO, nifty-segmentation-ISO, patch,FEV1pp_utah,...')
    parser.add_argument('-s', '--segmentation', type=int, default=1, help='Do we perfrom CIP based segmentation of nifty file.')   
    parser.add_argument('-p', '--patch_size', type=int, default=32, help='Size of patch.')
    parser.add_argument('-ol', '--overlap_size', type=int, default=20, help='Size of overlap between patches.')
    parser.add_argument('-ut', '--upperThreshold', type=int, default=240, help='Upper Threshold for Lung in CT image.')
    parser.add_argument('-lt', '--lowerThreshold', type=int, default=-1024, help='Lower Threshold for Lung in CT image.')
    parser.add_argument('-a', '--acceptRate', type=float, default=0.7, help='If portion of the patch inside of the mask exceeds value, it would be accepted')
    parser.add_argument('-n', '--max_no_patches', type=int, default=1000, help='The maximum number of patches in any subject.')
    parser.add_argument('-f', '--folds', type=int, default=2, help='The number of folds for cross validation.')
    
    
    
    
    args = parser.parse_args()
    input_csv = args.input_csv
    print(input_csv)
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_csv = args.output_csv
    patch_size = args.patch_size
    overlap_size = args.overlap_size
    segmentation = args.segmentation
    upperThreshold = args.upperThreshold
    lowerThreshold = args.lowerThreshold
    acceptRate = args.acceptRate
    max_no_patches = args.max_no_patches
    folds = args.folds
    
    
    df = pd.read_csv(input_csv)
    null_count = np.sum(df.isnull().any(axis=1))
    if null_count > 0:
        print("There are null values in input csv: ", input_csv)
        sys.exist()
    if 'sid' not in df.columns:
        print("There is no 'sid' columns in input csv: %s" % (input_csv))
        sys.exist()
    if 'nifty' not in df.columns:
        print("There is no 'nifty' columns in input csv: ", input_csv)
        sys.exist()
    if 'segmentation-nifty' not in df.columns and segmentation != 1:
        print("There is no 'segmentation-nifty' columns in input csv: ", input_csv, " and Segmentation is False.")
        sys.exist()
    if 'patch' in df.columns:
        allSubjects =  np.asarray(df['sid'])
        kf = KFold(n_splits=folds, shuffle=True, random_state=100)
        fold = 1
        for train_index, test_index in kf.split(allSubjects):
            Train, Test = allSubjects[train_index], allSubjects[test_index]
            print("TRAIN:", Train.shape, "TEST:", Test.shape)
            df['fold'+str(fold)]=''
            for index,row in df.iterrows():
                if row['sid'] in Train:
                    df.loc[index, 'fold'+str(fold)] = 'Train'
                else:
                    df.loc[index, 'fold'+str(fold)] = 'Test'
            print(np.unique(np.asarray(df['fold'+str(fold)]),return_counts=True))
            fold += 1
        
        df.to_csv(output_csv, sep=',', index=None)
        sys.exit()

    df['iso-nifty'] = ''
    df['iso-segmentation-nifty'] = ''
    df['patch'] = ''
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    patch_name = '_' + str(patch_size) + '_' + str(patch_size) + '_' + str(patch_size) + '_overlap_' + str(overlap_size) + '_' + str(overlap_size) + '_' + str(overlap_size) + '.npy'
    for index,row in df.iterrows():
        current_nifty_path = os.path.join(data_dir, row['nifty'])
        current_nifty_name = current_nifty_path.split('/')[-1]
        prefix = current_nifty_name.split('.')[0]
        current_output_dir = os.path.join(output_dir, row['sid'])
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        #Isotropic nifty file
        nifty_iso_path = os.path.join(current_output_dir, prefix+'_BSpline_Iso1.0mm.nii.gz')
        #Isotropic segmentation file
        seg_path = os.path.join(current_output_dir, prefix+'_partialLungLabelMap_BSpline_Iso1.0mm.nii.gz')
        if segmentation != 1:
            current_seg_path = os.path.join(data_dir, row['segmentation-nifty'])
        #Patch file
        patch_path = os.path.join(current_output_dir, prefix + patch_name )
        
        #Step-1 Covert Nifty --> Isotropic Volume
        if not os.path.exists(nifty_iso_path): 
            _ = convert_to_isotropic(current_nifty_path, nifty_iso_path)
        df.loc[index, 'iso-nifty'] = nifty_iso_path
        
        #Step-2 Create lung segmentation from Isotropic Nifty Volume
        if not os.path.exists(seg_path):
            if segmentation == 1:
                try:
                    s = subprocess.check_output(["GeneratePartialLungLabelMap", "--ict", nifty_iso_path, "-o", seg_path])
                except:
                    print("Failed in partial lung label map: " + str(nifty_iso_path) )
                    sys.exit()
            else:
                # Convert given segmentation to isotropic segmentation
                _ = convert_to_isotropic(current_seg_path, seg_path)
        df.loc[index, 'iso-segmentation-nifty'] = seg_path
        
        #Step-3 
        #extract patches
        if not os.path.exists(patch_path):
            patchImgData = extract_patch(nifty_iso_path, seg_path, upperThreshold, lowerThreshold, patch_size, overlap_size,acceptRate, max_no_patches) 
            if patchImgData.shape[0] > 20:
                np.save(patch_path, patchImgData)
        df.loc[index, 'patch'] = patch_path
        
    #Step-4 Divide data into folds 
    allSubjects =  np.asarray(df['sid'])
    kf = KFold(n_splits=folds, shuffle=True, random_state=100)
    fold = 1
    for train_index, test_index in kf.split(allSubjects):
        Train, Test = allSubjects[train_index], allSubjects[test_index]
        print("TRAIN:", Train.shape, "TEST:", Test.shape)
        df['fold'+str(fold)]=''
        for index,row in df.iterrows():
            if row['sid'] in Train:
                df.loc[index, 'fold'+str(fold)] = 'Train'
            else:
                df.loc[index, 'fold'+str(fold)] = 'Test'
        print(np.unique(np.asarray(df['fold'+str(fold)]),return_counts=True))
        fold += 1
    
    df.to_csv(output_csv, sep=',', index=None)

if __name__ == '__main__':
    main(sys.argv[1:])