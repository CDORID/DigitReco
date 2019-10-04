# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:36:39 2019

@author: user
"""

import os 
import cv2
import Cropping as crop
import Preprocessing as pre
import imutils


def quality(definition = "M"):
    if definition == 'H':
        test_path = 'HQ_digital/'
        cropping_path = 'HQ_digital/cropped/'
        pointed_path = 'HQ_digital/points/'
        preprocessed_path = 'HQ_digital/prepro/'
    elif definition == 'M':
        test_path = 'MQ_digital/data/'
        cropping_path = 'cropping_test/'
        pointed_path = 'points_found/'
        preprocessed_path = 'preprocessed'
    
    return test_path, cropping_path, pointed_path, preprocessed_path
        
        
def test_files(path ='', cropping_path = 'cropping_test/',pointed_path = 'points_found/',
               preprocessed_path = 'preprocessed/', nfiles = 5):
    
    arr_files = os.listdir(path)[:nfiles]
    for file in arr_files:
        img =  cv2.imread(path+str(file))
        
        
        
        #prepro = pre.first_preprocess(img)
        
        #cv2.imwrite(preprocessed_path+str(file), prepro)
        
        cropped, pointed = crop.first_cropping(img)
        
        cv2.imwrite(pointed_path+str(file), pointed)
        
        cv2.imwrite(cropping_path+str(file), cropped)
    
    print('cropping done in '+str(cropping_path)+' for '+str(nfiles)+' files')
      
    return



nfiles = 1
test_path, cropping_path, pointed_path, preprocessed_path = quality('M')
test_files(path = test_path, cropping_path = cropping_path, pointed_path = pointed_path,
           preprocessed_path = preprocessed_path, nfiles = nfiles)