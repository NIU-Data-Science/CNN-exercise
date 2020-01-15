# -*- coding: utf-8 -*-
"""
Created on Thu Oct 03 13:04:08 2019

@author: Dr. Mark M. Bailey | National Intelligence University
"""

#Import libraries
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os

#directory where images to be categorized live
directory = ''

#Helper function
def get_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_list.append(os.path.join(root, name))
    return file_list

#Image classifier function (for list of images)
def image_classifier(image_path, model_path):
    classifier = load_model(model_path)
    image_list = get_files(image_path)
    prediction_list = []
    for i in range(len(image_list)):
        test_image = image.load_img(image_list[i], target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        #Generate prediction and reference label
        result = classifier.predict(test_image)
        prediction_list.append(result)
    out_list = dict(zip(image_list, prediction_list))
    return out_list