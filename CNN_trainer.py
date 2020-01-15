# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:30:02 2018

@author: Dr. Mark M. Bailey | National Intelligence University
"""

"""
Usage notes:
    This training script requires a file hierarchy as follows, which exists in the SAME directory as this script:
        
        training_set
            Label1
            Label2
        test_set
            Label1
            Label2
    
    This script will export the model artifact (*.h5) to the same directory as this script.
"""

print('Loading...')
#Import Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os

#Helper function
def get_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_list.append(os.path.join(root, name))
    return file_list

print('CNN will build your convolutional neural network!')
print('====================================================')
print('Accessing image data...')


model_name = 'CNN_model'
training_files_list = get_files(os.path.join(os.getcwd(), 'training_set'))
train_number = len(training_files_list)

test_files_list = get_files(os.path.join(os.getcwd(), 'test_set'))
test_number = len(test_files_list)

print('Training model...')
#Instantiate the convolutional neural network
classifier = Sequential()

#Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Add a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
classifier.fit_generator(training_set, steps_per_epoch = train_number, epochs = 25, validation_data = test_set, validation_steps = test_number)
classes = training_set.class_indices

#Export CNN model
print('Exporting model...')

import json
with open('classes.json', 'w') as outfile:
    json.dump(classes, outfile)

model_name_str = model_name + '.h5'
classifier.save(model_name_str)
print('CNN model exported as {}'.format(model_name_str))

print('You are a great American!!')