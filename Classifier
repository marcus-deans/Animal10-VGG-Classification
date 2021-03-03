

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
import keras
from keras import layers, applications, optimizers
from keras.layers import Input, Dense, Activation, MaxPool2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils import class_weight, shuffle
import os 
import random
import cv2

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import requests
from PIL import Image
from io import BytesIO

#The Original Kaggle dataset is in Italian, so to more easily interpret, we convert Italian labels to English
translate = {"cane": "Dog", "cavallo": "Horse", "elefante": "Elephant", "farfalla": "Butterfly", "gallina": "Chicken", "gatto": "Cat", "mucca": "Cow", "pecora": "Sheep", "scoiattolo": "Squirrel", "ragno": "Spider"}

#Directly for the animal image classification database from Kaggle
foldernames = os.listdir('/kaggle/input/animals10/raw-img/')
files, files2, target, target2 = [], [], [], []

#Iterate through the database and retrieve our relevant files
for i, folder in enumerate(foldernames):
    filenames = os.listdir("/kaggle/input/animals10/raw-img/" + folder);
    count = 0
    #Due to the specific nature of the database being used, there are 1446 images of a specific class (others are higher)
    #Hence use a maximum of 1400 images from a specific classes for consistency of data as well as brevity
    for file in filenames:
        if count < 1400:
            files.append("/kaggle/input/animals10/raw-img/" + folder + "/" + file)
            target.append(translate[folder])
        else:
            files2.append("/kaggle/input/animals10/raw-img/" + folder + "/" + file)
            target2.append(translate[folder])
        count += 1

#Create dataframes to read the images 
df = pd.DataFrame({'Filepath':files, 'Target':target})

#Split into training, dev, and test sets with a 60/20/20 split respsectively
train, dev, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

#Perform data augmentation which will artifically grow dataset, allowing CNN to better generalize learning
#Images were sheared, rotated, zoomed, and shifted by up to 20% of the relevant factor (shear, degrees, zoom, width, height respectively)
#Note that horizontal but not vertical flips were employed to simulate real-world conditions
#Simultaneously normalize by dividing by 255 within the image data generator
augdata = ImageDataGenerator(rescale=1./255,
        shear_range = 0.2,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#Create the test set which will not have augmented data (want real performance not on data of identical origin)
augdata_test = ImageDataGenerator(rescale=1./255, samplewise_center = True)

#Create image sets for the train, dev, and test sets. The training set has augmented data whereas the dev and test sets do not
#A standard image size of 224 is used for the VGG16 CNN. Use lanczos interpolation in frequency domain to reduce potential aliasing from resizing
train_flow = augdata.flow_from_dataframe(train, x_col = 'Filepath', y_col = 'Target', target_size=(224, 224), interpolation = 'lanczos', validate_filenames = False)
dev_flow = augdata_test.flow_from_dataframe(dev, x_col = 'Filepath', y_col = 'Target', target_size=(224, 224), interpolation = 'lanczos', validate_filenames = False)
test_flow = augdata_test.flow_from_dataframe(test, x_col = 'Filepath', y_col = 'Target', target_size=(224, 224), interpolation = 'lanczos', validate_filenames = False)


#Reducing learning rate of CNN during plateaus to continue progress. Learning rate is projected to be minute at end of CNN and hence min_lr=1e-8
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 1, verbose=1,factor=0.2, min_delta=0.0001, min_lr=0.00000001)

#Use transfer learning with VGG16 CNN with previously determined image classification weights for faster learning and high accuracy
vgg16_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Create a final classification neural network on the VGG16 output in order to determine which of the 10 animals the image is, dropout weights are standard
classification_model = Sequential() #Sequential model for ease of implementation
classification_model.add(Flatten(input_shape=vgg16_model.output_shape[1:])) #Simplify VGG16 output to 1D vector
classification_model.add(Dropout(0.1)) #Dropout layer to reduce parameters
classification_model.add(Dense(256, activation='relu')) #Relu function to clean up VGG16 output
classification_model.add(Dropout(0.1)) #Dropout layer to reduce parameters
classification_model.add(Dense(10, activation = 'softmax')) #Softmax for one of 10 possible classifications

#We create the final model using the Model command, taking the input as that of VGG16 and the output as the classification NN with input as output of VGG16
model = Model(inputs=vgg16_model.inputs, outputs=classification_model(vgg16_model.output)) 

#Create the model using standard, robust gradient descent optimizer having added momentum, and using categorical_crossentropy (instead of sparse since 2D target)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(lr=1e-3, momentum=0.9), metrics = ['accuracy'])
model.summary() #Print model configuration for examination

#Fit CNN to training set and perform validation with the dev set, using previously established learning rate reduction parameters
history = model.fit_generator(train_flow, epochs = 12, validation_data = dev_flow, callbacks=[ModelCheckpoint('VGG16.model', monitor='val_acc'), learning_rate_reduction])

#Create plots of the training performance using the monitors that were established for learning rate and the model fitting
epochs = range(1, len(history.history['accuracy'])+1) #compute total number of epochs
train_loss_values = history.history['loss'] #Loss values for training set
dev_loss_values = history.history['val_loss'] #Loss values for dev set
train_accuracy_values = history.history['accuracy'] #Accuracy values for training set
dev_accuracy_values = history.history['val_accuracy'] #Accuracy values for dev set

#Create two side-by-side subplots in order to visualize the model's training performance
f, ax = plt.subplots(nrows=1, ncols = 2, figsize=(20,5)) 

#Create first subplot for training and validation loss 
ax[0].plot(epochs, train_loss_values,  marker='v', color = 'magenta', label='Training Loss')
ax[0].plot(epochs, dev_loss_values, marker='v', color = 'green', label='Validation Loss')
ax[0].set_title('Training & Validation Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='best')
ax[0].grid(True)

#Create second subplot for training and validation accuracy
ax[1].plot(epochs, train_accuracy_values, marker='^', color = 'magenta', label='Training Accuracy')
ax[1].plot(epochs, dev_accuracy_values, marker='^', color = 'green', label='Validation Accuracy')
ax[1].set_title('Training & Validation Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend(loc='best')
ax[1].grid(True)

#Save the plots for future reference as they will not be immediately shown during program run
f.savefig('AccuracyAndLossPlot.eps', format='eps') 
f.savefig('AccuracyAndLossPlot.png', format='png')

#Delete values for clean up memory and for efficiency of program
del epochs, train_loss_values, dev_loss_values, train_accuracy_values, dev_accuracy_values

#Evaluate CNN accuracy on the final test set and print accuracy
score = model.evaluate(test_flow)
print("Test Accuracy ", score[1]*100, "%")
