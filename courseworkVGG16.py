# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:03:33 2019

@author: Mark Thieu
"""

import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn


from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,roc_curve,auc

from PIL import Image
from PIL import Image as pil_image
from PIL import ImageDraw

from time import time
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from IPython.display import SVG

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread


from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger,ReduceLROnPlateau,LearningRateScheduler


#define loss/accuracy plot
def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train accuracy")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation accuracy")
    ax[0].legend()
    ax[1].legend()
#define label assignemtn
def label_assignment(img,label):
    return label
#define traning data
def training_data(label,data_dir):
    for img in tqdm(os.listdir(data_dir)):
        label = label_assignment(img,label)
        path = os.path.join(data_dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(imgsize,imgsize))
        
        X.append(np.array(img))
        Z.append(str(label))
        
#initialize data directories  
chihuahua_dir = r'./Images/n02085620-Chihuahua'
chow_dir =r'./Images/n02112137-chow'
norwegian_elkhound_dir = r'./Images/n02091467-Norwegian_elkhound'
pekinese_dir =r'./Images/n02086079-Pekinese'
labrador_dir = r'./Images/n02099712-Labrador_retriever'
komondor_dir = r'./Images/n02105505-komondor'
german_shepherd_dir = r'./Images/n02106662-German_shepherd'
husky_dir = r'./Images/n02110185-Siberian_husky'
afghan_hound_dir = r'./Images/n02088094-Afghan_hound'
basset_dir = r'./Images/n02088238-basset'
X = []
Z = []
imgsize = 150
#assign data directories to trainig data
training_data('chihuahua',chihuahua_dir)
training_data('chow',chow_dir)
training_data('norwegian_elkhound',norwegian_elkhound_dir)
training_data('pekinese',pekinese_dir)
training_data('labrador',labrador_dir)
training_data('komondor',komondor_dir)
training_data('german_shepherd',german_shepherd_dir)
training_data('husky',husky_dir)
training_data('afghan_hound',afghan_hound_dir)
training_data('basset',basset_dir) 

#modify data to categorical format
label_encoder= LabelEncoder()
Y = label_encoder.fit_transform(Z)
y = to_categorical(Y,10)
X = np.array(X)
X=X/255
#split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#values to be used later
batch_size = 128
num_classes = y.shape[1]
epochs = 20
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    #zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

datagen.fit(X_train)

fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Dog: '+Z[l])
        
plt.tight_layout()
#base model 
base_model = VGG16(include_top=False,
                  input_shape = (imgsize,imgsize,3),
                  weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
    
for layer in base_model.layers:
    print(layer,layer.trainable)
#initialise model
model = Sequential()
#add the VGG16 to our model
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))


#using checkpointing
checkpoint = ModelCheckpoint(
    './base.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)

csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)

callbacks = [checkpoint,csvlogger,reduce]

#Optimizer
opt = SGD(lr=1e-4,momentum=0.99)
opt1 = Adam(lr=1e-2)
#Compile cnn
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
#Train the model
history = model.fit_generator(
    datagen.flow(X_train,y_train,batch_size=batch_size),
    validation_data  = (X_test,y_test),
    #validation_steps = 1000,
    #steps_per_epoch  = 1000,
    epochs = 20, 
    verbose = 1,
    callbacks=callbacks
)
#make predictions (will give a probability distribution)
pred = model.predict(X_test)
#now pick the most likely outcome
pred = np.argmax(pred,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#and calculate accuracy
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))
#plot loss/accuracy
show_final_history(history)
model.load_weights('./base.model')
model_score = model.evaluate(X_test,y_test)
print("Model Test Loss:",model_score[0])
print("Model Test Accuracy:",model_score[1])
#save model    
model.save("model.h5")
print("Weights Saved")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    