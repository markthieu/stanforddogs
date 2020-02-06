# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 23:42:25 2019

@author: Mark Thieu
"""
#import dependenies
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator


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
    
    
#define label assignment
def label_assignment(img,label):
    return label
#define training data
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

#assign data directories to tranining data
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
#transform features to categorical data
label_encoder= LabelEncoder()
y = label_encoder.fit_transform(Z)
y = to_categorical(y,10)
X = np.array(X)
X=X/255
#split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#values to be used later
batch_size = 128
num_classes = y.shape[1]
epochs = 20
save_dir = './' 
model_name = 'keras_trained_model.h5'
#image data generator, based on tutorial 8
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


model = Sequential()
#Convolution
model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', input_shape= (150, 150, 3), activation='relu'))
#Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#Adding a second convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Adding a third convolutional layer
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening
model.add(Flatten())
#Full connection
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes, activation = 'softmax'))
model.summary()


#Compile the model
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
    epochs = epochs, 
    verbose = 1,
)

#make predictions (will give a probability distribution)
pred = model.predict(X_test)
#now pick the most likely outcome
pred = np.argmax(pred,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#and calculate accuracy
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))
#plot,represent loss and accuracy
show_final_history(history)
model_score =model.evaluate(X_test,y_test)  
print("Model Test Loss:",model_score[0])
print("Model Test Accuracy:",model_score[1]) 
# Save model and weights
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)