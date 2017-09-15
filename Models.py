
# coding: utf-8

# In[1]:

import csv 
import zipfile
import cv2
import numpy as np
import tensorflow as tf 


# In[2]:

# # Downloading the Car Data zip file

# with zipfile.ZipFile("10data.zip","r") as zip_ref:
#     zip_ref.extractall()


# In[2]:

lines = []

# Opening the csv files within Car Data

with open('./9data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i in reader:
        lines.append(i)
        


# In[3]:

from sklearn.model_selection import train_test_split


train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# In[4]:

import random
def augment_image(image):
    image = image.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (image[:,:,0] + value) > 255 
    if value <= 0:
        mask = (image[:,:,0] + value) < 0
    image[:,:,0] += np.where(mask, 0, value)
    
    
    h,w = image.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        image[:,0:mid,0] *= factor
    else:
        image[:,mid:w,0] *= factor
    return new_img.astype(np.uint8)


# In[5]:


from sklearn.utils import shuffle
 
import skimage

def generator(samples, batch_size):
    num_samples = len(samples)
    samples = shuffle(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #Adding in the left and right cameras
                for i in range(0, 2):
                    correction = 0.2
                    if i == 0:
                        if batch_sample[3] == 'steering':
                            continue
                        else:
                            name = './9data/IMG/'+ batch_sample[0].split('/')[-1]
                            center_image = cv2.imread(name)
#                             center_image = cv2.GaussianBlur(center_image, (3,3), 0)
                            center_image = cv2.resize(center_image, (80, 40), interpolation = cv2.INTER_AREA)
                            #Preprocess 
                            center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                            center_angle = float(batch_sample[3])
                            
                            center_image = augment_image(center_image)
                            
                            images.append(center_image)
                            angles.append(center_angle)
 
                    #Left Camera
                    if i == 1:
                        if batch_sample[3] == 'steering':
                            continue
                        else:
                            name = './9data/IMG/'+ batch_sample[0].split('/')[-1]
                            left_image = cv2.imread(name)
#                             left_image = cv2.GaussianBlur(left_image, (3,3), 0)
                            left_image = cv2.resize(left_image, (80, 40), interpolation = cv2.INTER_AREA)
                            #Preprocess
                            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
                            left_angle = float(batch_sample[3]) + correction
                        
                            left_image = augment_image(left_image)
                            images.append(left_image)
                            angles.append(left_angle)
                            

                    #Right Camera
                    if i == 2:
                        if batch_sample[3] == 'steering':
                            continue
                        else:
                            name = './9data/IMG/'+ batch_sample[0].split('/')[-1]
                            right_image = cv2.imread(name)
#                             right_image = cv2.GaussianBlur(right_image, (3,3), 0)
                            right_image = cv2.resize(right_image, (80, 40), interpolation = cv2.INTER_AREA)

                            #Preprocess
                            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
                            right_angle = float(batch_sample[3]) - 0.3
                        
                            right_image = augment_image(right_image)
                            
                            images.append(right_image)
                            angles.append(right_angle)
                           
                
                

            augmented_images, augmented_measurements = [], []
                
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)



            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            
            
            yield shuffle(X_train, y_train)
            


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size= 20 )
validation_generator = generator(validation_samples, batch_size= 20)    


# In[6]:

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D


# In[29]:

model = Sequential()

model.add(Lambda(lambda x: x/127. - 1.0, input_shape=(40, 80, 3)))
model.add(Cropping2D(cropping=((18, 7), (0, 0))))

model.add(Convolution2D(5, 3, 3, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.9))
model.add(Convolution2D(10, 3, 3, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.8))
model.add(Convolution2D(20, 3, 3, activation='relu'))
# model.add(Dropout(0.6))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Dropout(0.5))
model.add(Flatten())
# model.add(Dense(100))
# model.add(Dropout(0.3))
# model.add(Dense(50))
# model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')

print('saved')


# In[ ]:



