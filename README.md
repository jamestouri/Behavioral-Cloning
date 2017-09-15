# Behaviorial Cloning Project

This project contains the data set and code written in Python for a autonomous driving simulator.  The car was to be trained manually then trained through a deep learning neural network. [Here is the video of the finished project](https://youtu.be/WUrMyoc62Ro)

## The Data Set


Udacity provided a data set to start with.  It was a good starting but had little recovery data, which I spent most of the time training the car for. 

Here is an image from the Data Set:

![](https://github.com/jamestouri/Behavioral-Cloning/blob/master/img1.png)

The images are 160 x 320 x 3

## Measurement for Steering

There were 3 cameras on the car at all times.  I adjusted the measurement on the right side more than the one on the left due to the car constantly moving to the right.

![Center Image](https://github.com/jamestouri/Behavioral-Cloning/blob/master/centerimg.png)
![Left Image](https://github.com/jamestouri/Behavioral-Cloning/blob/master/leftimg.png)
![Right Image](https://github.com/jamestouri/Behavioral-Cloning/blob/master/rightimg.png)

These are three images that were at the same location of the car.  Adjusting the measurements to have a kind of recovery data was very beneficial, plus having more data. 

## Preprocessing

At first I was going through the NVIDIA Blog for self driving, following the convolutional neural and their image resizing.  Overall it didn't quite work for me, however I did convert BGR to YUV as they did in their blog.  

I resized my images to 40 x 80 x 3 and cropped the image by taking off 18 from top and 7 from bottom. 


## Augmenting the Image

I added random brightness and random shadow, both methods learned from the previous project, The Traffic Sign Classifier. 
I also rotated all the images to mirror the original. 



## Architecture Rundown 

|Layer|Description|
|-----|-----------|
|Input|40 x 80 x 3|
|CNN Layer|5 x 3 x 3|
|Dropout|0.9|
|CNN Layer|10 x 3 x 3|
|Dropout|0.8|
|CNN Layer|20 x 3 x 3|
|Dropout|0.5|
|Flatten| |
|Fully Connected Layer|1|

## Training Set
Optimizer: Adam

Number of Epochs: 5

Images Generated: 18000

Also used Kera's Fit Generator to speed up training

## Recap
Very fun project. My favorite so far



