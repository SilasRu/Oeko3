# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 22:19:54 2019

@author: Dario
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from PIL import Image
import glob

image_train_bers = []
for filename in glob.glob("C:\\Users\\Dario\\Documents\\GitHub\\Oeko3\\data\\external\\train\\Train\\spectrograms_berset\\*.png"):
    im=Image.open(filename)
    image_train_bers.append(im)

image_train_goess = []
for filename in glob.glob("C:\\Users\\Dario\\Documents\\GitHub\\Oeko3\\data\\external\\train\\Train\\spectrograms_goess\\*.png"):
    im=Image.open(filename)
    image_train_goess.append(im)

image_test_bers = []
for filename in glob.glob("C:\\Users\\Dario\\Documents\\GitHub\\Oeko3\\data\\external\\train\\Test\\Berset_t\\*.png"):
    im=Image.open(filename)
    image_test_bers.append(im)

image_test_goess = []
for filename in glob.glob("C:\\Users\\Dario\\Documents\\GitHub\\Oeko3\\data\\external\\train\\Test\\Goess_t\\*.png"):
    im=Image.open(filename)
    image_test_goess.append(im)



plt.imshow(image_train_bers[1])
plt.imshow(image_train_goess[1])
lb = len(image_train_bers)
lg = len(image_train_goess)

shape = (lb+lg),2

y_train = np.zeros(shape)
#goess in first column
y_train[0:lb,1] = 1
y_train[lb:,0] = 1



im_array_tot = []
for i in range(lb):
    im_b = np.array(image_train_bers[i])
    im_array_tot.append(im_b)

im_array_goes = []
for i in range(lg):
    im_g = np.array(image_train_goess[i])
    im_array_goes.append(im_g)
    
    
len(im_array_tot)




im_array_tot.extend(im_array_goes)

np.array(im_array_goes).ndim


np.array(im_array_tot[0]).shape

x = np.reshape((im_array_tot[0]),(28,28))

np.a

x_train = np.array(im_array_tot).reshape(np.array(im_array_tot[0]).shape(28,28,1))

x_train[0].size()
shape(im_array_tot)



plt.imshow(im_array_tot[0])
