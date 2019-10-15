# -*- coding: utf-8 -*-
"""
Created on Thu Sep 05 17:09:21 2019

@author: SornSook
"""

import keras
from keras.models import Sequential
from keras.layers import Input,Flatten, Conv2D, MaxPooling2D,UpSampling2D,add, concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import optimizers
import keras.backend as K   

import numpy as np

# The loss function is MSE and MAE 
def single_column_v1_3(input_size=(480,640,3)):
    factor = 4
    inputs = Input(input_size)
    # Encoder
    conv_1 = Conv2D(8*factor, (7, 7), activation='relu', padding='same',trainable =True)(inputs)
    pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
    
    conv_2 = Conv2D(16*factor, (5, 5), activation='relu', padding='same',trainable = True)(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
    
    conv_3 = Conv2D(32*factor, (5, 5), activation='relu', padding='same',trainable = True)(pool_2)
    pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)
    
    conv_4 = Conv2D(64*factor, (5, 5), activation='relu', padding='same',trainable = True)(pool_3)
    
    
    up_5   = UpSampling2D(size = (2,2))(conv_4)
    conv_5 = Conv2D(32*factor, (5, 5), activation='relu', padding='same',trainable = True)(up_5)
    up_6   = UpSampling2D(size = (2,2))(conv_5)
    conv_6 = Conv2D(16*factor, (5, 5), activation='relu', padding='same',trainable = True)(up_6)
    up_7   = UpSampling2D(size = (2,2))(conv_6)
    conv_7 = Conv2D(8*factor, (7, 7), activation='relu', padding='same',trainable = True)(up_7)
    
    layer_final = Conv2D(1, (1, 1), activation='linear')(conv_7)
    
    model = Model(input = inputs, output = layer_final)
    
    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='mse', metrics=['mae'])
#    model.compile(optimizer = optimizers.Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model


############## Loss function ##################
def custom_loss_function(y_true, y_pred):
    
    # calculate density loss
    den_loss = K.mean(K.abs(y_true-y_pred))
    # calculate count loss
    count_true = np.squeeze(y_true).sum()
    count_pred = np.squeeze(y_pred).sum()
    
    count_loss = K.mean(K.abs(count_true-count_pred)/(count_true+1))
    
    total_loss = count_loss + den_loss 
    return total_loss


# Change the loss function which related to density map and count
def single_column_v5_7(input_size=(480,640,3)):
    
    inputs = Input(input_size)
    # Encoder
    conv_1 = Conv2D(64, (7, 7), activation='relu', padding='same',trainable = True)(inputs)
    pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
    
    conv_2 = Conv2D(128, (5, 5), activation='relu', padding='same',trainable = True)(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
    
    conv_3 = Conv2D(256, (5, 5), activation='relu', padding='same',trainable = True)(pool_2)
    pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)
    
    conv_4 = Conv2D(512, (5, 5), activation='relu', padding='same',trainable = True)(pool_3)
    
    
    up_5   = UpSampling2D(size = (2,2))(conv_4)
    conv_5 = Conv2D(256, (5, 5), activation='relu', padding='same',trainable = True)(up_5)
    up_6   = UpSampling2D(size = (2,2))(conv_5)
    conv_6 = Conv2D(128, (5, 5), activation='relu', padding='same',trainable = True)(up_6)
    up_7   = UpSampling2D(size = (2,2))(conv_6)
    conv_7 = Conv2D(64, (7, 7), activation='relu', padding='same',trainable = True)(up_7)
    
    layer_final = Conv2D(1, (1, 1), activation='linear')(conv_7)
    
    model = Model(input = inputs, output = layer_final)
    
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss=custom_loss_function, metrics=['mae'])
    model.summary()
    
    return model

def skip_connection (input_size = (480,640,3)):
    
    inputs = Input(input_size)
    conv_1 = Conv2D(64, (9, 9), activation='relu', padding='same',trainable = True)(inputs)
    pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)


    convM_11= Conv2D(32, (7, 7), activation='relu', padding='same',trainable = True)(pool_1)
    convM_12= Conv2D(32, (7, 7), activation='relu', padding='same',trainable = True)(concatenate([pool_1, convM_11], axis=-1))
    convM_13= Conv2D(16, (7, 7), activation='relu', padding='same',trainable = True)(concatenate([pool_1, convM_12], axis=-1))
    convM_14= Conv2D(16, (7, 7), activation='relu', padding='same',trainable = True)(convM_13)
    pool_2 = MaxPooling2D(pool_size=(2,2))(convM_14)

    convM_21= Conv2D(32, (7, 7), activation='relu', padding='same',trainable = True)(pool_2)
    convM_22= Conv2D(32, (7, 7), activation='relu', padding='same',trainable = True)(concatenate([pool_2, convM_21], axis=-1))
    convM_23= Conv2D(16, (7, 7), activation='relu', padding='same',trainable = True)(concatenate([pool_2, convM_22], axis=-1))
    convM_24= Conv2D(16, (5, 5), activation='relu', padding='same',trainable = True)(convM_23)

    convM_31= Conv2D(32, (7, 7), activation='relu', padding='same',trainable = True)(convM_24)
    convM_32= Conv2D(32, (7, 7), activation='relu', padding='same',trainable = True)(concatenate([convM_24, convM_31], axis=-1))
    convM_33= Conv2D(16, (7, 7), activation='relu', padding='same',trainable = True)(concatenate([convM_24, convM_32], axis=-1))
    convM_34= Conv2D(16, (5, 5), activation='relu', padding='same',trainable = True)(convM_33)

    convM_41= Conv2D(32, (7, 7), activation='relu', padding='same',trainable = True)(convM_34)
    convM_42= Conv2D(32, (7, 7), activation='relu', padding='same',trainable = True)(concatenate([convM_34, convM_41], axis=-1))
    convM_43= Conv2D(16, (7, 7), activation='relu', padding='same',trainable = True)(concatenate([convM_34, convM_42], axis=-1))
    convM_44= Conv2D(16, (5, 5), activation='relu', padding='same',trainable = True)(convM_43)

    conv_2 = Conv2D(128, (1, 1), activation='relu', padding='same',trainable = True)(convM_44)
    conv_3 = Conv2D(1, (1, 1), activation='relu', padding='same',trainable = True)(conv_2)

    up_1   = UpSampling2D(size = (2,2))(conv_3)
    up_2   = UpSampling2D(size = (2,2))(up_1)

    layer_final = up_2

    model = Model(input = inputs, output = layer_final)
    
    # model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse', metrics=['mae'])
    model.compile(optimizer = optimizers.Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model

##############################################################################################
def skip_net (input_size = (480,640,3)):
    factor  = 4
    inputs  = Input(input_size)
    conv_11 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(inputs)
    conv_12 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_11)
    conv_13 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_12)
    # conv_14 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_13)
    pool_1  = MaxPooling2D(pool_size=(2,2))(conv_13)
    
    conv_21 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(pool_1)
    conv_22 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_21)
    # conv_23 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_22)
    pool_2 = MaxPooling2D(pool_size=(2,2))(conv_22)
    
    conv_31 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(pool_2)
    conv_32 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_31)
    # conv_33 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_32)
    pool_3 = MaxPooling2D(pool_size=(2,2))(conv_32)
    
    conv_41 = Conv2D(64*factor, (3, 3), activation='relu', padding='same',trainable = True)(pool_3)
    conv_42 = Conv2D(64*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_41)
    # conv_43 = Conv2D(64*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_42)
    
    up_5   = UpSampling2D(size = (2,2))(conv_42)
    conv_51 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(up_5)
    conv_52 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_51)
    # conv_53 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_52)
    
    up_6   = UpSampling2D(size = (2,2))(conv_52)
    conv_61 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(up_6)
    conv_62 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_61)
    # conv_63 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_62)

    up_7   = UpSampling2D(size = (2,2))(conv_62)
    conv_71 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(up_7)
    conv_72 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_71)
    conv_73 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_72)
    # conv_74 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_73)


    layer_final = Conv2D(1, (1, 1), activation='linear')(conv_73)
    
    model = Model(input = inputs, output = layer_final)
    
    # model.compile(optimizer=optimizers.Adam(lr=1e-3), loss=custom_loss_function, metrics=['mae'])
    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='mse', metrics=['mae'])
    model.summary()
    
    return model

############################################################################################
def skip_net_forward (input_size = (480,640,3)):
    factor = 4
    inputs  = Input(input_size)
    conv_11 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(inputs)
    conv_12 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_11)
    conv_13 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_12)
    pool_1  = MaxPooling2D(pool_size=(2,2))(conv_13)
    
    conv_21 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(pool_1)
    conv_22 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_21)
    pool_2 = MaxPooling2D(pool_size=(2,2))(conv_22)
    
    conv_31 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(pool_2)
    conv_32 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_31)
    pool_3 = MaxPooling2D(pool_size=(2,2))(conv_32)
    
    conv_41 = Conv2D(64*factor, (3, 3), activation='relu', padding='same',trainable = True)(pool_3)
    conv_42 = Conv2D(64*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_41)
    
    up_5   = UpSampling2D(size = (2,2))(conv_42)
    conv_51 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(concatenate([up_5, conv_32], axis=-1))
    conv_52 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_51)
    
    up_6   = UpSampling2D(size = (2,2))(conv_52)
    conv_61 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(concatenate([up_6, conv_22], axis=-1))
    conv_62 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_61)
    
    up_7   = UpSampling2D(size = (2,2))(conv_62)
    conv_71 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(concatenate([up_7, conv_13], axis=-1))
    conv_72 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_71)
    conv_73 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_72)
    
    layer_final = Conv2D(1, (1, 1), activation='linear')(conv_73)
    
    model = Model(input = inputs, output = layer_final)
    
    # model.compile(optimizer=optimizers.Adam(lr=1e-3), loss=custom_loss_function, metrics=['mae'])
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse', metrics=['mae'])
    model.summary()
    
    return model

############################################################################################

#dummy_1 = np.zeros()

def skip_net_backward(out1,out2,out3,input_size = (480,640,3)):
    factor = 3
    inputs  = Input(input_size)
    conv_11 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(inputs)
    conv_12 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_11)
    conv_13 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_12)
    pool_1  = MaxPooling2D(pool_size=(2,2))(concatenate([conv_13, out1]),axis=-1)
    
    conv_21 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(pool_1)
    conv_22 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_21)
    pool_2 = MaxPooling2D(pool_size=(2,2))(concatenate([conv_22, out2]),axis=-1)
    
    conv_31 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(pool_2)
    conv_32 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_31)
    pool_3 = MaxPooling2D(pool_size=(2,2))(concatenate([conv_32, out3]),axis=-1)
    
    conv_41 = Conv2D(64*factor, (3, 3), activation='relu', padding='same',trainable = True)(pool_3)
    conv_42 = Conv2D(64*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_41)
    
    up_5   = UpSampling2D(size = (2,2))(conv_42)
    conv_51 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(up_5)
    conv_52 = Conv2D(32*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_51)
    o3                  = conv_52
    
    
    up_6   = UpSampling2D(size = (2,2))(conv_52)
    conv_61 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(up_6)
    conv_62 = Conv2D(16*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_61)
    o2                  = conv_62
    
    
    up_7   = UpSampling2D(size = (2,2))(conv_62)
    conv_71 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(up_7)
    conv_72 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_71)
    conv_73 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_72)
    o1                  = conv_73 
    
    layer_final = Conv2D(1, (1, 1), activation='linear')(conv_73)
    
    model = Model(input = inputs, output = layer_final)
    
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss=custom_loss_function, metrics=['mae'])
    model.summary()
    
    return model, o1,o2,o3

def dummy(input_size = (480,640,3)):
    
    factor = 3
    inputs  = Input(input_size)
    inputs  = Input(input_size)
    conv_11 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(inputs)
    conv_12 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_11)
    conv_13 = Conv2D(8*factor, (3, 3), activation='relu', padding='same',trainable = True)(conv_12)
    
    model = Model(input = inputs, output = conv_13)
    
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss=custom_loss_function, metrics=['mae'])
    model.summary()
    
    return model
    
def tra_count (input_size = (480,640,3)):
    inputs  = Input(input_size)
    Conv_11 = Conv2D(16, (7,7), activation = 'relu',padding = 'same', trainable =True)(inputs)
    pool_11 = MaxPooling2D(pool_size=(2,2))(Conv_11)
    Conv_12 = Conv2D(16, (5,5), activation = 'relu',padding = 'same', trainable =True)(pool_11)
    pool_12 = MaxPooling2D(pool_size=(2,2))(Conv_12)
    Conv_13 = Conv2D(16, (5,5), activation = 'relu',padding = 'same', trainable =True)(pool_12)
    pool_13 = MaxPooling2D(pool_size=(2,2))(Conv_13)
    
    
    Conv_21 = Conv2D(24, (5,5), activation = 'relu',padding = 'same', trainable =True)(inputs)
    pool_21 = MaxPooling2D(pool_size=(2,2))(Conv_21)
    Conv_22 = Conv2D(24, (3,3), activation = 'relu',padding = 'same', trainable =True)(pool_21)
    pool_22 = MaxPooling2D(pool_size=(2,2))(Conv_22)
    Conv_23 = Conv2D(24, (3,3), activation = 'relu',padding = 'same', trainable =True)(pool_22)
    pool_23 = MaxPooling2D(pool_size=(2,2))(Conv_23)
    
    
    Conv_31 = Conv2D(64//4, (3,3), activation = 'relu',padding = 'same', trainable =True)(inputs)
    pool_31 = MaxPooling2D(pool_size=(2,2))(Conv_31)
    Conv_32 = Conv2D(128//4, (3,3), activation = 'relu',padding = 'same', trainable =True)(pool_31)
    pool_32 = MaxPooling2D(pool_size=(2,2))(Conv_32)
    Conv_33 = Conv2D(256//4, (3,3), activation = 'relu',padding = 'same', trainable =True)(pool_32)
    pool_33 = MaxPooling2D(pool_size=(2,2))(Conv_33)
    Conv_34 = Conv2D(512//4, (3,3), activation = 'relu',padding = 'same', trainable =True)(pool_33)
    pool_34 = MaxPooling2D(pool_size=(2,2))(Conv_34)
    Conv_35 = Conv2D(512//4, (3,3), activation = 'relu',padding = 'same', trainable =True)(pool_34)

    
    Conv3 = UpSampling2D(size = (2,2))(Conv_35)
    
    layer_final = concatenate([pool_13, Conv3], axis=-1)
    layer_final = concatenate([layer_final, pool_23], axis=-1)
    
    layer_final = UpSampling2D(size = (2,2))(layer_final)
    layer_final = UpSampling2D(size = (2,2))(layer_final)
    layer_final = UpSampling2D(size = (2,2))(layer_final)
    
    layer_final = Conv2D(1, (1, 1), activation='linear')(layer_final)
    model = Model(input = inputs, output = layer_final)
    
    # model.compile(optimizer=optimizers.Adam(lr=1e-3), loss=custom_loss_function, metrics=['mae'])
    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='mse', metrics=['mae'])
    model.summary()
    
    return model

