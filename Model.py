import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Input,Flatten, Conv2D, MaxPooling2D,UpSampling2D,add, Concatenate, concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import optimizers

from keras.initializers import RandomNormal
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD




def MCNN(pretrained_weights = None,input_size = (480,640,3)):
    inputs = Input(input_size)

    # Column m
    conv_m = Conv2D(20, (7,7),padding='same',activation='relu',trainable=True)(inputs)
    conv_m = MaxPooling2D(pool_size=(2,2))(conv_m)
    conv_m = (conv_m)
    conv_m = MaxPooling2D(pool_size = (2, 2))(conv_m)
    conv_m = Conv2D(20, (5,5),padding = 'same', activation = 'relu',trainable=True)(conv_m)
    conv_m = Conv2D(10, (5,5),padding = 'same', activation = 'relu',trainable=True)(conv_m)
    #conv_m = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_m)

    # Column s
    conv_s = Conv2D(24, (5,5),padding = 'same', activation = 'relu',trainable=True)(inputs)
    conv_s = MaxPooling2D(pool_size = (2, 2))(conv_s)
    conv_s = (conv_s)
    conv_s = Conv2D(48, (3,3),padding = 'same', activation = 'relu',trainable=True)(conv_s)
    conv_s = MaxPooling2D(pool_size = (2,2))(conv_s)
    conv_s = Conv2D(24, (3,3),padding = 'same', activation = 'relu',trainable=True)(conv_s)
    conv_s = Conv2D(12, (3,3),padding = 'same', activation = 'relu',trainable=True)(conv_s)
    #conv_s = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_s)

    # Column L
    conv_l = Conv2D(16, (9, 9), padding = 'same', activation = 'relu',trainable=True)(inputs)
    conv_l = MaxPooling2D(pool_size = (2, 2))(conv_l)
    conv_l = (conv_l)
    conv_l = Conv2D(32, (7, 7), padding = 'same', activation = 'relu',trainable=True)(conv_l)
    conv_l = MaxPooling2D(pool_size = (2, 2))(conv_l)
    conv_l = Conv2D(16, (7, 7), padding = 'same', activation = 'relu',trainable=True)(conv_l)
    conv_l = Conv2D(8, (7, 7), padding = 'same', activation = 'relu',trainable=True)(conv_l)
    #conv_l = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_l)

    conv_merge = Concatenate(axis = 3)([conv_m, conv_s, conv_l])
    result = Conv2D(1, (1, 1), padding = 'same')(conv_merge)
    result = UpSampling2D((2, 2))(result)
    result = UpSampling2D((2, 2))(result)

    model = Model(input = inputs, output = result)
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse', metrics=['mae'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)


    model_name = 'MCNN'

    return model,model_name

def SCNN(pretrained_weights = None,input_size = (480,640,3)):
    inputs = Input(input_size)

    conv2_1 = Conv2D(20, (7, 7), activation='relu', padding='same')(inputs)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
    conv2_2 = Conv2D(40, (5, 5), activation='relu', padding='same')(pool2_1)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv2_3 = Conv2D(20, (5, 5), activation='relu', padding='same')(pool2_2)
    pool2_3 = MaxPooling2D(pool_size=(2, 2))(conv2_3)
    conv2_4 = Conv2D(10, (5, 5), activation='relu', padding='same')(pool2_3)
    pool2_4 = MaxPooling2D(pool_size=(2,2))(conv2_4)

    up3_1 = UpSampling2D((2, 2))(conv2_2)
    layer_inter_1 = Conv2D(1, (1, 1), activation='linear')(up3_1)

    up4_1 = UpSampling2D((2, 2))(conv2_3)
    up4_2 = UpSampling2D((2, 2))(up4_1)
    layer_inter_2 = Conv2D(1, (1, 1), activation='linear')(up4_2)

    up5_1 = UpSampling2D((2, 2))(conv2_4)
    up5_2 = UpSampling2D((2, 2))(up5_1)
    up5_3 = UpSampling2D((2, 2))(up5_2)

    layer_inter_3 = Conv2D(1, (1, 1), activation='linear')(up5_3)
    layer_final = layer_inter_1+layer_inter_2+layer_inter_3


    model = Model(input = inputs, output = layer_final)
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse', metrics=['mae'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)


    model_name = 'Single_column'

    return model,model_name

def VGG16(pretrained_weights = None,input_size = (480,640,3)):
    inputs = Input(input_size)

    # VGG
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3_3)
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4_3)
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv5_3)


    up1 = UpSampling2D((2, 2))(pool2)
    #conv4_1 = Conv2D(40, (5, 5), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(up1)
    up3 = UpSampling2D((2, 2))(up2)
    up4 = UpSampling2D((2, 2))(up3)
    up5 = UpSampling2D((2, 2))(up4)
    #conv5_1 = Conv2D(20, (7, 7), activation='relu', padding='same')(up2)
    layer_final = Conv2D(1, (1, 1), activation='linear')(up5)

    model = Model(input = inputs, output = layer_final)
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse', metrics=['mae'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    model_name = 'VGG16'


    return model,model_name

    
def unet(pretrained_weights = None,input_size = (480,640,3)):

    inputs = Input(input_size)
    
    conv1_1 = Conv2D(64/2, (3, 3), activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(64/2, (3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    
    conv2_1 = Conv2D(128/2, (3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(128/2, (3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    
    conv3_1 = Conv2D(256/2, (3, 3), activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(256/2, (3, 3), activation='relu', padding='same')(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    
    conv4_1 = Conv2D(512/2, (3, 3), activation='relu', padding='same')(pool3)
    conv4_2 = Conv2D(512/2, (3, 3), activation='relu', padding='same')(conv4_1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)
    
    conv5_1 = Conv2D(1024/2, (3, 3), activation='relu', padding='same')(pool4)
    conv5_2 = Conv2D(1024/2, (3, 3), activation='relu', padding='same')(conv5_1)
    
    up6 = Conv2D(512/2, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv5_2))
    up6 = concatenate([conv4_2,up6], axis = 3)
    
    conv6_1 = Conv2D(512/2, (3, 3), activation='relu', padding='same')(up6)
    conv6_2 = Conv2D(512/2, (3, 3), activation='relu', padding='same')(conv6_1)
    
#    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3_2], axis=3)
    up7 = Conv2D(256/2, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6_2))
    up7 = concatenate([conv3_2,up7], axis = 3)
    conv7_1 = Conv2D(256/2, (3, 3), activation='relu', padding='same')(up7)
    conv7_2 = Conv2D(256/2, (3, 3), activation='relu', padding='same')(conv7_1)
    
#    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7_2), conv2_2], axis=3)
    up8 = Conv2D(128/2, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7_2))
    up8 = concatenate([conv2_2,up8], axis = 3)
    conv8_1 = Conv2D(128/2, (3, 3), activation='relu', padding='same')(up8)
    conv8_2 = Conv2D(128/2, (3, 3), activation='relu', padding='same')(conv8_1)
    
#    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8_2), conv1_2], axis=3)
    up9 = Conv2D(64/2, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8_2))
    up9 = concatenate([conv1_2,up9], axis = 3)
    conv9_1 = Conv2D(64/2, (3, 3), activation='relu', padding='same')(up9)
    conv9_2 = Conv2D(64/2, (3, 3), activation='relu', padding='same')(conv9_1)
    
    conv10 = Conv2D(1, (1, 1), activation='linear')(conv9_2)
    
    '''
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    '''
    model = Model(input = inputs, output = conv10)
    
    model.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse', metrics=['mae'])
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    model_name = 'U_net'
    return  model,model_name



#############################################################################################################
# Neural network model : VGG + Conv
#
def CrowdNet():
    #Variable Input Size
#    rows = None
#    cols = None

    rows = 480
    cols = 640
    
    input_size = (480,640)
    inputs = Input(input_size)
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)
    pool1   = MaxPooling2D(pool_size=(2,2))(conv1_2)
    
    conv2_1 = Conv2D(128,(3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(128,(3, 3), activation='relu', padding='same')(conv2_1)
    pool2   = MaxPooling2D(pool_size=(2,2))(conv2_2)
    
    conv3_1 = Conv2D(256,(3, 3), activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(256,(3, 3), activation='relu', padding='same')(conv3_1)
    conv3_3 = Conv2D(256,(3, 3), activation='relu', padding='same')(conv3_2)
    pool3   = MaxPooling2D(pool_size=(2,2))(conv3_3)
    
    conv4_1 = Conv2D(512,(3, 3), activation='relu', padding='same')(pool3)
    conv4_2 = Conv2D(512,(3, 3), activation='relu', padding='same')(conv4_1)
    conv4_3 = Conv2D(512,(3, 3), activation='relu', padding='same')(conv4_2)
    pool4   = MaxPooling2D(pool_size=(2,2))(conv4_3)
    
    up1 = UpSampling2D((2,2))(pool4)
    up2 = UpSampling2D((2,2))(up1)
    up3 = UpSampling2D((2,2))(up2)
    up4 = UpSampling2D((2,2))(up3)
    
    conv5_1 = Conv2D(512,(3, 3), activation='relu',dilation_rate = 2, padding='same')(up4)
    conv5_2 = Conv2D(512,(3, 3), activation='relu',dilation_rate = 2,padding='same')(conv5_1)
    conv5_3 = Conv2D(512,(3, 3), activation='relu',dilation_rate = 2,padding='same')(conv5_2)
    conv5_4 = Conv2D(256,(3, 3), activation='relu',dilation_rate = 2,padding='same')(conv5_3)
    conv5_5 = Conv2D(128,(3, 3), activation='relu',dilation_rate = 2, padding='same')(conv5_4)
    conv5_6 = Conv2D(64 ,(3, 3), activation='relu',dilation_rate = 2, padding='same')(conv5_5)
    layer_final = Conv2D(1  ,(3, 3), activation='linear',dilation_rate = 2, padding='same')(conv5_6)
    
    model = Model(input = inputs, output = layer_final)
    #Batch Normalisation option
    '''
    batch_norm = 0
    kernel = (3, 3)
    init = RandomNormal(stddev=0.01)
    model = Sequential()

    #custom VGG:
    
    if(batch_norm):
        model.add(Conv2D(64, kernel_size = kernel, input_shape = (rows,cols,3),activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())

    else:
        model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same',input_shape = (rows, cols, 3), kernel_initializer = init))
        model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))




    #Conv2D
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(256, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(1, (1, 1), activation='relu', dilation_rate = 1, kernel_initializer = init, padding = 'same'))

    #Upsampling
    model.add(UpSampling2D((2, 2)))
    model.add(UpSampling2D((2, 2)))
    model.add(UpSampling2D((2, 2)))
    '''
    sgd = SGD(lr = 1e-7, decay = (5*1e-4), momentum = 0.95)
#    model.compile(optimizer=sgd, loss=euclidean_distance_loss, metrics=['mse'])
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse', metrics=['mae'])
#    model = init_weights_vgg(model)


    #### Naming this model
    model_name = 'CrowdNet'

    return model,model_name
