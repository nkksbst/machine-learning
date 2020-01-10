# -*- coding: utf-8 -*-
'''
Created on Tue Dec 24 08:12:12 2019
@author: Monikka Busto
Background Segmentation Using U-Net
'''
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image
from config import *
from tqdm import trange

def get_patches(img):
    patches = np.ones((NUM_PATCHES, XY_STRIDE, XY_STRIDE, NUM_OF_CHANNELS), dtype=np.float32)
    
    index = 0
    for x in range(0, int(X_DIM/XY_STRIDE)):
        for y in range(0, int(Y_DIM/XY_STRIDE)):
            patch = img[XY_STRIDE * y : XY_STRIDE * y + XY_STRIDE, XY_STRIDE * x : XY_STRIDE * x + XY_STRIDE]
            patches[index,:,:,0] = patch
            index += 1
    return patches

def build_mask(patches):
    mask =  np.ones((Y_DIM, X_DIM), dtype=np.float32)
    index = 0
    for x in range(0, int(X_DIM/XY_STRIDE)):
        for y in range(0, int(Y_DIM/XY_STRIDE)):
            mask[XY_STRIDE * y : XY_STRIDE * y + XY_STRIDE, XY_STRIDE * x : XY_STRIDE * x + XY_STRIDE] = patches[index,:,:,0]
            index +=1
    return mask        
            
def load_image(file_path):
    im = Image.open(file_path)
    np_im = np.array(im)
    return np_im

def load_data(img_path, mask_path):
    img_paths = glob.glob( img_path + '*')
    mask_paths = glob.glob(mask_path + '*')
    # initialize data container
    x = np.zeros((len(img_paths), SUB_IMG_SIZE, SUB_IMG_SIZE, NUM_OF_CHANNELS), dtype=np.float32)
    y = np.zeros((len(img_paths), SUB_IMG_SIZE, SUB_IMG_SIZE, NUM_OF_CHANNELS), dtype=np.float32)
    # load images and masks
    for i in trange((len(img_paths)), desc='Loading image patches'):
        img = load_image(img_paths[i])
        x[i,:,:,0] = img
        mask = load_image(mask_paths[i])
        y[i,:,:,0] = mask

    y[y == 1] = 255
    # normalize data
    x = x / 255.0
    y = y / 255.0
    
    return x, y    
   
def plot(X, Y):
    index = 395
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(X[index,:,:, 0], cmap='gray')
    ax[0].set_title('Specular')
    ax[1].imshow(Y[index,:,:, 0], interpolation='bilinear', cmap='gray')
    ax[1].set_title('Mask');
    
def conv(tensor, k_filters, k_size = 3, batch_norm = True):
    
    x = Conv2D(filters = k_filters, 
               kernel_size = (k_size, k_size),
               kernel_initializer = 'he_normal', 
               padding = 'same')(tensor)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = k_filters, 
               kernel_size = (k_size, k_size),
               kernel_initializer = 'he_normal', 
               padding = 'same')(tensor)
    x = Activation('relu')(x)
    return x

def build_unet(img, k_filters = 16, dropout = 0.5, batch_norm = True):
    # 128x128x16
    c1 = conv(img, k_filters=k_filters*1, k_size=3, batch_norm=batch_norm)
    # 64x64x16
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    # 64x64x32
    c2 = conv(p1, k_filters=k_filters*2, k_size=3, batch_norm=batch_norm)
    # 32x32x32
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    # 32x32x64
    c3 = conv(p2, k_filters=k_filters*4, k_size=3, batch_norm=batch_norm)
    # 16x16x64
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    # 16x16x128
    c4 = conv(p3, k_filters=k_filters*8, k_size=3, batch_norm=batch_norm)
    # 8x8x128
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    # 8x8x256
    c5 = conv(p4, k_filters=k_filters*16, k_size=3, batch_norm=batch_norm)
    
    # 16x16x256
    u6 = Conv2DTranspose(k_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    # 16x16x128
    c6 = conv(u6, k_filters=k_filters*8, k_size=3, batch_norm=batch_norm)

    # 32x32x128
    u7 = Conv2DTranspose(k_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    # 32x32x64
    c7 = conv(u7, k_filters=k_filters*4, k_size=3, batch_norm=batch_norm)

    # 64x64x64
    u8 = Conv2DTranspose(k_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    # 64x64x32
    c8 = conv(u8, k_filters=k_filters*2, k_size=3, batch_norm=batch_norm)

    # 128x128x32
    u9 = Conv2DTranspose(k_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    # 128x128x16
    c9 = conv(u9, k_filters=k_filters*1, k_size=3, batch_norm=batch_norm)
    
    # 128x128x1
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[img], outputs=[outputs])
    return model


def train(X_train, X_test, y_train, y_test):
    input_img = Input((SUB_IMG_SIZE, SUB_IMG_SIZE, NUM_OF_CHANNELS), name = 'image')
    model = build_unet(input_img, k_filters=16, dropout=0.05, batch_norm=True)
    model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()    
    
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-u-net-surfgrad', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    results = model.fit(X_train, y_train, batch_size=32, epochs=10, callbacks=callbacks,
                        validation_data=(X_test, y_test))
      
    return model
    
def main():
    X, Y = load_data(SUB_IMAGE_DIR_IMG, SUB_IMAGE_DIR_MASK) 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1975)
    model = train(X_train, X_test, y_train, y_test)
    model.evaluate(X_test, y_test, verbose = 1)
    # loss: 0.0086, accuracy: 0.99
    
    preds_train = model.predict(X_train, verbose = 1)
    preds_test = model.predict(X_test, verbose = 1)
    
    preds_train_thres = (preds_train > 0.5).astype(np.uint8)
    preds_test_thres = (preds_test > 0.5).astype(np.uint8)
    
    # normalize data
    trial_image = load_data(TRIAL_IMG_PATH)
    patches = get_patches(trial_image)
    preds_trial = model.predict(patches, verbose = 1)
    preds_trial_thres = (preds_trial > 0.5).astype(np.uint8)
    
    mask = build_mask(preds_trial_thres)
    imageio.imwrite('mask.png', mask)
    
if __name__ == '__main__':
    main()
