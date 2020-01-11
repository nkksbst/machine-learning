# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 08:46:32 2020

@author: Monikka Busto
"""
%matplotlib 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
from PIL import Image

def load_image(file_path):
    im = Image.open(file_path)
    np_im = np.array(im)
    return np_im


x = np.linspace(0, 10, 100)

# functions
fig = plt.figure(figsize=(6, 4))
plt.plot(x, np.sin(x))
plt.xlabel("This is an x label")
plt.ylabel("This is a y label")
plt.savefig('fig_filename.png')

# images
plt.imshow(img, cmap = 'gray')

# subplots
paths = glob.glob('image_directory/*')
for img_idx, path in enumerate(paths):
    img = load_image(path)
    subplot_idx = img_idx + 1
    plt.subplot(2, 5, subplot_idx) # 10 images
    plt.imshow(img, cmap = 'gray')    
