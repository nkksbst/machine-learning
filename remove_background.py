# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:54:47 2019
@author: Monikka Busto
--------------------------------------------------
isolates the unit from the background
1. binarize image using adaptive thresholding
2. find the bounding rectangle using binary image
3. warp the detected unit to portrait mode
--------------------------------------------------
"""
import numpy as np
import cv2
import imageio
import glob
from PIL import Image
from config import *

def load_image(file_path):
    im = Image.open(file_path)
    np_im = np.array(im)
    return np_im

def save_image(img, file_path, filetype):
    img_format = Image.fromarray(img)
    img_format.save(file_path + filetype)

def draw_contour(orig, img_edge):
    contours, _ = cv2.findContours(image = img_edge, 
                                          mode = cv2.RETR_TREE, 
                                          method = cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    rect = cv2.minAreaRect(contours[0]) 
    box = cv2.boxPoints(rect)
    box = box.astype('int')
    img_box = cv2.drawContours(orig, contours = [box], 
                                 contourIdx = -1, 
                                 color = (255, 0, 0), thickness = 10)
    return img_box

def remove_background(img, img_edge):
    contours, _ = cv2.findContours(image = img_edge, 
                                          mode = cv2.RETR_TREE, 
                                          method = cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    rect = cv2.minAreaRect(contours[0])
    # bounding rectangle params
    width = int(rect[1][0])
    height = int(rect[1][1])
    box = cv2.boxPoints(rect)   # BL, TL, TR, BR
    box = box.astype('float32') 
    # detected_corners 
    new_corners = np.array([
            [0, height - 1],
            [0,0],
            [width-1, 0],
            [width-1, height-1]], dtype = 'float32')
    # warp the boxed area to portrait
    M = cv2.getPerspectiveTransform(box, new_corners)
    img_cropped = cv2.warpPerspective(img, M, (width, height))
    # rotate if warped image is landscape    
    if(width > height):
        img_cropped = cv2.rotate(img_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    return img_cropped

def get_edges(image):
    gray =cv2.blur(image,(50,50))
    edged = cv2.Canny(gray, 0, 10)

    kernel = np.ones((50,50),np.uint8) 
    dilated = cv2.dilate(edged,kernel,iterations = 1)
   
    kernel = np.ones((200,1), np.uint8)  # vertical kernel
    img_edges = cv2.dilate(dilated, kernel, iterations=1)

    return img_edges

def gamma_correct(image, gamma=1.0):
	invGamma = 1.0 / gamma
	gamma_lookup = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype('uint8')
	return cv2.LUT(image, gamma_lookup)

def main():
    for path in glob.glob(ORIG_DIR + '/*/*.bmp'):
        img = load_image(path)
        mask_path = path.replace(ORIG_DIR, MASK_DIR)
        mask_path= mask_path.replace('bmp', 'png')
        mask = load_image(mask_path)
        mask = np.invert(mask)
        cropped = remove_background(img, mask)
        imageio.imwrite(path.replace(ORIG_DIR, CRP_DIR), cropped)        
if __name__ == '__main__':
    main()                  