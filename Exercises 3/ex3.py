# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:48:24 2017

@author: xxh
"""

import cv2
import numpy as np

def Add_gaussian_Noise(image, pa, pb):
    row, col = image.shape
    mean = pa
    sigma = pb
    gaussian = np.random.normal(mean, sigma, (row,col))
    gaussian = gaussian.reshape(row, col)
    noise = image + gaussian
    return noise


def Add_saltandpepper_Noise(image, pa, pb):
    row, col = image.shape
    out = np.copy(image)
    salt = np.ceil(row*col*pb)
    coords = [np.random.randint(0, i-1, int(salt)) for i in image.shape]
    out[coords] = 255
    
    pepper = np.ceil(row*col*pa)
    coords = [np.random.randint(0, i-1, int(pepper)) for i in image.shape]
    out[coords] = 0
    return out
    

image = cv2.imread("Test_images/Lenna.png", 0)
cv2.imshow("original image", image)
cv2.imwrite('ex3output/originalimage.png', image)
    
noise_img = np.copy(image)
pa = 0
pb = 50
noise_img = Add_gaussian_Noise(noise_img, pa, pb)
cv2.imshow("Gaussian Noise", noise_img)
cv2.imwrite("ex3output/gaussiannoise.png", noise_img)

noise_dst = np.copy(noise_img)
noise_dst = cv2.blur(noise_dst, (3,3))
cv2.imshow("Box filter", noise_dst)
cv2.imwrite("ex3output/boxfilter.png", noise_dst)
    
noise_dst1 = np.copy(noise_img)
noise_dst1 = cv2.GaussianBlur(noise_dst1,(3,3),1.5)
cv2.imshow("gaussian filter", noise_dst1)
cv2.imwrite("ex3output/faussian.png", noise_dst1)
    
noise_dst2 = np.copy(noise_img)
noise_dst2 = noise_dst2.astype(np.uint8)
noise_dst2 = cv2.medianBlur(noise_dst2, 3)
cv2.imshow("median", noise_dst2)
cv2.imwrite('ex3output/median.png', noise_dst2)
    
pa = 0.01
pb = 0.01
noise_img2 = Add_saltandpepper_Noise(image, pa, pb)
cv2.imshow("saltandpepper", noise_img2)
cv2.imwrite("ex3output/saltpepper.png", noise_img2)
    
noise_dst3 = np.copy(noise_img2)
noise_dst3 = cv2.blur(noise_dst3,(3,3))
cv2.imshow("box filter", noise_dst3)
cv2.imwrite('ex3output/boxfilter_saltpepper.png', noise_dst3)
    
noise_dst4 = np.copy(noise_img2)
noise_dst4 = cv2.GaussianBlur(noise_dst4, (3,3), 1.5)
cv2.imshow('gaussian filter', noise_dst4)
cv2.imwrite('ex3output/gaussianfilter_saltpepper.png', noise_dst4)
    
noise_dst5 = np.copy(noise_img2)
noise_dst5 = cv2.medianBlur(noise_dst5, 3)
cv2.imshow('median filter', noise_dst5)
    
    
    
cv2.waitKey(0)
    
    
    

    
        