# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:12:44 2017

@author: xxh
"""

import cv2

lena = cv2.imread('Test_images/Lenna.png')

[b,g,r] = cv2.split(lena)

cv2.imshow('original',lena)
cv2.imwrite('outputimages/lena.png', lena)
print('r[20,25] = ', r[20,25], 'g[20,25] = ', g[20,25],'b[20,25] = ', b[20,25])

cv2.imshow('blue',b)
cv2.imwrite('outputimages/blue.png',b)
cv2.imshow('red',r) 
cv2.imwrite('outputimages/red.png',r)
cv2.imshow('green',g)
cv2.imwrite('outputimages/green.png',g)

hsv_lena = cv2.cvtColor(lena, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv_lena', hsv_lena)
cv2.imwrite('outputimages/hsv_lena.png', hsv_lena)
[h,s,v] = cv2.split(hsv_lena)
print('h[20,25] = ', h[20,25], 's[20,25] = ', s[20,25], 'v[20,25] = ', v[20,25])

cv2.imshow('hue',h)
cv2.imwrite('outputimages/hue.png',h)
cv2.imshow('saturation',s)
cv2.imwrite('outputimages/saturation.png',s)
cv2.imshow('value',v)
cv2.imwrite('outputimages/value.png',v)

ycbcr_lena = cv2.cvtColor(lena, cv2.COLOR_BGR2YCR_CB)
cv2.imshow('ycbcr_lena', ycbcr_lena)
cv2.imwrite('outputimages/ycbcr_lena.png', ycbcr_lena)
[y,cb,cr] = cv2.split(ycbcr_lena)
print('y[20,25] = ', y[20,25],'cb[20,25] = ', cb[20,25],'cr[20,25] = ', cr[20,25])

cv2.imshow('y',y)
cv2.imwrite('outputimages/y.png',y)
cv2.imshow('outputimages/cb',cb)
cv2.imwrite('outputimages/cb.png',cb)
cv2.imshow('outputimages/cr',cr)
cv2.imwrite('outputimages/cr.png',cr)

cv2.waitKey(0)
