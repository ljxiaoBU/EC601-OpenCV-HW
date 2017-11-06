
import cv2

gray = cv2.imread('Test_images/baboon.jpg',0)
cv2.imwrite('ex4output/original.jpg',gray)

value = 128
retval,threshold = cv2.threshold(gray, value, 255,2)
cv2.imshow('threshold',threshold)
cv2.imwrite('ex4output/threshold.jpg',threshold)
retval, thresh1 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('binary',thresh1)
cv2.imwrite('ex4output/binary.jpg',thresh1)


max_val = 255
retval,thresh2 = cv2.threshold(gray,125,max_val,cv2.THRESH_BINARY_INV)
retval, thresh3 = cv2.threshold(gray, 27, max_val, cv2.THRESH_BINARY)

band = cv2.bitwise_and(thresh2,thresh3)
cv2.imshow('band',band)
cv2.imwrite('ex4output/band.jpg',band)

retval,thresh4 = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
semi = cv2.bitwise_and(gray,thresh4)
cv2.imshow('semi',semi)
cv2.imwrite('ex4output/semi.jpg',semi)

thresh5 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,10)
cv2.imshow('adaptive', thresh5)
cv2.imwrite('ex4output/adaptive.jpg',thresh5)

cv2.waitKey(0)