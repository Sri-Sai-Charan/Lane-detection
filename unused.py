#!/usr/bin/python3

import cv2 as cv
import numpy as np

def find_orientation(frame):
    img = frame.copy()
    img_canny = cv.Canny(img,255,255,0)
    number_of_white = np.argwhere(img_canny[:,:]==255)
    center_pts = (np.mean(number_of_white[0]),np.mean(number_of_white[1]))
    print(center_pts)
    cv.imshow('canny',img_canny)
    mid = int(img.shape[0]/2)
    top_half = img_canny[:mid,:]
    bottom_half = img_canny[mid:,:]
    cv.imshow('top half',top_half)
    cv.imshow('bottom half',bottom_half)
    return frame

def callback(x):
    print(x)



def dynamic_canny(frame):
    img = frame.copy()
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    canny = cv.Canny(frame, 85, 255,0) 

    cv.namedWindow('image') # make a window with name 'image'
    cv.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
    cv.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

    while(1):
        # numpy_horizontal_concat = np.concatenate((frame, canny), axis=1) # to display image side by side
        # cv.imshow('image', numpy_horizontal_concat)
        k = cv.waitKey(1) & 0xFF
        if k == 27: #escape key
            break
        l = cv.getTrackbarPos('L', 'image')
        u = cv.getTrackbarPos('U', 'image')

        canny = cv.Canny(frame, l, u)
        cv.imshow('canny',canny)
        print(l,u)
    # cv.destroyAllWindows()
