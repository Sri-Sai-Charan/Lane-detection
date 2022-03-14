#!/usr/bin/python3


import numpy as np
import cv2 as cv


def find_lanes(frame,frame_post_mask,limits):
    img= frame_post_mask.copy()
    ret, thresh = cv.threshold(img, 127, 255, 0)
    img_canny = cv.Canny(thresh,100,200,2)
    
    _,contours, hierarchy  = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(limits[1])
    for i in contours:
        if cv.arcLength(i,False)>limits[1]:
            cv.drawContours(frame, [i], -1, (0,255,0), 3)
            # cv.namedWindow('contours',cv.WINDOW_NORMAL)
            # cv.imshow('contours',img)
            # print(cv.arcLength(i,False))

        if limits[0]> cv.arcLength(i,False) > limits[1]:
            cv.drawContours(frame, [i], -1, (0,0,255), 3)
        # cv.namedWindow('contours',cv.WINDOW_NORMAL)
        # cv.imshow('contours',img)
            print(cv.arcLength(i,False))
    return frame


def mask_poly(frame,poly_points):
    img=frame.copy()
    
    img = cv.fillPoly(img,[poly_points],(0,0,0))
    # cv.namedWindow('post_mask',cv.WINDOW_NORMAL)
    # cv.imshow('post_mask',img)
    
    return img