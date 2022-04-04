#!/usr/bin/python3

import numpy as np
import cv2 as cv

def main():
    for count in range(25):
        img = cv.imread('media/problem_1/'+str(count)+'.png')
        cv.imshow('Frame',img)
        cv.waitKey(0)
        
if __name__ == '__main__':
    main()