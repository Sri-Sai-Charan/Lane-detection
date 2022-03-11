#usr/bin/python3

import numpy as np
import cv2 as cv

def find_lanes(frame,frame_post_mask):
    img= frame_post_mask.copy()
    ret, thresh = cv.threshold(img, 127, 255, 0)
    img_canny = cv.Canny(thresh,100,200,2)
    
    contours, hierarchy  = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for i in contours:
        if cv.arcLength(i,False)>1000:
            cv.drawContours(frame, [i], -1, (0,255,0), 3)
            # cv.namedWindow('contours',cv.WINDOW_NORMAL)
            # cv.imshow('contours',img)
            # print(cv.arcLength(i,False))

        if 1000>cv.arcLength(i,False)>100:
            cv.drawContours(frame, [i], -1, (0,0,255), 3)
        # cv.namedWindow('contours',cv.WINDOW_NORMAL)
        # cv.imshow('contours',img)
            # print(cv.arcLength(i,False))
    return frame


def mask_poly(frame):
    img=frame.copy()
    poly_points= np.array([[0,0], [0,539],[400,330],[550,330],[959,539],[959,0]])
    img = cv.fillPoly(img,[poly_points],(0,0,0))
    # cv.namedWindow('post_mask',cv.WINDOW_NORMAL)
    # cv.imshow('post_mask',img)
    
    return img


    
def main():
    vid = cv.VideoCapture('media/problem_2.mp4')
    count =0
    while(vid.isOpened()):
        ret,frame = vid.read()
        if ret == False:
            break  

        frame_post_mask = mask_poly(frame)
        frame = find_lanes(frame,frame_post_mask)

        print("frame :",count)
        count +=1
        cv.namedWindow('frame',cv.WINDOW_NORMAL)
        cv.imshow('frame',frame)
        # cv.waitKey(1)
        cv.waitKey(0)
        # break
    vid.release()
if __name__ == '__main__':
    main()               