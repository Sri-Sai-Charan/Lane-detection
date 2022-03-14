#!/usr/bin/python3

from my_function import *

def main():
    vid = cv.VideoCapture('media/problem_3.mp4')
    count =0
    poly_points_1= np.array([[0,0], [0,720],[540,440],[740,440],[1280,720],[1280,0]])
    poly_points_2= np.array([[0,652], [0,720],[1280,720],[1280,652]])

    while(vid.isOpened()):
        ret,frame = vid.read()
        if ret == False:
            break  
   

        
        frame_post_mask = mask_poly(frame,poly_points_1)
        frame_post_mask = mask_poly(frame_post_mask,poly_points_2)
        frame = find_lanes(frame,frame_post_mask)



        count +=1
        cv.putText(frame, str(count), (0,100), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv.LINE_AA)
        cv.namedWindow('frame',cv.WINDOW_NORMAL)
        cv.imshow('frame',frame)
        # cv.waitKey(1)
        cv.waitKey(0)
        # break
    vid.release()


if __name__ == '__main__':
    main()