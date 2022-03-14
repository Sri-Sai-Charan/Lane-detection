#usr/bin/python3


from my_function import *




    
def main():
    vid = cv.VideoCapture('media/problem_2.mp4')
    count =0
    poly_points= np.array([[0,0], [0,539],[400,330],[550,330],[959,539],[959,0]])
    limits = [1000,50]
    while(vid.isOpened()):
        ret,frame = vid.read()
        if ret == False:
            break  
    
        frame_post_mask = mask_poly(frame,poly_points)
        frame = find_lanes(frame,frame_post_mask,limits)

        count +=1
        cv.namedWindow('frame',cv.WINDOW_NORMAL)
        cv.imshow('frame',frame)
        
        # cv.waitKey(1)
        cv.waitKey(0)
        # break
    vid.release()
if __name__ == '__main__':
    main()               