#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import cv2 as cv




# GLobal params
num_windows = 10
margin=110 
minpix=20
poly_points_1= np.array([[0,0], [0,720],[540,440],[740,440],[1280,720],[1280,0]])
poly_points_2= np.array([[0,652], [0,720],[1280,720],[1280,652]])
array_for_avg = []
array_for_avg_prediction = []

             
# Source points for homography
src = np.array([[605,440],[720,440], [1146, 651], [210, 690]], dtype="float32")

# Destination points for homography
dst = np.array([[0, 0], [250, 0], [250, 500], [0, 500]], dtype="float32")

#Homography 
H_matrix = cv.getPerspectiveTransform(src, dst)
Hinv = inv(H_matrix)

             
def rolling_avg(arr):
    global array_for_avg
    array_for_avg.append(arr)

    if len(array_for_avg) < 10:             
        array_for_avg.append(arr)
        return arr
    else:
        array_for_avg.pop(0)              
        array_for_avg.append(arr)
        return np.int32(np.mean(np.array(array_for_avg),axis=0))

def rolling_avg_prediction(arr):
    global array_for_avg_prediction           
    array_for_avg_prediction.append(arr)

    if len(array_for_avg_prediction) < 10:
        array_for_avg_prediction.append(arr)
        return arr
    else:
        array_for_avg_prediction.pop(0)
        array_for_avg_prediction.append(arr)
        return np.int32(np.mean(np.array(array_for_avg_prediction),axis=0))

def mask_poly(frame,poly_points):
    img=frame.copy()
    
    img = cv.fillPoly(img,[poly_points],(0,0,0))
    # cv.namedWindow('post_mask',cv.WINDOW_NORMAL)
    # cv.imshow('post_mask',img)
    
    return img 

def correct_dist(initial_img):
	# Intrnsic camera matrix
	k = [[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
		 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
	k = np.array(k)
	# Distortion Matrix
	dist = [[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]]
	dist = np.array(dist)
	img_2 = cv.undistort(initial_img, k, dist, None, k)

	return img_2

# Turn prediction for lanes
def turn_predict(image_center, right_lane_pos, left_lane_pos):
    lane_center = left_lane_pos + (right_lane_pos - left_lane_pos)/2
    if (lane_center - image_center < 0):
		
        return ("Right Turn")
    elif (lane_center - image_center < 8):
        return ("Straight Road")
    else:
    	return ("Left Turn")


# Process image and homography operations
def image_preprocessing(img):

	# crop_img = img[420:720, 40:1280, :]  # To get the region of interest
	frame_post_mask = mask_poly(img,poly_points_1)
	crop_img = mask_poly(frame_post_mask,poly_points_2)
	# undist_img = correct_dist(crop_img)

	hsl_img = cv.cvtColor(crop_img, cv.COLOR_BGR2HLS)
	# To seperate out Yellow colored lanes
	lower_mask_yellow = np.array([20, 120, 80], dtype='uint8')
	upper_mask_yellow = np.array([45, 200, 255], dtype='uint8')
	mask_yellow = cv.inRange(hsl_img, lower_mask_yellow, upper_mask_yellow)

	yellow_detect = cv.bitwise_and(hsl_img, hsl_img, mask=mask_yellow).astype(np.uint8)

	# To seperate out White colored lanes
	lower_mask_white = np.array([0, 200, 0], dtype='uint8')
	upper_mask_white = np.array([255, 255, 255], dtype='uint8')
	mask_white = cv.inRange(hsl_img, lower_mask_white, upper_mask_white)

	white_detect = cv.bitwise_and(hsl_img, hsl_img, mask=mask_white).astype(np.uint8)

	# Combine both
	lanes = cv.bitwise_or(yellow_detect, white_detect)
	new_lanes = cv.cvtColor(lanes, cv.COLOR_HLS2BGR)
	final = cv.cvtColor(new_lanes, cv.COLOR_BGR2GRAY)
	img_blur = cv.bilateralFilter(final, 9, 120, 100)

	canny_image = cv.Canny(img_blur, 100, 200)

	pts,prediction , lane_classfied , rads = calculate_curves(canny_image)
	rads[0] = rads[0]/5
	rads[1] = rads[1]/3
	color_blend = np.zeros_like(img).astype(np.uint8)
	cv.fillPoly(color_blend, pts, (0,255, 0))
	circles_image = draw_circles(color_blend)
	circles_image = cv.cvtColor(circles_image, cv.COLOR_GRAY2BGR)

	new_warp = cv.warpPerspective(color_blend, Hinv, (crop_img.shape[1], crop_img.shape[0]))
	result = cv.addWeighted(img, 1, new_warp, 0.5, 0)
	cv.putText(result, prediction, (50, 50),cv.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2, cv.LINE_AA)

	scale_percent = 20 # percent of original size
	width = int(circles_image.shape[1] * scale_percent / 100)
	height = int(circles_image.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized_circles_image = cv.resize(circles_image, dim, interpolation = cv.INTER_AREA)
	# cv.imshow("re-sized circle",resized_circles_image)

	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized_original_image = cv.resize(img, dim, interpolation = cv.INTER_AREA)

	# cv.imshow("re-sized original",resized_original_image)
	canvas = np.ones([880,1880,3]).astype(np.uint8)
	canvas[:720,:1280,:] = result[:,:,:]
	canvas[0:144,1280:1536] = resized_original_image[:,:,:]

	resized_lanes = cv.resize(lanes, dim, interpolation = cv.INTER_AREA)

	canvas[0:144,1537:1793] = resized_lanes[:,:,:]
	canvas[144:744,1280:1580] =  lane_classfied[:600,:300,:]
	canvas[144:744,1580:1880] =  circles_image[:600,:300,:]

	cv.putText(canvas, "(1)", (1300, 50),cv.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2, cv.LINE_AA)
	cv.putText(canvas, "(2)", (1570, 50),cv.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2, cv.LINE_AA)
	cv.putText(canvas, "(3)", (1300, 200),cv.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2, cv.LINE_AA)
	cv.putText(canvas, "(4)", (1570, 200),cv.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2, cv.LINE_AA)
	text_1 = '(1) : Original Image  (2) : Image processing'
	text_2 = '(3) : Warped and classified lanes (4) : Sliding windows curve detection'
	curvature = 'Left Curvature = '+str(rads[0]) + '  '+ 'Right Curvature = ' + str(rads[1])
	cv.putText(canvas,text_1, (50, 750),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
	cv.putText(canvas,text_2, (50, 800),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    
	cv.putText(canvas,curvature, (50, 850),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
	cv.imshow('canvas',canvas)
	return canvas           
def get_curve(img, leftx, rightx):

	#create points from 0 to 399 to plot the curve
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

	#max of y i.e 399
	y_eval = np.max(ploty)

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty, leftx, 2)
	right_fit_cr = np.polyfit(ploty, rightx, 2)

	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	#car position
	car_pos = img.shape[1]/2

	#calculating the lane center
	l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
	r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
	lane_center_position = (r_fit_x_int + l_fit_x_int) /2
	center = (car_pos - lane_center_position)  / 10
	return [left_curverad, right_curverad, center]


def draw_circles(frame):             
	img = frame.copy()
	final = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# Filter noise
	img_blur = cv.bilateralFilter(final, 9, 120, 100)

	# Apply edge detection
	img_edge = cv.Canny(img_blur, 100, 200)
	corners = cv.goodFeaturesToTrack(img_edge,100,0.001,50)
	corners = np.int0(corners)
	
	# Iterate over the corners and draw a circle at that location
	for i in corners:
		x,y = i.ravel()
		cv.circle(img_edge,(x,y),25,(255),1)
	return img_edge

def calculate_curves(img_edge):
	new_img = cv.warpPerspective(img_edge, H_matrix, (300, 600))
	histogram = np.sum(new_img, axis=0)
	out_img = np.dstack((new_img,new_img,new_img))*255

	midpoint = np.int(histogram.shape[0]/2)
					
	# Compute the left and right max pixels
	leftx_ = np.argmax(histogram[:midpoint])
	rightx_ = np.argmax(histogram[midpoint:]) + midpoint

	left_lane_pos = leftx_
	right_lane_pos = rightx_
	image_center = int(new_img.shape[1]/2)

	prediction = turn_predict(image_center, right_lane_pos, left_lane_pos)	
			
	window_height = np.int(new_img.shape[0]/num_windows)
	nonzero = new_img.nonzero()             
	nonzeroy = np.array(nonzero[0])            
	nonzerox = np.array(nonzero[1])

	leftx_p = leftx_
	rightx_p = rightx_

	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(num_windows):
		# Identify window boundaries in x and y (and right and left)
		win_y_down = new_img.shape[0] - (window+1)*window_height
		win_y_up = new_img.shape[0] - window*window_height
		win_x_left_down = leftx_p - margin
		win_x_left_up = leftx_p + margin
		win_x_right_down = rightx_p - margin
		win_x_right_up = rightx_p + margin
		window_pts = np.array([[win_x_left_up,win_y_down],
								[win_x_right_up,win_y_down],
								[win_x_left_down,win_y_up],
								[win_x_right_down,win_y_up]])
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_left_down) & (nonzerox < win_x_left_up)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_right_down) & (nonzerox < win_x_right_up)).nonzero()[0]
		
		# Append these indices to the list
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		
		# If found > minpix pixels, move to next window
		if len(good_left_inds) > minpix:
			leftx_p = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_p = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 


	if leftx.size == 0 or rightx.size == 0 or lefty.size == 0 or righty.size == 0:
		return

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	ploty = np.linspace(0, new_img.shape[0]-1, new_img.shape[0] )

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Fit a second order polynomial to each
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Extracting points from fit
	left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
	image_center = img_edge.shape[0]/2

	pts = np.hstack((left_line_pts, right_line_pts))
	pts = np.array(pts, dtype=np.int32)
	rads = get_curve(new_img,left_fitx,right_fitx)

	return pts , prediction , out_img , rads


def main():
	cap = cv.VideoCapture('media/problem_3.mp4')
	fourcc = cv.VideoWriter_fourcc('m','p','4', 'v')
	out = cv.VideoWriter('output/problem_3_output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (1880,880))
	while cap.isOpened():
		success, frame = cap.read()
		if success is False:
			break
		initial_image = frame
		final = image_preprocessing(initial_image)
		out.write(final)
		cv.waitKey(1)

	cap.release()

if __name__ == '__main__':
	main()            