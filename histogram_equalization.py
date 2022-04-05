#!/usr/bin/python3

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



# Class containing functions to perform equalization
class histogram_equalization:

    # Function to generate histogram of input image 
    def generate_histogram(self, intensity):
        histogram = np.zeros([256], dtype='uint32')
        cols = intensity.shape[0]
        rows = intensity.shape[1]
        for i in range(cols):
            for j in range(rows):
                histogram[intensity[i, j]] += 1
        return histogram

    # Function to generate cumulative distribution function graph from histogram
    def generate_cdf(self, intensity):
        cdf = np.zeros([256])
        histogram = self.generate_histogram(intensity)
        temp = histogram/np.sum(histogram)
        cdf[0] = temp[0]
        var = temp[0]
        for i in range(1, len(temp)):
            var += temp[i]
            cdf[i] = var
        return cdf

    # Function to perform histogram equalization of CDF
    def equalization(self, intensity):
        equalized_stream = np.copy(intensity)
        cdf = self.generate_cdf(intensity)
        cols = intensity.shape[0]
        rows = intensity.shape[1]
        for i in range(cols):
            for j in range(rows):
                equalized_stream[i, j] = int(cdf[intensity[i, j]]*255)
        return equalized_stream


    # Function to perform adaptive histogram equalization 
    def adaptive_equalization(self,intensity,tile_size):
        
        # Blank canvas to create final image
        blank=np.zeros(intensity.shape,dtype='uint8')
        
        i=0
        j=0
        

        for j in range (0,intensity.shape[1],tile_size):
            i=0
            for i in range(0,intensity.shape[0],tile_size):
                # Create tiles from image
                tile= intensity[i:i+tile_size,j:j+tile_size]
                # Perform equalization on each tile
                equalized_tile= self.equalization(tile)
                # Append tile to blank canvas
                blank[i:i+tile_size,j:j+tile_size]=equalized_tile
        
        return blank
        
def main():
    obj = histogram_equalization()

    # Codec to create videowriter object
    fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    # Create video for original images
    # out1 = cv.VideoWriter('original_video.avi', fourcc, 16, (1224, 370))
    # Create video to record histogram equalized images
    out2 = cv.VideoWriter('output/Histogram.avi', fourcc, 16, (1224, 370))
    # Create video to record adaptive histogram equalized images
    out3 = cv.VideoWriter('output/Adaptive_histogram.avi', fourcc, 16, (1224, 370))

    for count in range(25):
        image = cv.imread('media/problem_1/'+str(count)+'.png')
        # image = cv.imread(frame)
        image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
        # Split HSV channels
        h = image_hsv[:, :, 0]
        s = image_hsv[:, :, 1]
        v = image_hsv[:, :, 2]
        

        # out1.write(image)
        
        # Ordinary histogram equalization
        histogram_eq = np.dstack((h, s, obj.equalization(v)))
        histogram_eq = cv.cvtColor(histogram_eq, cv.COLOR_HSV2BGR)
        out2.write(histogram_eq)
        
        # Adaptive histogram equalization
        adaptive_eq=np.dstack((h,s,obj.adaptive_equalization(v,50)))
        adaptive_eq = cv.cvtColor(adaptive_eq, cv.COLOR_HSV2BGR)
        out3.write(adaptive_eq)
        
        # Uncomment to view each frame if required
        cv.imshow('original',image)
        cv.imshow('Only hist eq',histogram_eq)
        cv.imshow('Adaptive eq',adaptive_eq)
        cv.waitKey(1)
        # Can be uncommented if histogram plots need to be viewed
        # fig,(ax1,ax2)=plt.subplots(2)
        # fig.suptitle('Histogram and CDF for Original Image')
        # ax1.plot(range(256),obj.generate_histogram(v))
        # ax1.set_title('Histogram')
        # ax2.plot(range(256),obj.generate_cdf(v))
        # ax2.set_title('CDF')
        # fig.tight_layout()
        # plt.show()

        # fig,(ax1,ax2)=plt.subplots(2)
        # fig.suptitle('Histogram and CDF for Histogram Equalized Image')
        # ax1.plot(range(256),obj.generate_histogram(obj.equalization(v)))
        # ax1.set_title('Histogram')
        # ax2.plot(range(256),obj.generate_cdf(obj.equalization(v)))
        # ax2.set_title('CDF')
        # fig.tight_layout()
        # plt.show()

        # fig, (ax1, ax2) = plt.subplots(2)
        # fig.suptitle('Histogram and CDF for Adaptive Histogram Equalized Image')
        # ax1.plot(range(256), obj.generate_histogram(obj.adaptive_equalization(v, 50)))
        # ax1.set_title('Histogram, tile_size = 50')
        # ax2.plot(range(256), obj.generate_cdf(obj.adaptive_equalization(v, 50)))
        # ax2.set_title('CDF, tile_size = 50')
        # fig.tight_layout()
        # plt.show()
if __name__ == '__main__':
    main()






# Gather all images under one iteratable variable
# files = sorted(glob.glob("adaptive_hist_data" + "/*.png"))
# # print(files)

# # Define object of class histogram equalization
# obj = histogram_equalization()


# for frame in files:
#     image = cv.imread(frame)
#     image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
#     # Split HSV channels
#     h = image_hsv[:, :, 0]
#     s = image_hsv[:, :, 1]
#     v = image_hsv[:, :, 2]
    

#     out1.write(image)
    
#     # Ordinary histogram equalization
#     histogram_eq = np.dstack((h, s, obj.equalization(v)))
#     histogram_eq = cv.cvtColor(histogram_eq, cv.COLOR_HSV2BGR)
#     out2.write(histogram_eq)
    
#     # Adaptive histogram equalization
#     adaptive_eq=np.dstack((h,s,obj.adaptive_equalization(v,50)))
#     adaptive_eq = cv.cvtColor(adaptive_eq, cv.COLOR_HSV2BGR)
#     out3.write(adaptive_eq)
    
#     # Uncomment to view each frame if required
#     cv.imshow('original',image)
#     cv.imshow('Only hist eq',histogram_eq)
#     cv.imshow('Adaptive eq',adaptive_eq)
#     cv.waitKey(1)



