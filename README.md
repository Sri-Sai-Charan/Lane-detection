# Lane-detection

This program detects and classifies lanes and Determines which direction the car has to turn


## Overview

Approach:
Masking, HSL color filtering and Canny we first mask the given image to filter out
noise as similar to problem 2, we then further filter the image using cvtColor to convert the
image from BGR to HSL which allows us to identify the yellow and white lanes for better lane
detection.

Warping image we warp to a birds eye view using getPerspectiveTransform (to find the
homography matrix) we do this to allow detection of direction to be easier as we can determine
direction based on the max and min points present in the warped image.

Sliding window as we now have a warped image of the lanes and the edge points we
can start constructing the curve which lets us calculate the direction of motion and the radius of
curvature. By checking for adjacent area close to the edge points detected we now map the
curve far better than other conventional image recognition methods.

We then can construct a polygon using the extracted points which is later used for
superimposing the original image with the detected path.

Finally we can combine all previous mentioned images to understand the working of this
pipeline better.

## Results

![](https://github.com/Sri-Sai-Charan/Lane-detection/blob/main/output/problem_3_output.gif)

## Folder Structure
```
📦Lane-detection
 ┣ 📂media
 ┃ ┣ 📜problem_2.mp4
 ┃ ┗ 📜problem_3.mp4
 ┣ 📂output
 ┃ ┣ 📜problem_2_output.avi
 ┃ ┣ 📜problem_3_output.avi
 ┃ ┗ 📜problem_3_output.gif
 ┣ 📜README.md
 ┣ 📜histogram_equalization.py
 ┣ 📜straight_lines.py
 ┗ 📜turn_prediction.py
 ```
 

