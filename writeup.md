**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify undistorted image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/output_test1.jpg "Undistorted"
[image2]: ./output_images/undistorted.jpg "Road Transformed"
[image3]: ./output_images/binary.jpg "Binary Example"
[image4]: ./output_images/warped.jpg "Warp Example"
[image5]: ./output_images/detection.jpg "Fit Visual"
[image6]: ./output_images/lane.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #10 through #46 of the file called `detect_lanes.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

To demonstrate how I approached these steps, the first frame of video named `project_video.mp4` will be used below.

#### 1. Provide an example of a distortion-corrected image.

Using the camera calibration and distortion coefficients computed above, I simply applied `cv2.undistort()` to obstain this result:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #48 - #117 and #202 - #206 in `detect_lanes.py`). For gradient thresholds, I used two methods: absolute value on Sobel x direction; magnitude and direction of Sobel output. For color thresholds, I used two methods: L channel of HLS color space (targeting at white lane markings); B channel of Lab color space (targeting at yellow lane markings).

Here's an example of my output for this step. Note that I applied perspective transform before this step, because lanes are mostly upright after transformation and easier to be detected.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in lines #191 through #199 in the file `detect_lanes.py` I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(238, 684), (1060, 684), (588, 451), (688, 451)])
dst = np.float32([(320, 720), (960, 720), (320, 0), (960, 0)])
```

After perspective transform I obtained the following result:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

My approach is based on the `convolution` approach given in the course. At each frame, the image is divided into 9 levels vertically, and centroids are computed from bottom to top. At each level, centroids are searched near those of the last level, and those of the last frame (in the same level), thus maintaining spatial and temporal continuity. Details can be found on lines #208 - #250 of `detect_lanes.py`. Then I collected all inlier points and fit a 2nd order polynomial. Example of lane detection is given below:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #252 through #280 in my code in `detect_lanes.py`. Curvature is computed according to the bottom of the image, which is then converted to radius in meters. Offset of vehicle is computed by the substracting average of left/right lane markings from image center (car position).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #282 through #328 in my code in `detect_lanes.py`. Basically I overlayed left lane markings, right lane markings and in-between spaces to the transformed image, and apply inversed perspective transformation. Then I add texts of curvature (average of left and right) and offset. Here is an example of my result:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult part was tuning the parameters of gradient and color thresholds. On some frames (shadows, occlusions, color change, etc.), enforcing spatial continuity is not sufficient, thus I added temporal continuity (comparing with centroids of last frame). If sharp features occur near the real lane markings, the algorithm will likely to detect those as lane markings. A better set of thresholds (or better approach) to discriminate lane markings from distractions may be needed to make the algorithm more robust.