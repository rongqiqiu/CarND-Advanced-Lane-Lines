import numpy as np
import cv2
import glob
import imageio
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

# Step 1: calibrate camera using images under ./camera_cal/

images = glob.glob('./camera_cal/calibration*.jpg')

nx = 9
ny = 6 # Dimension of the chessboard

objpoints = []
imgpoints = []
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

for filename in images:
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # Draw chessboard corners and output
        img = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
        # cv2.imwrite('./output_images/chessboard_calibration' + filename[24:], img)

# Calibrate the camera using detected chessboard corners
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Undistort calibration images and output
for filename in images:
    image = cv2.imread(filename)
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    # cv2.imwrite('./output_images/output_calibration' + filename[24:], undistorted)

# Undistort test images and output
test_images = glob.glob('./test_images/*.jpg')
for filename in test_images:
    image = cv2.imread(filename)
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    # cv2.imwrite('./output_images/output_' + filename[14:], undistorted)

# Threshold on absolute value of Sobel output in either 'x' or 'y' direction
def abs_sobel_thresh(img, orient = 'x', sobel_kernel = 3, thresh = (0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    absolute = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * absolute / np.max(absolute))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

# Threshold on the magnitude of Sobel output
def mag_thresh(img, sobel_kernel = 3, thresh = (0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    mag = np.sqrt(sobelx * sobelx + sobely * sobely)
    scaled = np.uint8(255 * mag / np.max(mag))
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary_output

# Threshold on the direction of Sobel output
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output

# Threshold on gradient of the image, which is a combination of absolute value, magnitude and direction of Sobel output
def gradient_thresh(img, sobel_kernel = 3):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=sobel_kernel, thresh=(20, 255))
    mag_binary = mag_thresh(img, sobel_kernel=sobel_kernel, thresh=(20, 255))
    dir_binary = dir_thresh(img, sobel_kernel=sobel_kernel, thresh=(0.0, 0.2))
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

# Threshold on L channel of HLS color space, targeting at white lane markings
def white_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]

    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel >= thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

# Threshold on B channel of Lab color space, targeting at yellow lane markings
def yellow_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = hls[:,:,2]

    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel >= thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output

# Threshold on color, which is a combination of HLS and Lab color spaces
def color_thresh(img, thresh=(0, 255)):
    white_binary = white_thresh(img, thresh=(200, 255))
    yellow_binary = yellow_thresh(img, thresh=(150, 255))
    combined = np.zeros_like(white_binary)
    combined[(white_binary == 1) | (yellow_binary == 1)] = 1
    return combined

# Create a binary mask containing a window of designated size
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

# Find window centroids of the current frame given information from last frame
# margin is the maximum difference of centroids of current and last level
# last_margin is the maximum difference of centroids of current and last frame
def find_window_centroids(image, window_width, window_height, margin, last_margin, last_l_centroids, last_r_centroids):
    l_centroids = []
    r_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template for convolutions
    
    # Default positions to search for starting positions of the left and right lanes
    l_center = image.shape[1]/4
    r_center = 3*image.shape[1]/4

    # Go through each layer looking for max pixel locations
    for level in range(0,(int)(image.shape[0]/window_height)):
        # If it is the first frame, use centroids of the last level as the centers of search windows
        # Otherwise, use centroids of the last frame as the centers of search windows
        if last_l_centroids:
            last_l_center = last_l_centroids[level]
            last_r_center = last_r_centroids[level]
        else:
            last_l_center = l_center
            last_r_center = r_center
        # Get the vertical image slice of current level
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        if (level == 0):
            # For the first level, uum quarter bottom of image to get slice
            image_layer = np.sum(image[int(3*image.shape[0]/4):,:], axis=0)
        # convolve the window into the vertical slice of the image
        conv_signal = np.convolve(window, image_layer)
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        # Find the best centroid of current level for the left lane
        l_min_index = int(max(l_center+offset-margin,last_l_center+offset-last_margin,0))
        l_max_index = int(min(l_center+offset+margin,last_l_center+offset+last_margin,image.shape[1]))
        if l_max_index > l_min_index and np.max(conv_signal[l_min_index:l_max_index]) > 50:
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        else:
            # Use centroid of last frame by default
            l_center = last_l_center
        # Find the best centroid of current level for the right lane
        r_min_index = int(max(r_center+offset-margin,last_r_center+offset-last_margin,0))
        r_max_index = int(min(r_center+offset+margin,last_r_center+offset+last_margin,image.shape[1]))
        if r_max_index > r_min_index and np.max(conv_signal[r_min_index:r_max_index]) > 50:
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        else:
            # Use centroid of last frame by default
            r_center = last_r_center
        # Add what we found for the current layer
        l_centroids.append(l_center)
        r_centroids.append(r_center)

    return l_centroids, r_centroids

def process_image(image):
    mtx = process_image.mtx
    dist = process_image.dist
    last_l_centroids = process_image.last_l_centroids
    last_r_centroids = process_image.last_r_centroids
    # Because OpenCV assumes channels as B, G, R, we need to reverse the order of the channels
    # cv2.imwrite('./output_images/raw.jpg', image[:, :, ::-1])

    # Step 2: apply distortion correction to raw images
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    # cv2.imwrite('./output_images/undistorted.jpg', undistorted[:, :, ::-1])

    # Step 3: apply a perspective transform to rectify undistorted image
    src = np.float32([(238, 684), (1060, 684), (588, 451), (688, 451)])
    dst = np.float32([(320, 720), (960, 720), (320, 0), (960, 0)])

    img_size = (image.shape[1], image.shape[0])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undistorted, M, img_size)
    # cv2.imwrite('./output_images/warped.jpg', warped[:, :, ::-1])

    # Step 4: use color transforms, gradients, etc., to create a thresholded binary image
    gradient_binary = gradient_thresh(warped)
    color_binary = color_thresh(warped)
    combined = np.zeros_like(gradient_binary)
    combined[(gradient_binary == 1) | (color_binary == 1)] = 1
    # cv2.imwrite('./output_images/binary.jpg', np.stack((combined, combined, combined), axis = -1) * 255)

    # Step 5: detect lane pixels and fit to find the lane boundary
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching based on last level
    last_margin = 50 # How much to slide left and right for searching based on last frame

    l_centroids, r_centroids = find_window_centroids(combined, window_width, window_height, margin, last_margin, last_l_centroids, last_r_centroids)
    process_image.last_l_centroids = l_centroids
    process_image.last_r_centroids = r_centroids

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(combined)
    r_points = np.zeros_like(combined)

    # Go through each level and draw the windows
    for level in range(0,len(l_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,combined,l_centroids[level],level)
        r_mask = window_mask(window_width,window_height,combined,r_centroids[level],level)
        # Add graphic points from window mask here to total pixels found 
        l_points[l_mask == 1] = 1
        r_points[r_mask == 1] = 1

    left_points = np.zeros_like(combined)
    left_points[(l_points == 1) & (combined == 1)] = 1
    lefty, leftx = np.nonzero(left_points)

    right_points = np.zeros_like(combined)
    right_points[(r_points == 1) & (combined == 1)] = 1
    righty, rightx = np.nonzero(right_points)

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) * 255 # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((combined,combined,combined)),np.uint8) * 255 # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1.0, template, 0.5, 0.0) # overlay the orignal road image with window results
    # cv2.imwrite('./output_images/detection.jpg', output[:, :, ::-1])

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Step 6: determine the curvature of the lane and vehicle position with respect to center
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    # Now our radius of curvature is in meters
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    left_pos = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_pos = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    center_pos = (left_pos + right_pos) / 2.0
    car_pos = 640
    car_offset = (car_pos - center_pos) * xm_per_pix

    # Step 7: warp the detected lane boundaries back onto the original image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(combined).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx - 10.0, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([left_fitx + 10.0, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Step 8: output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([right_fitx - 10.0, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx + 10.0, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1.0, newwarp, 0.3, 0)
    # cv2.imwrite('./output_images/lane.jpg', result[:, :, ::-1])

    font = cv2.FONT_HERSHEY_SIMPLEX
    line1 = 'Radius of Curvature = ' + str(int(left_curverad + right_curverad)) + 'm'
    line2 = 'Vehicle is ' + '{:.2f}'.format(abs(car_offset)) + 'm '
    if car_offset < 0.0:
        line2 += 'left'
    else:
        line2 += 'right'
    line2 += ' of center'
    cv2.putText(result, line1, (20, 60), font, 1.0, (255, 255, 255))
    cv2.putText(result, line2, (20, 100), font, 1.0, (255, 255, 255))
    # cv2.imwrite('./output_images/final.jpg', result[:, :, ::-1])

    return result
    
process_image.mtx = mtx
process_image.dist = dist
process_image.last_l_centroids = []
process_image.last_r_centroids = []

input_video_filename = "project_video.mp4"
output_video_filename = "project_video_output.mp4"
input_video = VideoFileClip(input_video_filename)
output_video = input_video.fl_image(process_image)
output_video.write_videofile(output_video_filename, audio=False)
