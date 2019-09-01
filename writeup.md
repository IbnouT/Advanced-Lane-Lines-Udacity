# **Advanced Lane Lines Finding**

## Project2 for Self-Driving Cars Nanodegree @Udacity

---

<p align="center">
  <img src="./images/project_video_processed3.gif">
</p>

---

## **Overview**

The objective of this project is to build an advanced image processing pipeline for identifying lane lines on the road.

I've considered the following steps to calibrate the camera and build a processing pipeline:
1. Camera calibration: computation of the calibration matrix and distortion coefficients.
2. Distortion correction of raw images.
3. Creation of thresholded binary images: combination of colors selecton and sobel gradient operators
4. Perspective transform to rectify binary images: "birds-eye" view
5. Detection of lane pixels and fit to find the lane boundary
6. Computation of the curvature of the lane and vehicle position with respect to center
7. Warping the detected lane boundaries back onto the original image
8. Output visual display of the lane boundaries and estimation of lane curvature and vehicle position

The main entry point :
- [Pipeline.ipynb](./Pipeline.ipynb) : Jupyter notebook with full step by step implementation of the pipeline and testing on images & videos.


Final results of the videos processing can be found in the folder [output_videos](output_videos/):
* Project Video:
  - [project_video.mp4](output_videos/project_video.mp4)

* Challenge Video:
  - [challenge_video.mp4](output_videos/challenge_video.mp4)

* Harder Challenge Video:
  - [harder_challenge_video.mp4](output_videos/harder_challenge_video.mp4)


[![Video Challenge](./images/project_video_processed3.gif?raw=true)](https://youtu.be/6i8Ooc6_kjk)

[//]: # (Image References)

[orig_cal1]: ./camera_cal/calibration2.jpg "Calibration Image 1"
[corners_cal1]: ./output_images/with_corners_calibration2.jpg "Calibration Image with Corners 2"
[orig_cal2]: ./camera_cal/calibration1.jpg "Calibration Image 2"
[undistorted_cal2]: ./output_images/undistorted_calibration1.jpg "Undistorted Calibration Image 2"
[test_image]: ./test_images/test2.jpg "Calibration Image 2"
[undistorted_test]: ./output_images/undistorted_test2.jpg "Undistorted Calibration Image 2"
[test_image2]: ./output_images/undistorted_00147_image.jpg "Undistorted test inage 2"
[adjusted_test_image2]: ./output_images/adjusted_brightness_undistorted_00147_image.jpg "Adjusted test inage 2"
[white_test_image2]: ./output_images/white_color_undistorted_00147_image.jpg "White Selection test inage 2"
[yellow_test_image2]: ./output_images/yellow_color_undistorted_00147_image.jpg "Yellow Selection test inage 2"
[test_image3]: ./output_images/undistorted_42_image.jpg "Remove Other Colors"
[other_color_test_image3]: ./output_images/other_color_undistorted_42_image.jpg "Remove Other Colors"
[gradient_test_image3]: ./output_images/gradient_undistorted_42_image.jpg "Gradient Detection"
[combined_color_gradient]: ./output_images/thresholded_undistorted_42_image.jpg "Combined Color & Gradient Detection"

[thresholding]: ./images/Thresholding.png "Thresholding"

[pipeline]: ./images/pipeline.png "Pipeline"
[original]: ./test_images/solidYellowLeft.jpg "Original Image"
[color_selection]: ./test_images_output/1_color_select_solidYellowLeft.jpg "Selection of White/Yellow color"
[gray_scaled]: ./test_images_output/2_gray_scaled_solidYellowLeft.jpg "Gray Scaled"
[blur_grayed]: ./test_images_output/3_blur_grayed_solidYellowLeft.jpg "Noise Reduction"
[edges]: ./test_images_output/4_edges_solidYellowLeft.jpg "Edges Detection"
[edges_region]: ./test_images_output/5_masked_edges_solidYellowLeft.jpg "Region Selection 2"
[rawlines_1]: ./images/LinesYellowLeft.png "raw lines - YellowLeft"
[rawlines_2]: ./images/LinesSolidWhiteRight.png "raw lines - SolidWhiteRight"
[line_ex_1]: ./test_images_output/6_lines_detection_solidYellowLeft.jpg "Extended lines - - YellowLeft"
[line_ex_2]: ./test_images_output/6_lines_detection_solidWhiteRight.jpg "Extended lines - - SolidWhiteRight"
[region_selection]: ./images/region_select_solidYellowLeft.png "Region Selection 1"
[example_widget]: ./images/example_widget.png "Example Interactive Widget"
[road_up_hill]: ./images/road_up_hill.jpg "Example Road Up Hill"

---

## **Processing Steps**

### 1. Camera Calibration

In this step I used 9x6 chessboard images. For each image, I keep track of:
- the coordinates of the chessboard corners in the world, which is assumed to be the same for all images. Coordinates are append to the `objpoints` list.
- the (x, y) pixel position of each of the corners in the image plane where the chessboard has been successfully detection with `cv2.findChessboard`. The position is added to the `imgpoints` list.

The code for this step is contained in the 4th code cell of the IPython notebook.

| Calibration Image | Chessboard corners detected |
| :---: |:---:|
| ![alt text][orig_cal1]  | ![alt text][corners_cal1]  |

`cv2.calibrateCamera()` function is then used to compute the camera calibration matrix and distortion coefficients:
```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
```
This is how the calibration matrix and distortion coefficients are used later to undistort the raw images:
```
undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
```

| Another Calibration Image | Undistorted Calibration Image |
| :---: |:---:|
| ![alt text][orig_cal2]  | ![alt text][undistorted_cal2]  |

### 2. Building the Pipeline
#### Step 1: Distortion correction
The first step of the pipeline is to undistort the image as described above. See code snippet in 8th code cell of the IPython notebook which use an interactive function to explore the effect on each test image. The following is an example of the result applied to a test image:

| Test Image | Undistorted Test Image |
| :---: |:---:|
| ![alt text][test_image]  | ![alt text][undistorted_test]  |

#### Step 2: Thresholded Binary Image
This step combines color and gradient thresholding techniques.

##### 2.1 Adjusting brightness & contrast
I've noticed the brightness on the images can vary a lot, this is typical in the hardest challenge video. The issue with that is to get fixed color threshold values that can be applied to all images.
The idea I had to somehow overcome that issue is to compute the average luminescence of the image and if it is higher or lower than some threshold values then I increase or decrease the brightness. The contrast is also adjusted by a fixed value.
Threshold values are fixed empirically.
```
# Convert to HLS
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
mean_lumi = np.mean(hls[:,:,1])

if mean_lumi > 110:
    britghness = (100 - mean_lumi)*1.5
    image = apply_brightness_contrast(image, britghness, 20)
elif mean_lumi < 80:
    britghness = (80 - mean_lumi)*1.5
    image = apply_brightness_contrast(image, britghness, 20)
```
The result looks like the following:

| brighter Image | Adjusted brightness and contrast |
| :---: |:---:|
| ![alt text][test_image2]  | ![alt text][adjusted_test_image2]  |

##### 2.2 Color thresholding
###### 2.2.1 White Color

For white color detection I used an intersection of RGB (all channels) & HLS (only L channel) color spaces. The thresholding values are the results of some trials & errors exercise:
```
RGB_LOW_WHITE = np.array([200, 200, 200])
RGB_UPPER_WHITE = np.array([255,255,255])
HLS_LOW_WHITE = np.array([0, 200, 0])
HLS_UPPER_WHITE = np.array([255,255,255])

# create binary image for white colors
white_rgb_binary = cv2.inRange(image, RGB_LOW_WHITE, RGB_UPPER_WHITE) // 255
white_hls_binary = cv2.inRange(hls, HLS_LOW_WHITE, HLS_UPPER_WHITE) // 255
```
The resulting white mask is based on the intersection `(white_rgb_binary == 1) & (white_hls_binary == 1)`:

| Test Image | Selection of white color |
| :---: |:---:|
| ![alt text][test_image2]  | ![alt text][white_test_image2]  |

###### 2.2.1 Yellow Color
To select yellow color however I only considered `H` and `S` channels of HLS color space. Again threshold values are empiric.
```
HLS_LOW_YELLOW = np.array([10, 0, 100])
HLS_UPPER_YELLOW = np.array([40, 255, 255])

# create binary image for yellow color
yellow_hls_binary = cv2.inRange(hls, HLS_LOW_YELLOW, HLS_UPPER_YELLOW) // 255
```

| Test Image | Selection of yellow color |
| :---: |:---:|
| ![alt text][test_image2]  | ![alt text][yellow_test_image2]  |

###### 2.2.1 Other uninteresting Colors
One idea I explored was to remove as much as possible pixels with uninteresting colors. For example the road color on the images/video are in large extend close to a nuance of gray and blue. So removing those colors, and any other nuance of unlikely color of interest might help reduce noisiness and eliminate some false detection when using gradient thresholding.
```
HLS_LOW_GRAY = np.array([0, 0, 0])
HLS_UPPER_GRAY = np.array([255, 150, 20])
HLS_LOW_OTHER = np.array([90, 0, 0]) #95
HLS_UPPER_OTHER = np.array([255, 150, 255]) #120

gray_hls_binary = cv2.inRange(hls, HLS_LOW_GRAY, HLS_UPPER_GRAY) // 255
other_hls_binary = cv2.inRange(hls, HLS_LOW_OTHER, HLS_UPPER_OTHER) // 255
```
The following shows the new area of interest after removing uninteresting colors. We can see how it helped remove the area on the road with dark gray color which would have otherwise be detected when using gradient operators.

| Test Image | Partially removing uninteresting colors |
| :---: |:---:|
| ![alt text][test_image3]  | ![alt text][other_color_test_image3]  |

##### 2.3 Gradient thresholding
Here I used a union of :
- sobel directional gradient operators (`sobel x`, `sobel y`):

`def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255))` (in 13th cell code)

- grandient magnitude of both sobel directional operators:

`def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255))` (in 13th cell code)

The combined gradient (`sobel x` | `sobel y` | `magnitude`) is then intersected with the `other colors` area of interest defined in section `2.2.1` above.
```
# Calculate directional X gradient
sobel_x = abs_sobel_thresh(image, orient='x', sobel_kernel=sobel_kernel, thresh=thresh)
sobel_y = abs_sobel_thresh(image, orient='y', sobel_kernel=sobel_kernel, thresh=thresh)
mag_binary = mag_thresh(image, sobel_kernel=sobel_kernel, mag_thresh=mg_thresh)

combined_sobel = np.zeros_like(sobel_x)
combined_sobel[(sobel_x == 1) | (sobel_y == 1) | (mag_binary == 1)] = 1

# Combined binary threshold
combined_binary = np.zeros_like(sobel_x)
combined_binary[((combined_sobel == 1) & (gray_hls_binary == 0) & (other_hls_binary == 0)) |...]
```
An illustration of the result is provided as following:

| Test Image | Combined Gradient + other colors area |
| :---: |:---:|
| ![alt text][test_image3]  | ![alt text][gradient_test_image3]  |

##### 2.4 Combining all thresholding
Finally all the color and gradient thresholding are combined as shown in the following diagram:
![alt text][thresholding]
The code is implemented by the function `color_gradient_threshold` in the 14th cell code of the IPython notebook.
As example of output:

| Test Image | Combined Gradient + other colors area |
| :---: |:---:|
| ![alt text][test_image3]  | ![alt text][combined_color_gradient]  |

#### Step 3: Perspective Transform
