# Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[car_img]: ./examples/car_img.png
[noncar_img]: ./examples/noncar_img.png
[0.5]: ./examples/0.5.jpg
[1.0]: ./examples/1.0.jpg
[1.5]: ./examples/1.5.jpg
[2.0]: ./examples/2.0.jpg

[test-0]: ./examples/test-0.jpg
[test-2]: ./examples/test-2.jpg
[test-3]: ./examples/test-3.jpg
[test-4]: ./examples/test-4.jpg
[test-5]: ./examples/test-5.jpg

[proc-1-org]: ./examples/org-1.jpg
[proc-1-heat]: ./examples/process-heat-1.jpg
[proc-1-res]: ./examples/process-result-1.jpg
[proc-2-org]: ./examples/org-2.jpg
[proc-2-heat]: ./examples/process-heat-2.jpg
[proc-2-res]: ./examples/process-result-2.jpg
[proc-3-org]: ./examples/org-3.jpg
[proc-3-heat]: ./examples/process-heat-3.jpg
[proc-3-res]: ./examples/process-result-3.jpg
[proc-4-org]: ./examples/org-4.jpg
[proc-4-heat]: ./examples/process-heat-4.jpg
[proc-4-res]: ./examples/process-result-4.jpg
[proc-5-org]: ./examples/org-5.jpg
[proc-5-heat]: ./examples/process-heat-5.jpg
[proc-5-res]: ./examples/process-result-5.jpg



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in feature_extraction.py in the function `get_hog_features` on line 13.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_img] ![alt_text][noncar_img]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I used the parameters that were given in the lectures.  These parameters worked well so I never changed them.
These arguments were `orient = 9`, `pixels_per_cell = 8`, `cell_per_block = 2`.  I did however decide to use all 
the color channels and I used the `YCrCb` color space since it outperformed the other color spaces when classifying with 
a linear svm.
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG features and color features.  This can be seen in `classify.py`.  The main function
`extract_features` in `feature_extraction.py` extracts the color and HOG features for each image.  The features
are extracted in `main()` and are then passed to `process_data_for_train` to be normalized and split
into a test and train test.  The data is then passed as a dictionary to `create_model` where the linear SVM classifier
is saved as a pkl file.  

The HOG and color feature functions can be found in `feature_extraction.py`.  The `find_cars` function calls each function to
extract features that are concatenated into one large array.  This is then normalized and fed to the classifier to obtain a result
of 0 or 1, where 1 corresponds to a vehicle.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In `feature_extraction.py` the function `find_cars` uses a efficient method for doing the sliding window approach.
Here I extract the HOG features once and I sub-sample the results to get the overlaying windows.  I tried out different scale results 
`[0.5, 1.0, 1.5, 2.0]` to search for cars farther down the horizon and for cars closer to the camera.  This seems to work well,
but it takes an extremely long time since it has to search a large number of windows.  For the video, I decided to only use
a scale of 1.5 and an overlap of `cells_per_step = 1`.

Here are some images of different scale results with `cells_per_step = 1`

0.5       |  1.0 
:--------:|:---------:
![][0.5]  |  ![][1.0]


1.5       |  2.0
:--------:|:---------:
![][1.5]  |  ![][2.0]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color 
in the feature vector, which provided a nice result.  

In some of the examples there are false positives, but when you run the video pipeline, you can keep track of previous
frames to remove false positives.
To improve performance, the smaller scales search in a smaller range since the smaller vehicles can only exist if they
are further away from the camera.  The HOG features are generated once and are sub sampled for different windows.

Here are some example images:


![][test-0]   ![][test-2]
![][test-3]   ![][test-4]
![][test-5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In `WindowTracker.py` the `WindowTracker` class keeps track of the bounding boxes for the last n frames, where I 
chose n to be 5.  In this case a heat map is generated for all the frames in the current queue.  This heatmap is then
thresholded to remove labels that have not shown up in all the frames.

### Here are five frames:
The left most frame contains the results from the `find_cars` function.  The middle frame is a heat map of the resulting 
n frames from the queue that keeps track of previous frames, and the last frame is the end result after applying
`label()` to the heat map.

![][proc-1-org] ![][proc-1-heat] ![][proc-1-res]

![][proc-2-org] ![][proc-2-heat] ![][proc-2-res]

![][proc-3-org] ![][proc-3-heat] ![][proc-3-res]

![][proc-4-org] ![][proc-4-heat] ![][proc-4-res]

![][proc-5-org] ![][proc-5-heat] ![][proc-5-res]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

If the car is all the way in the far left lane, then the classifier picks up vehicles from cars in the 
opposite direction.  This might not be wanted on the highway, but it could be of use in avoiding incoming vehicles 
on regular streets.

A linear SVM was used, but the results could be improved if a rbf kernel was used.  The linear SVM seems slow
and by using a rbf kernel, it takes even longer to train and evaluate an image.  This leaves a lot of room for improvement
since the SVM approach cannot keep up with even a normal 30 fps camera.  If the classifier cannot evaluate at least 30
fps, then it would not be very useful in real time.

Currently searching with a multitude of different scales takes a very long time since the number of windows greatly 
increases. From what I have read, people have had better results by using a deep learning approach.  An approach
known as YOLO (You only train once) seems to work very well and has almost no false positives.