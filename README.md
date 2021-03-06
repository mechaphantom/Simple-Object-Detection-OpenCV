# Simple-Object-Detection-OpenCV
A basic vehicle detection application developed using the computer vision library, OpenCV. It examines the video of vehicles passing through a highway and puts those vehicles in the green box. It cannot be said to have very good results, but it is a very instructive exercise to begin with.

![d4](https://user-images.githubusercontent.com/57845882/132128651-683d169d-0cc7-400b-bcc1-d562d2c215b6.png)

## About Object Detector

Firstly, I masked the video using a gaussian mixture-based background/foreground segmentation algorithm. In this way, the program analyzed each frame in the video, separating its values in black or white format and masking it. This made it easier for us to detect moving objects (vehicles) in the video. 

![d3](https://user-images.githubusercontent.com/57845882/132128649-d127f583-0e9c-4690-9bbd-c9bb474b750a.png)

After that, I used the openCV library's findContours() function to define the boundaries of objects masked as white. I went around these contours and painted the moving objects green. Since I didn't want to work on the whole video, I wanted to detect the pixels in the video and work within a certain region. So I created the 'roi' object and did the masking on this area. Finally, I determined the coordinates of the moving objects and calculated their height and width and had a green rectangle drawn around them.

![d2](https://user-images.githubusercontent.com/57845882/132128647-b104aeb2-e9da-4353-a385-dba7f24994af.png)

## About Object Tracker

I used the Euclidean object tracking algorithm (EuclideanDistTracker). This object tracking algorithm is called centroid tracking as it relies on the Euclidean distance between existing object centroids and new object centroids between subsequent frames in a video. 

![euclidean](https://user-images.githubusercontent.com/57845882/132128790-3f66c641-f133-41cb-bb47-e4242f838e6a.png)

I created a list called Detections. I have the program increase the id by 1 whenever it sees a new object. Thus, the id's will appear at the top of the rectangles we have drawn for moving objects. Thanks to the Update function, first the middle point of the object is determined, and then, thanks to a For Loop, it is checked whether this object has been detected before. If this object has not been detected before, we assign a new id to it. We delete unused ids so that they do not appear on the screen.

![euclidean2](https://user-images.githubusercontent.com/57845882/132128792-fe7e138c-ca9b-4150-90e6-c8a43f8858d7.png)

## References (Resources that I have used)

https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

https://www.researchgate.net/figure/a-shows-the-Euclidean-distance-of-tracking-algorithms-and-the-proposed-method-From_fig1_279752104

https://manivannan-ai.medium.com/object-tracking-referenced-with-the-previous-frame-using-euclidean-distance-49118730051a

https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

https://stackoverflow.com/questions/64119213/improving-the-detection-accuracy-by-using-euclidean-distance-between-its-centroi

https://www.youtube.com/watch?v=1FJWXOO1SRI

https://opencv.org/multiple-object-tracking-in-realtime/

https://learnopencv.com/object-tracking-using-opencv-cpp-python/

https://docs.opencv.org/4.5.0/de/de1/group__video__motion.html

https://www.pythonpool.com/cv2-boundingrect/

https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/

https://docs.opencv.org/3.4.14/d6/d17/group__cudabgsegm.html

https://docs.opencv.org/4.5.1/d7/d4d/tutorial_py_thresholding.html

https://www.youtube.com/watch?v=HXDD7-EnGBY

https://www.pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/

https://stackoverflow.com/questions/42453605/how-does-cv2-boundingrect-function-of-opencv-work

https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

https://docs.opencv.org/4.5.1/d9/d8b/tutorial_py_contours_hierarchy.html

https://towardsdatascience.com/computer-vision-for-beginners-part-4-64a8d9856208

https://manivannan-ai.medium.com/object-tracking-referenced-with-the-previous-frame-using-euclidean-distance-49118730051a
