# stereo-vision
This program conatins a stereo vision pipeline system.Steps taken to build the stereo vision system are:  
Calibration:  
  Match the feature points  
  Estimate the fundamental matrix using RANSAC  
  Obtain Essential Matrix using K matrix and fundamental matrix
  Decompose Essential Matrix into T and R components
  Rectificy the images using homography and plot the epipolar lines  
  Calculate disparity and compare the windows in both images  
  Compute depth using baseline and focal length of the camera

## Dependencies

-   Python
-   Opencv 4.1.0
-   Numpy
-   Matplotlib

## Steps to run the package
1.Clone the project 

    git clone https://github.com/Madhunc5229/stereo-vision

2.cd into the code folder and the run the python file
    
    cd/stereo-vision/code  
    python3 curule.py
    python3 octagon.py
    python3 pendulum.py
    

## Feature matching  
![images_with_matching_keypoints](https://user-images.githubusercontent.com/61328094/163917726-347b2fe7-f786-41ed-98b7-979a9a73243b.png)

## Rectifing and plotting the epipolar lines
![sadf](https://user-images.githubusercontent.com/61328094/163919225-239df20b-b888-43ea-933a-bc75a0dab195.png)

## Disparity Gray scale image  
![dis_gray](https://user-images.githubusercontent.com/61328094/163918197-6785ede0-cafc-4a32-9ecc-922f766f0699.png)


## Disparity Gray heat map
![dis_heat](https://user-images.githubusercontent.com/61328094/163918230-dfab6c3a-e750-4a25-b12b-63b71de93a71.png)

## Depth gray scale image
![depth_gray](https://user-images.githubusercontent.com/61328094/163918353-f74f1f06-1ee6-4453-9edc-c92402d15121.png)

## Depth heat map
![depth_hot](https://user-images.githubusercontent.com/61328094/163918393-28d8923a-32ec-42ce-a7e8-fda1c32da173.png)



