#Scripts and tests
---
This subfolder contains various Python, Shell, and MatLab scripts used throughout the project to either collect data, pre-process CSI data, or generate and train Keras models.

##Key scripts
### './matlab' directory
---
'extract.m'

This script was provided by the AX-CSI team with the AX-CSI tool for WiFi6 CSI capture to extract the embedded CSI data in the pcap files generated by their tool. Along with the CSI data for a specific frame, the data structure the script returns contains:
+no. of cores
+no. of nss
+timestamp
+seq no.
+mac addr 

At this point in time, we were only interested in the raw CSI data.

'extract\_function.m'
This script was the conversion of 'extract.m' into a MatLab function for use in a Python script leveraging Octave to run MatLab scripts within a Python environment.

### './python' directory
---
'./input' directory
There are some scripts under './scripts' that expect to read input RGB images from this directory. That can be overridden though with the '--input' parameter in those scripts. I used this directory to organize different test cases I was using.

'./model' directory
This subdirectory stored all of the models that became of use throughout the entire project. The models are as follows:

+kitti.h5
+nyu.h5
+resnet50\_rnn\_\_panos2d3d.pth
+resnet50\_rnn\_\_st3d.pth
+yolov7.pt

All of these files are stored on Not Brienne in the UW Bothell CyberSecurity lab under the filepath...

### './scripts' directory
---
'preprocess.py'

This was a script I worked in to proof out the environment sensing portion of the project. It read in RGB images from one of the subfolders under the '../input/' directory, denoted by the '--input' parameter, and used both the DenseDepth and HorizonNet models to generate RGB images and determine vanishing points in those images. Based on the pixel coordinates it returns from the process of both those models, it calculates real world 3D geometrical coordinates of distance from the camera those points in the image to determine geometrical dimensions of the room.

All dependencies for this script should be present in this directory or the subdirectories.

'calc\_density.py'
This script was a testing script to proof out the environment density estimation process of this application. It uses the YOLO object detection model to identify all objects it could in the input RGB images. Then, for each image identified, it creates a 3D point cloud of that image using depth images that were produced in an earlier step. These point clouds create a rough estimate for dimensions of an identified object. These dimensions are then used to calculate the volume of each object. It then adds up all of the calculated volumes for each object and that represents the estimated density.

'preprocess\_mat\_files.py'
The AX-CSI tools generates pcap files to store CSI data in and the AX-CSI team provided an 'extract.m' script that extracts the CSI data from these pcaps and places them in \*.mat files.
This Python script takes these mat files, iterates through each CSI data structure it extracted and saves the data to csv files for further processing.

'create\_datasets.py'

This script takes the csv files generated from the previous step and creates the actual datasets that are used to train the models on. For each csv file, it creates separates the dataframe into 4 sub-dataframes for each antenna that the CSI data was collected from. It then iterates through all four dataframes simultaneously and vertically stacks one frame from each dataframe to create the MxN (antenna X subcarrier) CSI image. It then horizontally stacks this CSI image with environmental information. 

It creates three datasets per csv file - one with the raw CSI data, one with just the magnitude extracted out of the CSI data points, and one with those magnitudes normalized. The outputs from this script or numpy arrays stored in .npy files.

This was done to test if the models performed significantly better than any of the other cases.

'load\_data.py'

This script creates a randomized dataset of all the MxN CSI images created in the previous script. It is currently hard-coded to a 60/40 split of training/test data. It takes in an optional parameter 'norm' to load normalized or un-normalized data. The default is normalized. It also uses a dictionary to ensure loading an equal representation of all grid sectors across all test environments. 

'cnn\_model.py', 'cnn\_model\_norm.py', and 'cnn\_model\_tuner.py'

These scripts build the model that was ultimately used in this project. 'cnn\_model.py' was used to create the first iteration of the model and 'cnn\_model\_norm.py' was to test the model using normalized data. Once the composition of the model was finalized, 'cnn\_model\_tuner.py' was used to incorporate Tensorflow's tuning library to find the best performing hyperparameters for each layer of the model. The final model is documented in the final paper.

'location.py'
This script was a validation check for each prediction by the model. It took the prediction for the which grid in the room it thought the device was, took the true value and calculated the distance between the two points. It converted the room into a 7x3 grid of possible locations and used a simple distance formula and calculated the distance between the two points in terms of how many grids apart. It then multipled this distance by a scalar value that represented how large each sector was in the real world to get an representation of real-world distance.

'e2e\_test.py'
This script was an end-to-end timing test for pre-processing the CSI data, constructing the MxN CSI image, obtaining a location estimate from the model, and then calculating a distance error for the location estimate. 

'transformer\_model.py'
This script is a work-in-progress. I found a paper that had used a transformer model network architecture (the current hot topic in the world of ML development) to improve the performance over a time-series compared to traditional CNN-like architectures. I had started to try and implement such an architecture but due to time constraints could not see it through.

### './router\_scripts' directory
---
This subdirectory contains all of the Shell scripts that were written to automate collecting CSI data on the ASUS RT-AX86U router devices.

'scan\_environment.sh'
This script covers a room split into a 7x3 grid and automates the process of collecting CSI data across all sectors. It captures CSI data across all 4 antennas on the router device and then has two scenarios for each grid - a Line-of-Sight (LoS) and Non-Line-of-Sight (NLOS) scenario.

'live\_scan\_environment.sh'
This script is a reduced version of the above script in which it's supposed to be used for live testing purposes of only capturing one data point for a grid sector. Best used when doing live tests for the system.

'keep\_scan\_alive.sh'
There are certain cases in which the AX-CSI tool freezes while capturing and cannot continue. This script does a live refresh of the tool and allows the AX-CSI tool to pick back up where it left off and continue collecting data. Best to have this running simultaneously while collecting data to reduce freeze time.

'convert\_pcaps\_to\_mat.sh'
This script takes all pcap files generated in one run of data collection and converts them to .mat files for use in other pre-processing scripts. 