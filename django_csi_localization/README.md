# Django web server for CSI localization
---
The project contained here contains a Django web server whose purpose is to intake RGB images of environments from client deviecs to calculate its dimensions. It exposes one REST API endpoint to take in RGB images and returns estimates for length, width, height and density of the environment.

To determine L, W, and H estimates of the room the application leverages two models - one to generate depth images from the RGB images and another model to identify vanishing points (corners) of the room. It uses pixel coordinate estimates in the provided depth images in real world focal length calculations to determine distances from the point the user was standing. It then performs standard geometry to produce the estimates.

To estimate density, the application uses the same RGB images provided earlier and feeds them to YOLO, an object detection model, to find all objects in the environment. To estimate density of each object, a pseudo-LiDAR approach is taken using the depth images generated in the previous step to create 3D point clouds of each object. Bounding boxes that surround the identified objects are then used to calculate the dimensions of each image. And density is determined by simply adding the densities of all identified objects.


The following model files are stored on 'Not Brienne' in the UW Bothell CyberSecurity lab at the file path `<ENTER FILEPATH>`

+ nyu.h5
+ resnet50\_rnn\_\_panos2d3d.pth
+ resnet50\_rnn\_\_st3d.pth
+ yolov7.pt
+ kitti.h5

Place all of these files in the `./models` directory

---
## Using the application

To run the application:

`python3 manage.py runserver`

The application should be running on localhost port 8000

The only endpoint in the application takes a POST call to port 8000 on the URI /density

It takes in two body parameters 'front' and 'back' for the RGB images

![postman](img/postman.png)
