# CSI: Channel State Investigation
---
This project continues a line of research interested in addressing an issue of privacy in which people have reported finding hidden streaming devices planted in spaces that they trusted. There has been plenty of research that has gone into device detection. This work takes the next step in attempting to actually determine the location of the hidden streaming device that has been detected.

Previous localization efforts have drawn correlations between spikes in network activity with physical location in an environment to determine a location estimate within 2m of the device's actual location. These efforts typically averaged 2-3 minutes in determining a location estimate.

This work improves on this estimation time and also removes the physical requirement of having the user traverse an environment by leveraging Channel State Information (CSI). CSI is a PHY layer characteristic of transmitted signals, which has been proven to be more temporally stable than the RSSI value and provides richer, fine-grained data to learn position from.

## Subfolders
---
### ar_csi_localization/
This subfolder contains an Android Application project meant to be the main interface for users that want to use this system. It's purpose was to capture RGB images of the environment which are then sent to a Django web server to determine physical dimensions (L, W, H) of the environment. The version here presents users with an AR-capable, interactive interface in which the user can place anchors at key corners of the room for the app to calculate the dimensions.

### django_csi_localization
This subfolder contains a Django web application whose purpose is to intake RGB images of environments from client devices to calculate its dimensions. It exposes one REST API endpoint to take in RGB images.

### hidden-device-research
This subfolder contains all of the Python scripts used in pre-processing and collecting CSI data, building and running Keras models, and quick testing. These scripts use models from various projects to accomplish certain tasks such as generating depth images from RGB images, object detection, and identifying vanishing points in images of rooms.
