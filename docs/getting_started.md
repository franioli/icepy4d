# Getting started

## Data preparation

For a multi-temporal and multi-camera processing with ICEpy4D, data must be organized as follows:

```
├── config.yaml    # Configuration file
├── data/ 
    ├── img/       # Image folder (one subfolder per camera)
        ├── cam1/ 
        ├── cam2/ 
        ├── cam3/
        ...
    ├── calib/     # Calibration files folder (one file per camera)
        ├── cam1.txt
        ├── cam2.txt
        ├── cam3.txt
        ...
    ├── targets/   # Target files folder (one file per image)
        ├── img_cam1_epoch0.txt
        ├── img_cam1_epoch1.txt
        ├── img_cam1_epoch2.txt
        ...
        ├── img_cam2_epoch0.txt
        ├── img_cam2_epoch1.txt
        ├── img_cam2_epoch2.txt
        ...
        ├── img_cam3_epoch0.txt
        ├── img_cam3_epoch1.txt
        ├── img_cam3_epoch2.txt
        ...        
        ├── targets_world.txt
```

The data directory contains all the data needed for the processing. 
The `img` folder contains one subfolder per camera. 
The `calib` folder contains the calibration files for each camera. 
The `targets` folder contains the targets files. 
Targets file are stored all together in a single folder `targets` folder.
Each target file must be named as with the same name as the image that it belongs to, but with a textfile extension (".txt", ".csv"), and it contains the image coordinates of all the visible targets in that image.
Each file must contain the target label and the image coordinates x and y of all the visible targets.
For instance, the file named `img_cam1_epoch0.txt`, where `img_cam1_epoch0.jpg` is the image file, contains the following data:

```
label,x,y
F1,1501.8344,3969.0095
F2,1003.5037,3859.1558
```

Additionally, a file containing the world coordinates X,Y, Z of all the targets must be provided. This file should be named `targets_world.txt` and it must contain the following data:

```
label,X,Y,Z
F1,-499.8550,402.0301,240.3745
F2,-302.8139,442.8938,221.9927
```
World coordinates must be in a cartesian (e.g., local) or projected (e.g., UTM) coordinate system. 

The `config.yaml` file contains the configuration parameters for the processing.


Some example data can be downloaded from ....

## Configuration file