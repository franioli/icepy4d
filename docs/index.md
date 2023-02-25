# Welcome to ICEpy4D

## Image-based Continuos monitoring of glaciers' Evolution with deep learning sfm and low-cost stereo-cameras

[ICEpy4D](https://github.com/franioli/icepy4d) is a Python package for 4D Image-based Continuos monitoring of glaciers' Evolution with deep learning SfM and low-cost stereo-cameras.

`ICEpy4D` is a under active development.

[  [code](https://github.com/franioli/icepy4d) ![github](assets/GitHub-icon.png)  |  [documentation](franioli.github.io/icepy4d/) ]

## Project layout

    main.py       # Main code for running ICEpy4D
    src/
        icepy     # Source code for ICEpy4D
    data/ 
        img/      # Image folder (one subfolder per cam)
            cam1
            cam2 
        calib/    # Calibration files folder
        targets/  # Target files folder
    tools/        # Various tools for point collimation, estimate 3D rototranslation etc.   
    docs/         # The documentation homepage.
    tests/        # Folder containing code for unit tests with PyTest