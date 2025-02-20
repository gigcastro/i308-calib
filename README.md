# i308-calib
Calibration tool


## Installation:

Can be installed via pip:
    
    pip install -qq git+https://github.com/udesa-vision/i308-calib.git


## Quick Start

After installing you should be able to run the cli command calibration tool:

    calib-tool

### Arguments

- **video** str or int. 
Specifies the video device, on linux might be something like `/dev/video<N>`

- **resolution** str (optional)
 the requested resolution in the format "`<width>`x`<height>`" in pixels

- **checkerboard** str (optional) default=10x7
 the checkerboard layout in the format "`<width>`x`<height>`" in number of squares

- **square-size** str (optional) default=24.2
 the checkerboard square size in millimeters 

- **config** str (optional)
a .yaml configuration file


### Examples

calibrate camera 0 with default parameters:

    calib --video 0


calibrate with custom parameters:

    calib --video /dev/video3 --resolution 640x480 --checkerboard 9x6 --square-size 32.0 --data data_dir


calibrate with capture configuration file:

    calib --config cfg/capture.yaml



## Capture Configuration File

Some configuration files are provided.

In order to copy the configuration files to the working directory run the command:

```bash

 copy-configs

```


This should create the folder `cfg/` with some configuration files.

Example of capture configuration file (.yaml):

```yaml

    # video device, on linux might be /dev/video<N>
    video: 0

    # name: str (optional)
    #   a name to identify the device
    name: My Cute Camera

    # resolution: str (optional)
    #   requested resolution in the format "<width>x<height>" in pixels
    resolution: 640x480


```


