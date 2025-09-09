# seePlusPlus

This is an application to detect and track objects in C++ using Yolo (You only look once) with OpenCV.
It is used as an introduction project into machine vision with C++.

## Project Features

- Real-time object detection using YOLOv11 and OpenCV DNN module
- Visualizes detection results directly on images
- Build and run with provided docker image
- Github Actions for building
- Cross-platform build support (macOS, Windows, Linux)

![Yolo Detecting Chairs](res/detect_chairs.png) ![Yolo Detecting Cups](res/detect_cup.png)

## Coming Soon

- Static Code Analysis in Github Actions
- Automated Testing in Github Actions

## Requirements

- Docker
- Python >= 3.8
- Optional for local builds:
  - C++20
  - OpenCV 4.12.0
  - CMake >= 4.1

## Build

### With Container

A multi-arch Docker image is provided on Github to build the project. A script to build __Release__ or __Debug__ is provided.\
The build output is found in __build/Container__.\
MacOs and Linux:

```shell
python3 tools/build_scripts/build_local_release.py
```

```shell
python3 tools/build_scripts/build_local_debug.py
```

Windows:

```shell
python tools/build_scripts/build_local_release.py
```

```shell
python tools/build_scripts/build_local_debug.py
```

### Locally

If you want to build the project on your platform you need to have the [requirements](##Requirements) installed.
The project is built using python scripts and CMake which makes is platform independent.\ 
The build output is found in __build/Local.\
MacOs and Linux:

```shell
python3 tools/build_scripts/build_local_release.py
```

```shell
python3 tools/build_scripts/build_local_debug.py
```

Windows:

```shell
python tools/build_scripts/build_local_release.py
```

```shell
python tools/build_scripts/build_local_debug.py
```

### Installation OpenCV MacOs

Install C++ compilers and OpenCV according to the installation guide in [GeeksForGeeks](https://www.geeksforgeeks.org/installation-guide/how-to-install-opencv-for-c-on-macos/).

## Style Guide

This project uses the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html#Self_contained_Headers).
