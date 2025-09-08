#!/bin/bash
# Always print the welcome message, then run the given command

echo "👋 Welcome to the C++ with OpenCV Build Environment! 🚀"
echo "Available tools:"
g++ --version
cmake --version
echo "OpenCV version: $(pkg-config --modversion opencv4)"
echo "OpenCV libs: core, imgproc, imgcodecs, highgui, videoio, dnn"
echo "You're logged in as user: $(whoami)"
exec "$@"
