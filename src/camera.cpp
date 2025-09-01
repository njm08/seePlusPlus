#include "camera.hpp"
#include <stdexcept>
#include "camera.hpp"

Camera::Camera(const int cameraIndex) 
{
    // Open the configured camera and check if everything went well.
    videoCapture = cv::VideoCapture{cameraIndex};
    bool isOpened = videoCapture.isOpened();
    if (isOpened == false) {
        throw std::runtime_error("Could not open camera index " + std::to_string(cameraIndex));
    }
}

bool Camera::captureFrame(cv::Mat& frame)
{
    if (!videoCapture.isOpened()) 
    {
        throw std::runtime_error("Camera is not opened.");
    }
    else 
    {
        videoCapture >> frame; // Capture a new frame.
    }

    bool retValue = true;
    if (frame.empty()) 
    {
        retValue = false;
    }

    return retValue;
}