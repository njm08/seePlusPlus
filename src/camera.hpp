#pragma once

#include <opencv2/videoio.hpp>

//! \brief A class for capturing video from a camera device.
//! \param cameraIndex The index of the camera device to use (default is 0).
//! \throws std::runtime_error if the camera cannot be opened.
class Camera {
public:
    explicit Camera(const int cameraIndex = 0);
    ~Camera() = default;
    Camera(const Camera&) = delete; // Copying the camera object is not allowed, since it manages a hardware resource.
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&) = default; // Moving is allowed, since the resource can be transferred.
    Camera& operator=(Camera&&) = default;

    //! \brief Captures a single frame from the camera.
    //! \param frame A reference to a cv::Mat object where the captured frame will be stored.
    //! \returns true if the frame was captured successfully, false otherwise.
    bool captureFrame(cv::Mat& frame);

    private:
    cv::VideoCapture videoCapture;
};

