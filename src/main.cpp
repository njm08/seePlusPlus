#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "camera.hpp"

int main() {

  std::cout << "Hello, SeePlusPlus!" << std::endl;

  // TODO make settings file with deviceID.
  int deviceId = 1;  // When the Iphone is turned on, the webcam is 1 and 0 is the Iphone.
  Camera camera{deviceId};
  cv::Mat frame{};

  // Capture the video frame by frame and display it in a window.
  const std::string windowName = "Camera";
  while (true) 
  {
    if(!camera.captureFrame(frame)) 
    {
      std::cout << "Error: Empty frame. Exiting." << std::endl;
      break; // Exit loop if frame is empty
    } 
    else 
    {
      cv::imshow(windowName, frame);
    }

    int key = cv::waitKey(1);
    // Exit loop if 'q', 'Q', or ESC is pressed
    if (key == 'q' || key == 'Q' || key == 27) 
    {
      break;
    }

    // Exit loop if window is closed
    if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) 
    {
      break;
    }
  }

  cv::destroyAllWindows();
  return 0;
}
