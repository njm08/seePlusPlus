#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <chrono>

#include "camera.hpp"
#include "vision_utilities.hpp"
#include "yolo_detector.hpp"

//*********************. Configuration ********************
constexpr int CAMERA_INDEX = 0; // Change this to the appropriate camera index if needed.
//*********************************************************

int main() {

  std::cout << "Hello, SeePlusPlus!" << std::endl;

  // Load the DNN model for object detection. The model should be in ONNX format.
  // It can be created using the python script in the tools/yolo_export folder.
  std::cout << "Loading the DNN model..." << std::endl;
  const std::filesystem::path onnxFilePath = std::filesystem::current_path() / "res" / "yolo11n.onnx"; // TODO Use a config file or command line argument to get the path to the model.
  const std::filesystem::path classesFilePath = std::filesystem::current_path() / "res" / "coco.names";

  if (!std::filesystem::exists(onnxFilePath)) {
    std::cerr << "Error: Model file not found: " << onnxFilePath << std::endl;
    return 1;
  }
  if (!std::filesystem::exists(classesFilePath)) {
    std::cerr << "Error: Class names file not found: " << classesFilePath << std::endl;
    return 1;
  }

  // Create the YoloV11 detector with default configuration.
  // Create the YoloV11 detector with default configuration.
  object_detection::YoloConfig defaultYoloConfig {};
  object_detection::YoloV11 yoloDetector{onnxFilePath, classesFilePath, defaultYoloConfig};

  // Open the camera.
  std::cout << "Opening the camera..." << std::endl;
  Camera camera{CAMERA_INDEX};

  // Capture the video frame by frame, apply the Yolo detector, and display the results.
  std::cout << "Starting the video capture..." << std::endl;
  const std::string windowName = "Yolo Detection";
  auto classNames = yoloDetector.getClassNames();
  cv::Mat frame{};
  while (true) 
  {
    if(!camera.captureFrame(frame)) 
    {
      std::cout << "Error: Empty frame. Exiting." << std::endl;
      break; // Exit loop if frame is empty
    } 
    else 
    {
      // Crop the image to fit the DNN input size.
      auto [inputWidth, inputHeight] = yoloDetector.getImageSize();
      vision_utilities::cropCentered(frame, inputWidth, inputHeight);

      // Perform object detection and measure the time taken.
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      auto detectionResult = yoloDetector.detect(frame);
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      auto timeSpanMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
      auto frameRateHz = 1000.0 / timeSpanMs.count();

      // Draw the detection results and display the frame.
      yoloDetector.drawDetections(frame, detectionResult);
      yoloDetector.drawFPS(frame, frameRateHz);
      cv::imshow(windowName, frame);
    }

    // Exit loop if 'q', 'Q', or ESC is pressed
    int key = cv::waitKey(1);
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
