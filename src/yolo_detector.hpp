
#pragma once
#include <filesystem>
#include <vector>
#include <string>
#include <tuple>
#include <cstddef>
#include <opencv2/dnn.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace object_detection
{

//! \brief A struct representing a single detection result.
//! \param classId The class ID of the detected object.
//! \param confidence The confidence score of the detection.
//! \param box The bounding box of the detected object.
struct Detection {
    cv::Rect box;      // Bounding box of the detected object.
    int classId;       // Class ID of the detected object.
    float confidence;  // Confidence score of the detection.
    Detection(cv::Rect boundingBox, int id = -1, float conf = 0.0) : box(boundingBox),  classId(id), confidence(conf) {}
};

//! \brief Configuration parameters for the YOLO model.
//! \param inputWidth The width of the input image for the DNN model.
//! \param inputHeight The height of the input image for the DNN model.
//! \param confThreshold The confidence threshold for filtering detections.
//! \param nmsThreshold The non-maximum suppression threshold.  
struct YoloConfig {
    int inputWidth;
    int inputHeight;
    float confThreshold;
    float nmsThreshold;
    YoloConfig(int width = 640, int height = 640, float confThresh = 0.25f, float nmsThresh = 0.45f)
        : inputWidth(width), inputHeight(height), confThreshold(confThresh), nmsThreshold(nmsThresh) {}
};

//! \brief A class for performing object detection using a pre-trained DNN model with YOLOv11 architecture.
//! \details The class encapsulates the loading and inference of a DNN model for object detection.
//! \details It uses a YOLOv11 model for detection.
class YoloV11 
{
public:
    //! \brief Constructor initializes the DNN model with the given paths to the ONNX model and class names file.
    //! \param onnxModelPath The filesystem path to the ONNX model file.
    //! \param classesPath The filesystem path to the text file containing class names.
    //! \param config Configuration parameters for the YOLO model.
    explicit YoloV11(const std::filesystem::path& onnxModelPath, const std::filesystem::path& classesPath, YoloConfig config);
    ~YoloV11() = default;
    // Allow moving, but disable copying, since it is unclear if the DNN model can be copied.
    YoloV11(const YoloV11&) = delete;
    YoloV11& operator=(const YoloV11&) = delete;
    YoloV11(YoloV11&&) = default;
    YoloV11& operator=(YoloV11&&) = default;

    //! \brief Gets the expected input width and height of the DNN model.
    //! \return A tuple containing the input width and height.
    std::tuple<size_t, size_t> getImageSize() const;

    //! \brief Gets the class names used by the DNN model.
    //! \details Copy is done to avoid dangling references.
    //! \return A vector of class names.
    std::vector<std::string> getClassNames() const { return this->classes; }

    //! \brief Performs object detection on the input image.
    //! \param inputImage The input image on which to perform object detection. The image is not modified.
    //! \return A vector of Detection structs representing the detection results.
    std::vector<Detection> detect(cv::Mat& inputImage);

    //! \brief Draws detection results on the input image.
    //! \param image The image on which to draw the detection results. The image is modified.
    //! \param detections A vector of Detection structs representing the detection results.
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) const;

    //! \brief Draws the frames-per-second (FPS) on the input image.
    //! \param frame The image on which to draw the FPS. The image is modified.
    //! \param fps The frames-per-second value to display.
    void drawFPS(cv::Mat& frame, float fps);

private:
    cv::dnn::Net net;   //! \brief The DNN model used for object detection.
    std::vector<std::string> classes; //! \brief The class names used by the DNN model.
    YoloConfig config; //! \brief Configuration parameters for the YOLO model.

    void setBackend(); // TODO Experiment with different backends and targets.

    //! \brief Post-processes the raw output from the DNN model.
    //! \detail This includes filtering by confidence threshold and applying non-maximum suppression (NMS).
    //! \param output The raw output from the DNN model.
    //! \return A vector of Detection structs representing the final detection results after post-processing.
    [[nodiscard]] std::vector<Detection> postProcess(const std::vector<cv::Mat>& output);
};

} // namespace object_detection