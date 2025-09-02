
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>

#include "yolo_detector.hpp"

namespace object_detection
{
YoloV11::YoloV11(const std::filesystem::path& onnxModelPath, const std::filesystem::path& classesPath, YoloConfig config)
{
    // Store the config.
    this->config = config;

    // Open the file containing the onnx model and check if it is there.
    std::cout << "Looking for DNN model file (.onnx) at: " << onnxModelPath << std::endl;
    std::ifstream file(onnxModelPath);
    if (!file) {
        throw std::runtime_error("Could not open DNN model file: " + onnxModelPath.string());
    }

    // Now open the file containing the class names and read them into the classes vector.
    std::cout << "Looking for classes file (.names) at: " << classesPath << std::endl;
    std::ifstream ifs(classesPath);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Could not open class file: " + classesPath.string());
    }

    // Loop through the file line by line and add each class name to the classes vector.
    std::string line;
    while (std::getline(ifs, line)) 
    {
        if (!line.empty())
        {
            this->classes.push_back(std::move(line));
        }
    }

    // Load the pre-trained YOLOv11 model from the ONNX file.
    net = cv::dnn::readNetFromONNX(onnxModelPath);
    if (net.empty()) {
        throw std::runtime_error("Could not load the DNN model: " + onnxModelPath.string());
    }
}

std::tuple<size_t, size_t> YoloV11::getImageSize() const
{
    return {this->config.inputWidth, this->config.inputHeight};
}

std::vector<Detection> YoloV11::detect(cv::Mat& inputImage)
{    
    // Prepare the input blob for the DNN model.
    // Normalizes  values from [0,255] → [0,1] by multiplying with 1/255.0.
    // true indicates swap RB channels (OpenCV uses BGR by default; YOLO expects RGB).
    // false disables cropping, since we already did letterbox.
    cv::Mat blob = cv::dnn::blobFromImage(inputImage, 1/255.0, cv::Size(config.inputWidth, config.inputHeight), cv::Scalar(), true, false);

    // Set the input blob for the network.
    net.setInput(blob);

    // Perform forward pass to get the output of the output layers.
    std::vector<cv::Mat> outputs;
    net.forward(outputs);

    auto detections = this->postProcess(outputs);
    return detections;
}

std::vector<Detection> YoloV11::postProcess(const std::vector<cv::Mat>& outputs)
{
     // The YOLOv8/YOLOv11 output is a single 1×84×8400 tensor
    CV_Assert(outputs.size() == 1);
    cv::Mat output = outputs[0];

    // Reshape to 8400×84 for easier row access. Each row is a feature map. 
    output = output.reshape(1, output.size[1]); // Now: rows=84, cols=8400
    output = output.t();                        // Transpose to rows=8400, cols=84
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    // Iterate through each row (detection) and find objects that are above the threshold.
    for(int i = 0; i < output.rows; ++i)
    {
        constexpr int offset = 4; // First 4 entries are bounding box coordinates.
        constexpr int noClass = -1;
        float bestScore = 0.0f;
        int bestClassId = noClass;    

        // Iterate through all possible object classes in each box and get the best detected object.
        for(int j = offset; j < output.cols; ++j) // Start at 4 to skip bbox coordinates
        {
            // Get box, confidence and ID of object over the threshold value.
            float confidence = output.at<float>(i, j);
            if(confidence > config.confThreshold)
            {
                // Store the best confidence and class ID, since we only want one class per box.
                if(confidence > bestScore)
                {
                    bestScore = confidence;
                    bestClassId = j - offset; // Subtract 4 to get the class ID.
                }
            }
            // Add the object with the highest detected confidence.
            if (bestClassId != noClass)
            {
                // Extract bounding box coordinates.
                float cx = output.at<float>(i, 0); // Center x
                float cy = output.at<float>(i, 1); // Center y
                float w  = output.at<float>(i, 2); // Width
                float h  = output.at<float>(i, 3); // Height
                float x = cx - w / 2.0f; // Get top-left corner coordinates.
                float y = cy - h / 2.0f;

                cv::Rect box{static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h)};
                boxes.push_back(box);
                scores.push_back(bestScore);
                indices.push_back(bestClassId);
            }
        }
    }

    // Apply Non-Maximum Suppression (NMS) to filter overlapping boxes.
    std::vector<int> keptIndices;
    cv::dnn::NMSBoxes(boxes, scores, config.confThreshold, config.nmsThreshold, keptIndices);

    // Prepare the final detections vector.
    std::vector<Detection> detections;
    for (int idx : keptIndices)
    {
        detections.emplace_back(Detection{boxes[idx], indices[idx], scores[idx]});
    }

    return detections;
}

void YoloV11::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) const
{
    // Visual parameters
    constexpr int boxThickness = 2;   // Thickness of bounding box lines
    constexpr double fontScale = 0.5; // Size of the label text
    constexpr int fontThickness = 1;  // Thickness of the label text
    constexpr int baselineOffset = 2; // Extra padding below text

    // Lambda function to generate a distinct color for each class ID.
    // Generated by LLM.
    auto getClassColorBrg = [](int classId) {
        int hue = (classId * 37) % 180; // Use a prime number to distribute hues
        cv::Mat rgbColor;
        cv::Mat hsvColor(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255)); // Full saturation and value
        cv::cvtColor(hsvColor, rgbColor, cv::COLOR_HSV2BGR);
        return cv::Scalar(rgbColor.at<cv::Vec3b>(0, 0)); // Return as Scalar
    };

    for (const auto& det : detections) 
    {
        // Draw the bounding box depending on the class ID.
        cv::Scalar color = getClassColorBrg(det.classId);
        cv::rectangle(frame, det.box, color, boxThickness);

         // Prepare class name and confidence text.
        std::string label;
        if (det.classId < 0 || det.classId >= static_cast<int>(classes.size())) 
        {
            label = "Unknown";
        }
        else
        {
            label = classes[det.classId];
            label += cv::format(" %.2f", det.confidence);
        }


        // Measure text size
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontThickness, &baseline);
        baseline += baselineOffset;

        // Position label (above box if space, else inside)
        int top = std::max(det.box.y, labelSize.height);
        cv::Point labelOrigin(det.box.x, top);

        // Draw filled rectangle for label background
        cv::rectangle(frame, cv::Point(det.box.x, top - labelSize.height - baseline),
                      cv::Point(det.box.x + labelSize.width, top), color, cv::FILLED);

        // Draw label text (black on colored background)
        cv::putText(frame, label, labelOrigin - cv::Point(0, baseline), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), fontThickness);
    }
}

void YoloV11::drawFPS(cv::Mat& frame, float fps)
{
    // Visual parameters
    const double fontScale = 0.7;            // size of the text
    const int fontThickness = 2;             // thickness of the text
    const int margin = 10;                   // margin from edges
    const cv::Scalar textColor(0, 0, 0);     // black text

    // Prepare FPS label
    std::string label = cv::format("FPS: %.1f", fps);
    // Measure text size
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontThickness, &baseline);
    // Position text in top-right corner
    cv::Point textOrg(frame.cols - textSize.width - margin, margin + textSize.height);
    // Draw the FPS text on top
    cv::putText(frame, label, textOrg, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, fontThickness);
}

} // namespace object_detection