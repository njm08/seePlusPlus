#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp> // cv::Rect
#include <stdexcept>              // std::runtime_error

namespace vision_utilities
{
void cropCentered(cv::Mat& img, const size_t width, const size_t height)
{
    if (img.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }

    if (width == 0 || height == 0) {
        throw std::invalid_argument("Crop width and height must be non-zero.");
    }

    const int originalWidth = img.cols;
    const int originalHeight = img.rows;

    if (width > originalWidth || height > originalHeight) {
        throw std::invalid_argument("Crop size is larger than the image size.");
    }

    // Find the top left corner of the cropped image.
    const int x = (originalWidth - width) / 2;
    const int y = (originalHeight - height) / 2;
    cv::Rect roi(x, y, width, height);
    img = img(roi);
}
} // namespace vision_utilities