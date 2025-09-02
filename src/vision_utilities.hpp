#pragma once

#pragma once

#include <cstddef>
#include <opencv2/core/mat.hpp>

namespace vision_utilities
{

//! \brief Crops the input image to the specified height and width, centered around the image center.
//! \param img The input image to be cropped. The cropped image will replace the original image
//! \param height The desired height of the cropped image.
//! \param width The desired width of the cropped image.
//! \throws std::invalid_argument if the input image is empty, crop size is zero, or crop size is larger than the image size.
void cropCentered(cv::Mat& img, const size_t height, const size_t width);

} // namespace vision_utilities