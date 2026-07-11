#pragma once

#include <opencv2/opencv.hpp>

namespace liquid {

struct PreprocessParams {
    int blurKernel = 5;
    double claheClipLimit = 2.5;
    cv::Size claheTileGrid = cv::Size(8, 8);
    int backgroundKernel = 91;
};

struct PreprocessResult {
    cv::Mat gray;
    cv::Mat enhanced;
    cv::Mat background;
    cv::Mat normalizedAbs;
    cv::Mat darkFeature;
    cv::Mat brightFeature;
};

PreprocessResult preprocessRoi(const cv::Mat& roiBgrImage, const PreprocessParams& params = {});

}  // namespace liquid
