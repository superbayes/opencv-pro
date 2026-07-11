#pragma once

#include <opencv2/opencv.hpp>

#include "preprocess.h"

namespace liquid {

struct StreamDetectionResult {
    bool streamIsDark = true;
    double angleDegrees = 0.0;
    int centerX = 0;
    int halfWidth = 24;
    cv::Rect scanRect;
    cv::Mat rotatedGray;
    cv::Mat rotatedFeature;
    cv::Mat rotatedEdge;
};

StreamDetectionResult detectStream(const PreprocessResult& preprocessResult);

}  // namespace liquid
