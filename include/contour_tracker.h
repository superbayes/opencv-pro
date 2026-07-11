#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "stream_detector.h"

namespace liquid {

struct ContourTrack {
    std::vector<cv::Point2f> left;
    std::vector<cv::Point2f> right;
    std::vector<cv::Point2f> center;
    std::vector<float> widths;
    std::vector<unsigned char> validMask;
    double qualityScore = 0.0;
    cv::Rect scanRect;
};

ContourTrack trackContours(const StreamDetectionResult& streamResult);

}  // namespace liquid
