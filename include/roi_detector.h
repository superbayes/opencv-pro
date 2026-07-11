#pragma once

#include <opencv2/opencv.hpp>

namespace liquid {

struct ROIResult {
    cv::Rect rect;
    bool usedFallback = false;
};

ROIResult detectObservationRoi(const cv::Mat& bgrImage);

}  // namespace liquid
