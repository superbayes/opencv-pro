#pragma once

#include <opencv2/opencv.hpp>

namespace liquid {

/// <summary>
/// 表示检测到的ROI结果
/// </summary>
struct ROIResult {
    cv::Rect rect;
    bool usedFallback = false;
};

ROIResult detectObservationRoi(const cv::Mat& bgrImage);

}  // namespace liquid
