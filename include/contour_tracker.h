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
/// <summary>
/// 跟踪轮廓
/// 原理：根据流检测结果,跟踪轮廓的中心点和宽度,并计算轮廓的得分,得分越高,轮廓越居中,越接近方形,得分越低,轮廓越偏,越细长
/// </summary>
/// <param name="streamResult">流检测结果</param>
/// <returns>轮廓跟踪结果</returns>
ContourTrack trackContours(const StreamDetectionResult& streamResult);

}  // namespace liquid
