#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

namespace liquid {

struct LiquidFlowRectResult {
    bool found = false;
    cv::RotatedRect rotatedRect;
    cv::Rect boundingRect;
    std::vector<cv::Point> contour;
    cv::Mat binaryMask;
    double angleFromVertical = 0.0;
    double score = 0.0;
};

namespace detail {

inline double computeAngleFromVertical(const cv::RotatedRect& rect) {
    cv::Point2f points[4];
    rect.points(points);

    double longestLength = -1.0;
    cv::Point2f longestEdge(0.0F, 1.0F);
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f edge = points[(i + 1) % 4] - points[i];
        const double length = std::hypot(static_cast<double>(edge.x), static_cast<double>(edge.y));
        if (length > longestLength) {
            longestLength = length;
            longestEdge = edge;
        }
    }

    const double angle = std::atan2(std::abs(static_cast<double>(longestEdge.x)),
                                    std::abs(static_cast<double>(longestEdge.y))) *
                         180.0 / CV_PI;
    return angle;
}

inline cv::Rect makeValidRoi(const cv::Size& imageSize, const cv::Rect& roi) {
    const cv::Rect full(0, 0, imageSize.width, imageSize.height);
    if (roi.width <= 0 || roi.height <= 0) {
        return full;
    }
    return roi & full;
}

inline double contourFeatureScore(const cv::Mat& featureImage,
                                  const std::vector<cv::Point>& contour,
                                  const cv::Rect& boundingRect) {
    if (contour.empty() || boundingRect.width <= 0 || boundingRect.height <= 0) {
        return 0.0;
    }

    cv::Mat contourMask = cv::Mat::zeros(boundingRect.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> shiftedContours(1);
    shiftedContours[0].reserve(contour.size());
    for (const cv::Point& point : contour) {
        shiftedContours[0].emplace_back(point - boundingRect.tl());
    }

    cv::drawContours(contourMask, shiftedContours, 0, cv::Scalar(255), cv::FILLED);
    return cv::mean(featureImage(boundingRect), contourMask)[0] / 255.0;
}

}  // namespace detail

/// <summary>
/// 检测近似竖直但允许有一定倾斜角度的液流轮廓，并返回更精确的旋转包围框。
/// 适合暗液流落在较亮背景上的场景；若液流比背景更亮，可将 streamIsDark 设为 false。
/// </summary>
/// <param name="image">输入图像，支持灰度图或 BGR 图。</param>
/// <param name="roi">可选检测区域，默认在整幅图像内搜索。</param>
/// <param name="streamIsDark">true 表示检测暗液流，false 表示检测亮液流。</param>
/// <returns>检测结果，found=false 表示未找到满足几何约束的液流。</returns>
inline LiquidFlowRectResult getTiltedVerticalLiquidFlowRect(const cv::Mat& image,
                                                            const cv::Rect& roi = cv::Rect(),
                                                            bool streamIsDark = true) {
    LiquidFlowRectResult result;
    if (image.empty()) {
        return result;
    }

    const cv::Rect validRoi = detail::makeValidRoi(image.size(), roi);
    if (validRoi.width <= 0 || validRoi.height <= 0) {
        return result;
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image(validRoi).clone();
    } else {
        cv::cvtColor(image(validRoi), gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);

    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(8, 8));
    clahe->apply(blurred, enhanced);

    const int verticalKernelHeight = std::clamp((gray.rows / 6) | 1, 21, 151);
    const int verticalKernelWidth = std::clamp((gray.cols / 80) | 1, 3, 9);
    const cv::Mat verticalKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(verticalKernelWidth, verticalKernelHeight));

    cv::Mat feature;
    cv::morphologyEx(enhanced,
                     feature,
                     streamIsDark ? cv::MORPH_BLACKHAT : cv::MORPH_TOPHAT,
                     verticalKernel);
    cv::normalize(feature, feature, 0, 255, cv::NORM_MINMAX);

    cv::Mat gradX16;
    cv::Mat gradX;
    cv::Scharr(feature, gradX16, CV_16S, 1, 0);
    cv::convertScaleAbs(gradX16, gradX);

    cv::Mat response;
    cv::addWeighted(feature, 0.7, gradX, 0.3, 0.0, response);

    cv::Mat binaryMask;
    cv::threshold(response, binaryMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    const cv::Mat connectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 15));
    const cv::Mat cleanKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
    cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_CLOSE, connectKernel);
    cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_OPEN, cleanKernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return result;
    }

    double bestScore = -std::numeric_limits<double>::infinity();
    std::vector<cv::Point> bestContour;
    cv::RotatedRect bestRect;

    for (const std::vector<cv::Point>& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < 40.0) {
            continue;
        }

        const cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
        const double major = std::max(rotatedRect.size.width, rotatedRect.size.height);
        const double minor = std::max(1.0F, std::min(rotatedRect.size.width, rotatedRect.size.height));
        const double aspectRatio = major / minor;
        const double angleFromVertical = detail::computeAngleFromVertical(rotatedRect);

        if (major < gray.rows * 0.08 || aspectRatio < 2.5 || angleFromVertical > 25.0) {
            continue;
        }

        cv::Rect boundingRect = cv::boundingRect(contour) & cv::Rect(0, 0, gray.cols, gray.rows);
        if (boundingRect.width <= 0 || boundingRect.height <= 0) {
            continue;
        }

        const double fillRatio = area / std::max(1.0, static_cast<double>(rotatedRect.size.area()));
        const double featureScore = detail::contourFeatureScore(response, contour, boundingRect);
        const double verticalScore = std::max(0.0, 1.0 - angleFromVertical / 25.0);
        const double score =
            std::sqrt(area) * (0.8 + 0.2 * std::min(aspectRatio, 12.0)) * fillRatio * (0.5 + featureScore) *
            (0.5 + verticalScore);

        if (score > bestScore) {
            bestScore = score;
            bestContour = contour;
            bestRect = rotatedRect;
        }
    }

    if (bestContour.empty()) {
        return result;
    }

    cv::Mat selectedMask = cv::Mat::zeros(binaryMask.size(), CV_8UC1);
    cv::drawContours(selectedMask, std::vector<std::vector<cv::Point>>{bestContour}, 0, cv::Scalar(255), cv::FILLED);

    for (cv::Point& point : bestContour) {
        point += validRoi.tl();
    }
    bestRect.center.x += static_cast<float>(validRoi.x);
    bestRect.center.y += static_cast<float>(validRoi.y);

    result.found = true;
    result.contour = std::move(bestContour);
    result.rotatedRect = bestRect;
    result.boundingRect = cv::boundingRect(result.contour) & cv::Rect(0, 0, image.cols, image.rows);
    result.angleFromVertical = detail::computeAngleFromVertical(result.rotatedRect);
    result.score = bestScore;
    result.binaryMask = cv::Mat::zeros(image.size(), CV_8UC1);
    selectedMask.copyTo(result.binaryMask(validRoi));
    return result;
}

}  // namespace liquid

