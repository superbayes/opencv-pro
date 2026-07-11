#include "roi_detector.h"

#include <algorithm>

namespace liquid {
namespace {

cv::Rect makeCentralFallback(const cv::Size& imageSize) {
    const int width = static_cast<int>(imageSize.width * 0.58);
    const int height = static_cast<int>(imageSize.height * 0.72);
    const int x = (imageSize.width - width) / 2;
    const int y = (imageSize.height - height) / 2;
    return cv::Rect(x, y, width, height);
}

double contourScore(const std::vector<cv::Point>& contour, const cv::Size& imageSize) {
    const cv::Rect rect = cv::boundingRect(contour);
    const double area = static_cast<double>(rect.area());
    if (area <= 0.0) {
        return 0.0;
    }

    const cv::Point2f imageCenter(imageSize.width * 0.5F, imageSize.height * 0.5F);
    const cv::Moments moments = cv::moments(contour);
    cv::Point2f contourCenter = imageCenter;
    if (moments.m00 > 1e-6) {
        contourCenter.x = static_cast<float>(moments.m10 / moments.m00);
        contourCenter.y = static_cast<float>(moments.m01 / moments.m00);
    }

    const double distance = cv::norm(contourCenter - imageCenter);
    const double maxDistance = std::max(imageSize.width, imageSize.height) * 0.75;
    const double centerWeight = std::max(0.0, 1.0 - distance / maxDistance);
    const double aspect = rect.height > 0 ? static_cast<double>(rect.width) / rect.height : 0.0;
    const double aspectPenalty = 1.0 - std::min(1.0, std::abs(aspect - 1.0));
    return area * (0.65 + 0.35 * centerWeight) * (0.75 + 0.25 * aspectPenalty);
}

}  // namespace

ROIResult detectObservationRoi(const cv::Mat& bgrImage) {
    ROIResult result;
    result.rect = makeCentralFallback(bgrImage.size());
    result.usedFallback = true;

    if (bgrImage.empty()) {
        return result;
    }

    cv::Mat gray;
    if (bgrImage.channels() == 3) {
        cv::cvtColor(bgrImage, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = bgrImage.clone();
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0.0);

    cv::Mat thresholdMask;
    cv::threshold(blurred, thresholdMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(17, 17));
    cv::morphologyEx(thresholdMask, thresholdMask, cv::MORPH_CLOSE, morphKernel);
    cv::morphologyEx(thresholdMask, thresholdMask, cv::MORPH_OPEN, morphKernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholdMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double bestScore = 0.0;
    cv::Rect bestRect;
    for (const auto& contour : contours) {
        const cv::Rect rect = cv::boundingRect(contour);
        if (rect.area() < (bgrImage.rows * bgrImage.cols) / 20) {
            continue;
        }

        const double score = contourScore(contour, bgrImage.size());
        if (score > bestScore) {
            bestScore = score;
            bestRect = rect;
        }
    }

    if (bestScore > 0.0) {
        const int marginX = std::max(12, bestRect.width / 18);
        const int marginY = std::max(12, bestRect.height / 18);
        const cv::Rect expanded(bestRect.x - marginX,
                                bestRect.y - marginY,
                                bestRect.width + marginX * 2,
                                bestRect.height + marginY * 2);
        result.rect = expanded & cv::Rect(0, 0, bgrImage.cols, bgrImage.rows);
        result.usedFallback = false;
    }

    return result;
}

}  // namespace liquid
