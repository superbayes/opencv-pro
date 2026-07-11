#include "stream_detector.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace liquid {
namespace {

struct StreamCandidate {
    double score = -1.0;
    int centerX = 0;
    int halfWidth = 12;
};

std::vector<float> columnEnergy(const cv::Mat& image) {
    cv::Mat reduced;
    cv::reduce(image, reduced, 0, cv::REDUCE_SUM, CV_32F);
    std::vector<float> values(reduced.cols);
    for (int x = 0; x < reduced.cols; ++x) {
        values[x] = reduced.at<float>(0, x);
    }
    return values;
}

std::vector<float> smooth1D(const std::vector<float>& values, int radius) {
    std::vector<float> smoothed(values.size(), 0.0F);
    if (values.empty()) {
        return smoothed;
    }

    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        float weightedSum = 0.0F;
        float weightTotal = 0.0F;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int idx = std::clamp(i + offset, 0, static_cast<int>(values.size()) - 1);
            const float weight = static_cast<float>(radius + 1 - std::abs(offset));
            weightedSum += values[idx] * weight;
            weightTotal += weight;
        }
        smoothed[i] = weightTotal > 0.0F ? weightedSum / weightTotal : values[i];
    }

    return smoothed;
}

int argmaxInRange(const std::vector<float>& values, int begin, int end) {
    begin = std::clamp(begin, 0, static_cast<int>(values.size()) - 1);
    end = std::clamp(end, begin + 1, static_cast<int>(values.size()));

    int bestIndex = begin;
    float bestValue = values[begin];
    for (int i = begin + 1; i < end; ++i) {
        if (values[i] > bestValue) {
            bestValue = values[i];
            bestIndex = i;
        }
    }
    return bestIndex;
}

double estimateTiltFromResponse(const cv::Mat& response, int centerX, int halfWidth) {
    const int xMin = std::max(0, centerX - halfWidth * 3);
    const int xMax = std::min(response.cols, centerX + halfWidth * 3);
    if (xMax - xMin < 10) {
        return 0.0;
    }

    cv::Rect band(xMin, 0, xMax - xMin, response.rows);
    cv::Mat cropped = response(band);

    std::vector<cv::Point2f> points;
    points.reserve(cropped.rows);
    for (int y = 0; y < cropped.rows; ++y) {
        const cv::Mat row = cropped.row(y);
        cv::Scalar rowMean;
        cv::Scalar rowStd;
        cv::meanStdDev(row, rowMean, rowStd);
        double maxValue = 0.0;
        cv::Point maxPoint;
        cv::minMaxLoc(row, nullptr, &maxValue, nullptr, &maxPoint);
        if (maxValue >= rowMean[0] + rowStd[0] * 0.8) {
            points.emplace_back(static_cast<float>(maxPoint.x + band.x), static_cast<float>(y));
        }
    }

    if (points.size() < 30) {
        return 0.0;
    }

    cv::Vec4f line;
    cv::fitLine(points, line, cv::DIST_L2, 0.0, 0.01, 0.01);
    const double angleFromVertical = std::atan2(line[0], line[1]) * 180.0 / CV_PI;
    return std::clamp(angleFromVertical, -6.0, 6.0);
}

cv::Mat rotateKeepSize(const cv::Mat& image, double angleDegrees) {
    if (std::abs(angleDegrees) < 0.01) {
        return image.clone();
    }

    const cv::Point2f center((image.cols - 1) * 0.5F, (image.rows - 1) * 0.5F);
    const cv::Mat rotation = cv::getRotationMatrix2D(center, -angleDegrees, 1.0);
    cv::Mat rotated;
    cv::warpAffine(image, rotated, rotation, image.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return rotated;
}

StreamCandidate findCandidate(const std::vector<float>& featureEnergy, const std::vector<float>& edgeEnergy) {
    StreamCandidate best;
    if (featureEnergy.empty() || edgeEnergy.empty()) {
        return best;
    }

    const int cols = static_cast<int>(featureEnergy.size());
    const int begin = std::max(20, cols / 8);
    const int end = std::min(cols - 20, cols * 7 / 8);

    for (int center = begin; center < end; ++center) {
        for (int halfWidth = 4; halfWidth <= 26; ++halfWidth) {
            const int left = center - halfWidth;
            const int right = center + halfWidth;
            if (left < 1 || right >= cols - 1) {
                continue;
            }

            const double interiorScore = featureEnergy[center];
            const double boundaryScore = edgeEnergy[left] + edgeEnergy[right];
            const double outsidePenalty = (featureEnergy[std::max(0, left - halfWidth)] +
                                           featureEnergy[std::min(cols - 1, right + halfWidth)]) *
                                          0.25;
            const double score = interiorScore * 1.5 + boundaryScore - outsidePenalty;
            if (score > best.score) {
                best.score = score;
                best.centerX = center;
                best.halfWidth = halfWidth;
            }
        }
    }

    if (best.score < 0.0) {
        best.centerX = cols / 2;
        best.halfWidth = 12;
    }
    return best;
}

}  // namespace

StreamDetectionResult detectStream(const PreprocessResult& preprocessResult) {
    StreamDetectionResult result;
    if (preprocessResult.gray.empty()) {
        return result;
    }

    cv::Mat darkSmoothed;
    cv::Mat brightSmoothed;
    cv::GaussianBlur(preprocessResult.darkFeature, darkSmoothed, cv::Size(1, 41), 0.0);
    cv::GaussianBlur(preprocessResult.brightFeature, brightSmoothed, cv::Size(1, 41), 0.0);

    cv::Mat darkGrad16;
    cv::Mat brightGrad16;
    cv::Scharr(preprocessResult.darkFeature, darkGrad16, CV_16S, 1, 0);
    cv::Scharr(preprocessResult.brightFeature, brightGrad16, CV_16S, 1, 0);

    cv::Mat darkEdge;
    cv::Mat brightEdge;
    cv::convertScaleAbs(darkGrad16, darkEdge);
    cv::convertScaleAbs(brightGrad16, brightEdge);

    const StreamCandidate darkCandidate =
        findCandidate(smooth1D(columnEnergy(darkSmoothed), 6), smooth1D(columnEnergy(darkEdge), 4));
    const StreamCandidate brightCandidate =
        findCandidate(smooth1D(columnEnergy(brightSmoothed), 6), smooth1D(columnEnergy(brightEdge), 4));

    result.streamIsDark = darkCandidate.score >= brightCandidate.score;
    const cv::Mat& feature = result.streamIsDark ? preprocessResult.darkFeature : preprocessResult.brightFeature;
    cv::Mat absGrad = result.streamIsDark ? darkEdge : brightEdge;

    const StreamCandidate candidate = result.streamIsDark ? darkCandidate : brightCandidate;
    result.centerX = candidate.centerX;
    result.halfWidth = candidate.halfWidth;

    result.angleDegrees = estimateTiltFromResponse(feature, result.centerX, result.halfWidth);
    result.rotatedGray = rotateKeepSize(preprocessResult.gray, result.angleDegrees);
    result.rotatedFeature = rotateKeepSize(feature, result.angleDegrees);
    result.rotatedEdge = rotateKeepSize(absGrad, result.angleDegrees);

    cv::Mat rotatedFeatureSmoothed;
    cv::GaussianBlur(result.rotatedFeature, rotatedFeatureSmoothed, cv::Size(1, 41), 0.0);
    const StreamCandidate rotatedCandidate =
         findCandidate(smooth1D(columnEnergy(rotatedFeatureSmoothed), 6), smooth1D(columnEnergy(result.rotatedEdge), 4));
    result.centerX = rotatedCandidate.centerX;
    result.halfWidth = rotatedCandidate.halfWidth;

    const int scanMargin = std::max(16, result.halfWidth * 3);
    const int x = std::max(0, result.centerX - scanMargin);
    const int width = std::min(result.rotatedFeature.cols - x, scanMargin * 2);
    result.scanRect = cv::Rect(x, 0, width, result.rotatedFeature.rows);

    return result;
}

}  // namespace liquid
