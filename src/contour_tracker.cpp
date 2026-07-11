#include "contour_tracker.h"

#include <algorithm>
#include <cmath>

namespace liquid {
namespace {

struct RowDetection {
    float left = 0.0F;
    float right = 0.0F;
    float score = 0.0F;
    bool valid = false;
};

float meanInRange(const float* data, int begin, int end) {
    if (end <= begin) {
        return 0.0F;
    }

    float sum = 0.0F;
    for (int i = begin; i < end; ++i) {
        sum += data[i];
    }
    return sum / static_cast<float>(end - begin);
}

int searchPeak(const float* data, int begin, int end, float threshold) {
    begin = std::max(begin, 0);
    end = std::max(end, begin + 1);

    float bestValue = threshold;
    int bestIndex = -1;
    for (int i = begin; i < end; ++i) {
        if (data[i] > bestValue) {
            bestValue = data[i];
            bestIndex = i;
        }
    }
    return bestIndex;
}

RowDetection detectOnRow(const cv::Mat& feature,
                         const cv::Mat& edge,
                         int y,
                         int centerX,
                         int expectedHalfWidth,
                         int previousLeft,
                         int previousRight,
                         bool hasPrevious) {
    RowDetection detection;

    const cv::Mat featureRow = feature.row(y);
    const cv::Mat edgeRow = edge.row(y);
    const float* featureData = featureRow.ptr<float>();
    const float* edgeData = edgeRow.ptr<float>();

    const int cols = feature.cols;
    const int minWidth = std::max(8, expectedHalfWidth / 2);
    const int maxWidth = std::max(minWidth + 12, expectedHalfWidth * 3);
    const int windowRadius = std::max(10, expectedHalfWidth / 2);

    int expectedLeft = centerX - expectedHalfWidth;
    int expectedRight = centerX + expectedHalfWidth;
    if (hasPrevious) {
        expectedLeft = previousLeft;
        expectedRight = previousRight;
    }

    const int leftBegin = std::clamp(expectedLeft - windowRadius, 0, cols - 1);
    const int leftEnd = std::clamp(expectedLeft + windowRadius, leftBegin + 1, cols);
    const int rightBegin = std::clamp(expectedRight - windowRadius, 0, cols - 1);
    const int rightEnd = std::clamp(expectedRight + windowRadius, rightBegin + 1, cols);

    const float rowMean = meanInRange(edgeData, 0, cols);
    const float rowFeatureMean = meanInRange(featureData, 0, cols);
    const float threshold = rowMean * 1.25F + rowFeatureMean * 0.15F;

    const int leftPeak = searchPeak(edgeData, leftBegin, leftEnd, threshold);
    const int rightPeak = searchPeak(edgeData, rightBegin, rightEnd, threshold);
    if (leftPeak < 0 || rightPeak < 0) {
        return detection;
    }

    const int width = rightPeak - leftPeak;
    if (width < minWidth || width > maxWidth) {
        return detection;
    }

    const int innerBegin = std::clamp(leftPeak + 1, 0, cols);
    const int innerEnd = std::clamp(rightPeak, innerBegin, cols);
    const float innerFeature = meanInRange(featureData, innerBegin, innerEnd);
    if (innerFeature < rowFeatureMean * 0.9F) {
        return detection;
    }

    detection.left = static_cast<float>(leftPeak);
    detection.right = static_cast<float>(rightPeak);
    detection.score = (edgeData[leftPeak] + edgeData[rightPeak]) * 0.5F + innerFeature;
    detection.valid = true;
    return detection;
}

int findSeedRow(const cv::Mat& feature, const cv::Rect& scanRect) {
    int bestRow = feature.rows / 2;
    double bestScore = -1.0;
    for (int y = 0; y < feature.rows; ++y) {
        const cv::Scalar rowMean = cv::mean(feature(cv::Rect(scanRect.x, y, scanRect.width, 1)));
        if (rowMean[0] > bestScore) {
            bestScore = rowMean[0];
            bestRow = y;
        }
    }
    return bestRow;
}

void fillMissing(std::vector<float>& values, const std::vector<unsigned char>& validMask) {
    int firstValid = -1;
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        if (validMask[i]) {
            firstValid = i;
            break;
        }
    }

    if (firstValid < 0) {
        return;
    }

    int lastValid = -1;
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        if (!validMask[i]) {
            continue;
        }

        if (lastValid >= 0 && lastValid + 1 < i) {
            const float start = values[lastValid];
            const float end = values[i];
            for (int j = lastValid + 1; j < i; ++j) {
                const float t = static_cast<float>(j - lastValid) / static_cast<float>(i - lastValid);
                values[j] = start + (end - start) * t;
            }
        }
        lastValid = i;
    }

    for (int i = 0; i < firstValid; ++i) {
        values[i] = values[firstValid];
    }

    for (int i = lastValid + 1; i < static_cast<int>(values.size()); ++i) {
        values[i] = values[lastValid];
    }
}

void trackDirection(const cv::Mat& feature,
                    const cv::Mat& edge,
                    int seedRow,
                    int step,
                    int centerX,
                    int halfWidth,
                    std::vector<float>& leftValues,
                    std::vector<float>& rightValues,
                    std::vector<float>& scores,
                    std::vector<unsigned char>& validMask) {
    int previousLeft = centerX - halfWidth;
    int previousRight = centerX + halfWidth;
    bool hasPrevious = false;

    for (int y = seedRow; y >= 0 && y < feature.rows; y += step) {
        const RowDetection detection =
            detectOnRow(feature, edge, y, centerX, halfWidth, previousLeft, previousRight, hasPrevious);
        if (detection.valid) {
            leftValues[y] = detection.left;
            rightValues[y] = detection.right;
            scores[y] = detection.score;
            validMask[y] = 1;
            previousLeft = static_cast<int>(std::round(detection.left));
            previousRight = static_cast<int>(std::round(detection.right));
            hasPrevious = true;
        } else if (hasPrevious) {
            leftValues[y] = static_cast<float>(previousLeft);
            rightValues[y] = static_cast<float>(previousRight);
            scores[y] = 0.0F;
        } else {
            leftValues[y] = static_cast<float>(centerX - halfWidth);
            rightValues[y] = static_cast<float>(centerX + halfWidth);
            scores[y] = 0.0F;
        }
    }
}

}  // namespace

ContourTrack trackContours(const StreamDetectionResult& streamResult) {
    ContourTrack track;
    track.scanRect = streamResult.scanRect;

    if (streamResult.rotatedFeature.empty() || streamResult.rotatedEdge.empty()) {
        return track;
    }

    cv::Mat featureFloat;
    cv::Mat edgeFloat;
    streamResult.rotatedFeature.convertTo(featureFloat, CV_32F);
    streamResult.rotatedEdge.convertTo(edgeFloat, CV_32F);

    const int rows = featureFloat.rows;
    std::vector<float> leftValues(rows, static_cast<float>(streamResult.centerX - streamResult.halfWidth));
    std::vector<float> rightValues(rows, static_cast<float>(streamResult.centerX + streamResult.halfWidth));
    std::vector<float> scores(rows, 0.0F);
    std::vector<unsigned char> validMask(rows, 0);

    const int seedRow = findSeedRow(featureFloat, streamResult.scanRect);
    trackDirection(featureFloat,
                   edgeFloat,
                   seedRow,
                   -1,
                   streamResult.centerX,
                   streamResult.halfWidth,
                   leftValues,
                   rightValues,
                   scores,
                   validMask);
    if (seedRow + 1 < rows) {
        trackDirection(featureFloat,
                       edgeFloat,
                       seedRow + 1,
                       1,
                       streamResult.centerX,
                       streamResult.halfWidth,
                       leftValues,
                       rightValues,
                       scores,
                       validMask);
    }

    fillMissing(leftValues, validMask);
    fillMissing(rightValues, validMask);

    int validCount = 0;
    double scoreSum = 0.0;
    for (int y = 0; y < rows; ++y) {
        const float center = (leftValues[y] + rightValues[y]) * 0.5F;
        const float width = std::max(0.0F, rightValues[y] - leftValues[y]);
        track.left.emplace_back(leftValues[y], static_cast<float>(y));
        track.right.emplace_back(rightValues[y], static_cast<float>(y));
        track.center.emplace_back(center, static_cast<float>(y));
        track.widths.push_back(width);
        track.validMask.push_back(validMask[y]);
        if (validMask[y]) {
            ++validCount;
            scoreSum += scores[y];
        }
    }

    track.qualityScore = rows > 0 ? static_cast<double>(validCount) / rows : 0.0;
    if (validCount > 0) {
        track.qualityScore *= 0.7 + 0.3 * std::min(1.0, scoreSum / (validCount * 255.0));
    }

    return track;
}

}  // namespace liquid
