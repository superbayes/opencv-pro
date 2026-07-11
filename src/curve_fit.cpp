#include "curve_fit.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace liquid {
namespace {

float medianOfWindow(const std::vector<float>& values, int center, int radius) {
    std::vector<float> window;
    window.reserve(radius * 2 + 1);
    for (int offset = -radius; offset <= radius; ++offset) {
        const int index = std::clamp(center + offset, 0, static_cast<int>(values.size()) - 1);
        window.push_back(values[index]);
    }
    const auto middle = window.begin() + static_cast<std::ptrdiff_t>(window.size() / 2);
    std::nth_element(window.begin(), middle, window.end());
    return *middle;
}

std::vector<float> smoothSeries(const std::vector<float>& values, int radius) {
    std::vector<float> medians(values.size(), 0.0F);
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        medians[i] = medianOfWindow(values, i, radius);
    }

    std::vector<float> smoothed(values.size(), 0.0F);
    for (int i = 0; i < static_cast<int>(medians.size()); ++i) {
        float weightedSum = 0.0F;
        float weightTotal = 0.0F;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int index = std::clamp(i + offset, 0, static_cast<int>(medians.size()) - 1);
            const float sigma = std::max(1.0F, radius * 0.7F);
            const float weight = std::exp(-(offset * offset) / (2.0F * sigma * sigma));
            weightedSum += medians[index] * weight;
            weightTotal += weight;
        }
        smoothed[i] = weightTotal > 0.0F ? weightedSum / weightTotal : medians[i];
    }
    return smoothed;
}

void suppressOutliers(std::vector<float>& values, int radius, float tolerance) {
    const std::vector<float> baseline = smoothSeries(values, radius);
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        if (std::abs(values[i] - baseline[i]) > tolerance) {
            values[i] = baseline[i];
        }
    }
}

}  // namespace

void smoothContours(ContourTrack& track, int windowRadius) {
    if (track.left.empty() || track.right.empty()) {
        return;
    }

    std::vector<float> leftValues;
    std::vector<float> rightValues;
    leftValues.reserve(track.left.size());
    rightValues.reserve(track.right.size());
    for (size_t i = 0; i < track.left.size(); ++i) {
        leftValues.push_back(track.left[i].x);
        rightValues.push_back(track.right[i].x);
    }

    suppressOutliers(leftValues, windowRadius, 10.0F);
    suppressOutliers(rightValues, windowRadius, 10.0F);
    leftValues = smoothSeries(leftValues, windowRadius);
    rightValues = smoothSeries(rightValues, windowRadius);

    track.center.clear();
    track.widths.clear();
    for (size_t i = 0; i < track.left.size(); ++i) {
        track.left[i].x = leftValues[i];
        track.right[i].x = rightValues[i];
        const float centerX = (leftValues[i] + rightValues[i]) * 0.5F;
        const float width = std::max(0.0F, rightValues[i] - leftValues[i]);
        track.center.emplace_back(centerX, track.left[i].y);
        track.widths.push_back(width);
    }
}

}  // namespace liquid
