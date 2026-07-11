#include "debug_visualizer.h"

#include <fstream>
#include <iomanip>

namespace liquid {
namespace {

cv::Mat toBgr(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    if (image.channels() == 3) {
        return image.clone();
    }

    cv::Mat colored;
    cv::cvtColor(image, colored, cv::COLOR_GRAY2BGR);
    return colored;
}

void drawTrackOverlay(cv::Mat& canvas, const ContourTrack& track) {
    for (size_t i = 0; i < track.left.size(); ++i) {
        const cv::Scalar pointColor = track.validMask[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        cv::circle(canvas, track.left[i], 1, cv::Scalar(255, 0, 0), -1);
        cv::circle(canvas, track.right[i], 1, cv::Scalar(0, 0, 255), -1);
        cv::circle(canvas, track.center[i], 1, pointColor, -1);
    }
}

double meanWidth(const ContourTrack& track) {
    if (track.widths.empty()) {
        return 0.0;
    }

    double total = 0.0;
    for (float width : track.widths) {
        total += width;
    }
    return total / track.widths.size();
}

}  // namespace

void saveDebugImages(const std::filesystem::path& outputDir,
                     const cv::Mat& sourceImage,
                     const ROIResult& roiResult,
                     const PreprocessResult& preprocessResult,
                     const StreamDetectionResult& streamResult,
                     const ContourTrack& track) {
    std::filesystem::create_directories(outputDir);

    cv::Mat roiOverlay = sourceImage.clone();
    cv::rectangle(roiOverlay, roiResult.rect, cv::Scalar(0, 255, 255), 2);
    cv::imwrite((outputDir / "01_roi_overlay.png").string(), roiOverlay);

    if (!preprocessResult.gray.empty()) {
        cv::imwrite((outputDir / "02_gray.png").string(), preprocessResult.gray);
    }
    if (!preprocessResult.normalizedAbs.empty()) {
        cv::imwrite((outputDir / "03_normalized_abs.png").string(), preprocessResult.normalizedAbs);
    }
    if (!streamResult.rotatedFeature.empty()) {
        cv::imwrite((outputDir / "04_feature.png").string(), streamResult.rotatedFeature);
    }
    if (!streamResult.rotatedEdge.empty()) {
        cv::imwrite((outputDir / "05_edge.png").string(), streamResult.rotatedEdge);
    }

    if (!streamResult.rotatedGray.empty()) {
        cv::Mat contourOverlay = toBgr(streamResult.rotatedGray);
        cv::rectangle(contourOverlay, streamResult.scanRect, cv::Scalar(255, 255, 0), 1);
        drawTrackOverlay(contourOverlay, track);
        cv::imwrite((outputDir / "06_contours.png").string(), contourOverlay);
    }
}

void saveSummary(const std::filesystem::path& outputDir,
                 const std::string& imageName,
                 const ROIResult& roiResult,
                 const StreamDetectionResult& streamResult,
                 const ContourTrack& track) {
    std::filesystem::create_directories(outputDir);

    std::ofstream summary(outputDir / "summary.txt", std::ios::out | std::ios::trunc);
    summary << "image=" << imageName << '\n';
    summary << "roi=" << roiResult.rect.x << "," << roiResult.rect.y << ","
            << roiResult.rect.width << "," << roiResult.rect.height << '\n';
    summary << "used_fallback_roi=" << (roiResult.usedFallback ? "true" : "false") << '\n';
    summary << "stream_is_dark=" << (streamResult.streamIsDark ? "true" : "false") << '\n';
    summary << "angle_degrees=" << std::fixed << std::setprecision(3) << streamResult.angleDegrees << '\n';
    summary << "center_x=" << streamResult.centerX << '\n';
    summary << "half_width=" << streamResult.halfWidth << '\n';
    summary << "quality_score=" << std::fixed << std::setprecision(3) << track.qualityScore << '\n';
    summary << "mean_width=" << std::fixed << std::setprecision(3) << meanWidth(track) << '\n';
}

}  // namespace liquid
