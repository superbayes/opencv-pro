#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "contour_tracker.h"
#include "curve_fit.h"
#include "debug_visualizer.h"
#include "preprocess.h"
#include "roi_detector.h"
#include "stream_detector.h"

namespace fs = std::filesystem;

namespace {

std::vector<fs::path> collectImagePaths(const fs::path& imageDir) {
    std::vector<fs::path> paths;
    if (!fs::exists(imageDir)) {
        return paths;
    }

    for (const auto& entry : fs::directory_iterator(imageDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string extension = entry.path().extension().string();
        if (extension == ".jpg" || extension == ".png" || extension == ".bmp") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

bool processImage(const fs::path& imagePath, const fs::path& outputRoot) {
    const cv::Mat source = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (source.empty()) {
        std::cerr << "Failed to read image: " << imagePath << '\n';
        return false;
    }

    // Detect ROI
    const liquid::ROIResult roiResult = liquid::detectObservationRoi(source);// 检测ROI
    const cv::Mat roiImage = source(roiResult.rect).clone();

    //
    const liquid::PreprocessResult preprocessResult = liquid::preprocessRoi(roiImage);
    const liquid::StreamDetectionResult streamResult = liquid::detectStream(preprocessResult);// 检测流

    liquid::ContourTrack contourTrack = liquid::trackContours(streamResult);// 跟踪轮廓
    liquid::smoothContours(contourTrack);// 平滑轮廓    

    const fs::path imageOutputDir = outputRoot / imagePath.stem();
    liquid::saveDebugImages(imageOutputDir, source, roiResult, preprocessResult, streamResult, contourTrack);
    liquid::saveSummary(imageOutputDir, imagePath.filename().string(), roiResult, streamResult, contourTrack);

    std::cout << "Processed " << imagePath.filename().string() << " | quality="
              << contourTrack.qualityScore << " | center=" << streamResult.centerX
              << " | halfWidth=" << streamResult.halfWidth << '\n';
    return true;
}

}  // namespace

int main() {
    const fs::path workspace = fs::current_path();
    const fs::path imageDir = workspace / "image";
    const fs::path outputRoot = workspace / "output";

    const std::vector<fs::path> images = collectImagePaths(imageDir);
    if (images.empty()) {
        std::cerr << "No images found in: " << imageDir << '\n';
        return 1;
    }

    fs::create_directories(outputRoot);

    int successCount = 0;
    for (const auto& imagePath : images) {
        if (processImage(imagePath, outputRoot)) {
            ++successCount;
        }
    }

    std::cout << "Completed " << successCount << " / " << images.size() << " images." << '\n';
    return successCount == static_cast<int>(images.size()) ? 0 : 2;
}

