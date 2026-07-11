#pragma once

#include <filesystem>
#include <string>

#include <opencv2/opencv.hpp>

#include "contour_tracker.h"
#include "preprocess.h"
#include "roi_detector.h"
#include "stream_detector.h"

namespace liquid {

void saveDebugImages(const std::filesystem::path& outputDir,
                     const cv::Mat& sourceImage,
                     const ROIResult& roiResult,
                     const PreprocessResult& preprocessResult,
                     const StreamDetectionResult& streamResult,
                     const ContourTrack& track);

void saveSummary(const std::filesystem::path& outputDir,
                 const std::string& imageName,
                 const ROIResult& roiResult,
                 const StreamDetectionResult& streamResult,
                 const ContourTrack& track);

}  // namespace liquid
