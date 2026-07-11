#include "preprocess.h"

namespace liquid {
namespace {

int makeOdd(int value) {
    return value % 2 == 0 ? value + 1 : value;
}

cv::Mat normalizeTo8U(const cv::Mat& input) {
    cv::Mat normalized;
    cv::normalize(input, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    return normalized;
}

}  // namespace

PreprocessResult preprocessRoi(const cv::Mat& roiBgrImage, const PreprocessParams& params) {
    PreprocessResult result;
    if (roiBgrImage.empty()) {
        return result;
    }

    if (roiBgrImage.channels() == 3) {
        cv::cvtColor(roiBgrImage, result.gray, cv::COLOR_BGR2GRAY);
    } else {
        result.gray = roiBgrImage.clone();
    }

    const int blurKernel = makeOdd(std::max(3, params.blurKernel));
    cv::Mat blurred;
    cv::GaussianBlur(result.gray, blurred, cv::Size(blurKernel, blurKernel), 0.0);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(params.claheClipLimit, params.claheTileGrid);
    clahe->apply(blurred, result.enhanced);

    const int backgroundKernel = makeOdd(std::max(31, params.backgroundKernel));
    cv::GaussianBlur(result.enhanced, result.background, cv::Size(backgroundKernel, backgroundKernel), 0.0);

    cv::Mat enhanced16;
    cv::Mat background16;
    result.enhanced.convertTo(enhanced16, CV_16S);
    result.background.convertTo(background16, CV_16S);

    cv::Mat residual16 = enhanced16 - background16;
    result.normalizedAbs = normalizeTo8U(cv::abs(residual16));
    result.darkFeature = normalizeTo8U(background16 - enhanced16);
    result.brightFeature = normalizeTo8U(enhanced16 - background16);

    return result;
}

}  // namespace liquid
