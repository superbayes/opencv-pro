#include "ProcessLiquidFlowDLL.h"

#include <sstream>
#include <cstring>

#include <opencv2/opencv.hpp>

// 包含核心算法（header-only inline 函数）
#include "../opencv-pro-develop/include/ProcessLiquidFlow.hpp"

// ============================================================================
// ProcessLiquidFlow - DLL 导出入口
// ============================================================================
__declspec(dllexport) int __stdcall ProcessLiquidFlow(
    const unsigned char* imgData,
    int                  width,
    int                  height,
    int                  channels,
    const char*          savePath,
    LiquidFlowResult*    result,
    char*                diagLog,
    int                  diagLogSize)
{
    // ---- 1. 参数校验 ----
    if (!imgData || !result || !diagLog || diagLogSize <= 0)
        return -1;
    if (width <= 0 || height <= 0)
        return -1;
    if (channels != 1 && channels != 3)
        return -1;

    // 初始化输出
    memset(result, 0, sizeof(LiquidFlowResult));
    diagLog[0] = '\0';

    // ---- 2. byte[] → cv::Mat (必须 clone, 因 P/Invoke 期间 byte[] 可能被 GC 移动) ----
    cv::Mat image;
    try {
        int cvType = (channels == 3) ? CV_8UC3 : CV_8UC1;
        cv::Mat src(height, width, cvType, (void*)imgData);
        image = src.clone();
    }
    catch (...) {
        strncpy_s(diagLog, diagLogSize, "FAIL: cv::Mat construction failed\n", _TRUNCATE);
        return -1;
    }

    // ---- 3. 调用核心算法 ----
    std::ostringstream diag_stream;
    LiquidFlowResult coreResult = processLiquidFlowImageCore(
        image,
        (savePath ? std::string(savePath) : std::string()),
        diag_stream
    );

    // ---- 4. 复制输出 ----
    *result = coreResult;

    std::string logStr = diag_stream.str();
    strncpy_s(diagLog, diagLogSize, logStr.c_str(), _TRUNCATE);

    return (coreResult.success == 1) ? 0 : -2;
}
