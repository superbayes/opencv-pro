#pragma once

// LiquidFlowResult 定义见: opencv-pro-develop/include/LiquidFlowResult.h
//   成员: float best_angle  — 最佳旋转角度 (度)
//         float best_idx1   — 谷底1 x 坐标 (全局坐标系)
//         float best_idx2   — 谷底2 x 坐标 (全局坐标系)
//         int   success     — 1=成功, 0=失败
#include "LiquidFlowResult.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---- 导出函数 ----
// 返回值: 0=成功, -1=参数错误, -2=OpenCV 异常, -3=未知异常
// 调用约定: __stdcall (与 C# P/Invoke 的 CallingConvention.StdCall 对应)
__declspec(dllexport) int __stdcall ProcessLiquidFlow(
    const unsigned char* imgData,   // 原始像素数据 (BGR 8-bit, 行优先, 连续)
    int                  width,      // 图像宽度
    int                  height,     // 图像高度
    int                  channels,   // 通道数 (3=BGR, 1=Gray)
    const char*          savePath,   // grayImage 保存路径 (完整文件路径, UTF-8)
    LiquidFlowResult*    result,     // [out] 检测结果
    char*                diagLog,    // [out] 诊断日志缓冲区
    int                  diagLogSize // 缓冲区大小 (建议 ≥ 20480)
);

#ifdef __cplusplus
}
#endif
