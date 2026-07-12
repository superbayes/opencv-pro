#pragma once

// LiquidFlowResult: C/C++/C# 互操作兼容的结果结构体
// 用于 P/Invoke 封送，因此使用 #pragma pack(1) 确保跨平台对齐一致
#pragma pack(push, 1)
typedef struct {
    float best_angle;   // 最佳旋转角度 (单位: 度)
    float best_idx1;    // 谷底1 在全局图像中的 x 坐标
    float best_idx2;    // 谷底2 在全局图像中的 x 坐标
    int   success;      // 1 = 检测成功, 0 = 检测失败
} LiquidFlowResult;
#pragma pack(pop)
