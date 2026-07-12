# ProcessLiquidFlow C# 互操作方案

## 目标

将 `processLiquidFlowImage` 的液流检测能力暴露给 C# 调用方，通过 P/Invoke 调用原生 C++ DLL。

## 输出内容

| 输出项 | 类型 | 传递方式 |
|--------|------|---------|
| `best_angle` | float | `LiquidFlowResult` 结构体成员 |
| `best_idx1` | float | `LiquidFlowResult` 结构体成员 |
| `best_idx2` | float | `LiquidFlowResult` 结构体成员 |
| `success` | int | `LiquidFlowResult` 结构体成员 (1=成功, 0=失败) |
| `diag_log` | string | `StringBuilder` 预分配缓冲区 (out 参数) |
| `grayImage` | file | C# 指定 `savePath`，C++ 通过 `cv::imwrite` 写入磁盘 |

## 架构

```mermaid
flowchart TD
    subgraph C# Host Process
        CS[C# 调用方]
        CS -->|1-byte[] imgData| PInvoke
        CS -->|2-int width/height/channels| PInvoke
        CS -->|3-string savePath| PInvoke
        CS -->|4-StringBuilder diagLog| PInvoke
        PInvoke["[DllImport] ProcessLiquidFlow"]
        PInvoke -->|ref LiquidFlowResult| RESULT
    end

    subgraph opencv-pro-dll.dll
        EXPORT["extern C __declspec(dllexport) ProcessLiquidFlow"]
        EXPORT -->|byte[] -> cv::Mat| MAT[构造 cv::Mat 3ch BGR]
        MAT --> CALL[processLiquidFlowImageCore]
        CALL -->|cv::imwrite| DISK[保存 grayImage 到磁盘]
        CALL -->|填充| STRUCT[LiquidFlowResult]
        CALL -->|写入| SB[StringBuilder 缓冲区]
    end

    subgraph opencv-pro-test/include
        CORE["ProcessLiquidFlow.hpp"]
        OLD["processLiquidFlowImage - 不变"]
        NEW["processLiquidFlowImageCore - 新增"]
    end

    CALL --> NEW
    DISK --> FS[(文件系统)]
```

## 目录结构

```
opencv-pro/
├── opencv-pro-test/                    ← 现有 exe 项目（不动）
│   └── include/
│       └── ProcessLiquidFlow.hpp      ← 新增 processLiquidFlowImageCore()
│
├── opencv-pro-dll/                     ← 新建 DLL 项目
│   ├── ProcessLiquidFlowDLL.h          ← C 兼容导出头
│   ├── ProcessLiquidFlowDLL.cpp        ← DLL 入口实现
│   ├── opencv-pro-dll.vcxproj          ← MSBuild 项目文件
│   └── opencv-pro-dll.vcxproj.filters  ← 筛选器文件
│
├── opencv-pro.slnx                     ← 解决方案（需添加 DLL 项目）
└── plans/
    └── ProcessLiquidFlow-CSharp-Interop.md
```

## 类型定义

### C/C++ 侧 (`ProcessLiquidFlowDLL.h`)

```cpp
#pragma once

// C 兼容的结果结构体
#pragma pack(push, 1)
typedef struct {
    float best_angle;   // 最佳旋转角度 (度)
    float best_idx1;    // 谷底1的x坐标 (全局图像坐标系)
    float best_idx2;    // 谷底2的x坐标 (全局图像坐标系)
    int   success;      // 1=检测成功, 0=检测失败
} LiquidFlowResult;
#pragma pack(pop)

// 导出函数
// 返回值: 0=成功, -1=参数错误, -2=OpenCV异常, -3=未知异常
extern "C" __declspec(dllexport) int __stdcall ProcessLiquidFlow(
    const unsigned char* imgData,   // 原始 BGR 像素数据 (连续内存, 行优先)
    int                  width,      // 图像宽度
    int                  height,     // 图像高度
    int                  channels,   // 通道数 (通常为 3 = BGR)
    const char*          savePath,   // grayImage 保存路径 (完整文件路径, 含 .png/.jpg)
    LiquidFlowResult*    result,     // [out] 检测结果
    char*                diagLog,    // [out] 日志文本缓冲区
    int                  diagLogSize // 缓冲区大小 (建议 ≥ 2048)
);
```

### C# 侧声明

```csharp
using System;
using System.Runtime.InteropServices;
using System.Text;

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct LiquidFlowResult
{
    public float BestAngle;
    public float BestIdx1;
    public float BestIdx2;
    public int   Success;
}

public static class LiquidFlowNative
{
    [DllImport("opencv-pro-dll.dll", CallingConvention = CallingConvention.StdCall)]
    public static extern int ProcessLiquidFlow(
        byte[]       imgData,
        int          width,
        int          height,
        int          channels,
        string       savePath,
        ref LiquidFlowResult result,
        StringBuilder diagLog,
        int          diagLogSize
    );
}
```

## processLiquidFlowImageCore 函数设计

位于 [`ProcessLiquidFlow.hpp`](opencv-pro-test/include/ProcessLiquidFlow.hpp)，在现有 `processLiquidFlowImage` 之后新增。

### 签名

```cpp
inline LiquidFlowResult processLiquidFlowImageCore(
    const cv::Mat& image,
    const std::string& savePath,
    std::ostringstream& diag_log)
```

### 与现有 `processLiquidFlowImage` 的区别

| 方面 | processLiquidFlowImage (现有) | processLiquidFlowImageCore (新增) |
|------|------------------------------|----------------------------------|
| 输出 | 控制台 `std::cout` / `std::cerr` | 返回 `LiquidFlowResult` + 写入 `ostringstream` |
| grayImage | 仅用于可视化（画线） | 画线后 `cv::imwrite(savePath, grayImage)` 保存 |
| 异常处理 | try-catch → cerr | try-catch → 设置 result.success=0 |
| 返回值 | void | `LiquidFlowResult` |

### 伪代码

```cpp
inline LiquidFlowResult processLiquidFlowImageCore(
    const cv::Mat& image,
    const std::string& savePath,
    std::ostringstream& diag_log)
{
    LiquidFlowResult result = {0.0f, 0.0f, 0.0f, 0}; // success=0

    if (image.empty()) {
        diag_log << "FAIL: empty image\n";
        return result;
    }

    try {
        // === 完全复用现有流程 ===
        // 灰度化 → 双边滤波 → 二值化 → 连通域分析 → D1匹配 → 旋转扫描 → ...

        // 计算完成后
        result.best_angle = best_angle;
        result.best_idx1  = best_idx1;
        result.best_idx2  = best_idx2;
        result.success    = 1;

        // 保存 grayImage
        if (!savePath.empty() && !grayImage.empty()) {
            cv::imwrite(savePath, grayImage);
        }
        diag_log << "STATUS: SUCCESS\n";
    }
    catch (...) {
        diag_log << "STATUS: FAIL\n";
        result.success = 0;
    }

    return result;
}
```

## DLL 入口实现 (`ProcessLiquidFlowDLL.cpp`)

### 关键逻辑

1. **参数校验**: imgData、result、diagLog 非空；width/height > 0；channels ∈ {1,3}
2. **构造 cv::Mat**:
   ```cpp
   int cvType = (channels == 3) ? CV_8UC3 : CV_8UC1;
   cv::Mat image(height, width, cvType, (void*)imgData);
   // 注意：如果 imgData 来自 C# 的 byte[]，需要 clone() 确保生命周期
   cv::Mat imageClone = image.clone();
   ```
3. **调用 core 函数**: `LiquidFlowResult result = processLiquidFlowImageCore(imageClone, savePath, diag_log_ss);`
4. **复制日志**: `strncpy_s(diagLog, diagLogSize, diag_log_ss.str().c_str(), _TRUNCATE);`
5. **写入 result**: `*result_ptr = result;`
6. **返回值**: `return 0;`

## 调用流程 (C# 端示例)

```csharp
// 1. 从 Bitmap 获取原始像素数据
Bitmap bmp = new Bitmap(@"test.jpg");
Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly, bmp.PixelFormat);

byte[] imgData = new byte[bmpData.Stride * bmp.Height];
Marshal.Copy(bmpData.Scan0, imgData, 0, imgData.Length);
bmp.UnlockBits(bmpData);

// 2. 调用 DLL
var result = new LiquidFlowResult();
var diagLog = new StringBuilder(4096);

int ret = LiquidFlowNative.ProcessLiquidFlow(
    imgData,
    bmp.Width,
    bmp.Height,
    3,                         // BGR
    @"C:\temp\grayImage.png",  // 保存路径
    ref result,
    diagLog,
    diagLog.Capacity
);

// 3. 处理结果
if (result.Success == 1)
{
    Console.WriteLine($"angle={result.BestAngle}, idx1={result.BestIdx1}, idx2={result.BestIdx2}");
}
Console.WriteLine(diagLog.ToString());
```

## 注意事项

- `cv::Mat` 从 C# 的 `byte[]` 构造后必须 `clone()`，因为 P/Invoke 期间 `byte[]` 可能被 GC 移动
- `grayImage` 中已绘制两条绿色竖线标注谷底位置，便于 C# 端人工复核
- DLL 编译时建议使用 `/MT` (静态链接 CRT) 避免目标机器缺失 CRT DLL
- 日志建议缓冲区 ≥ 2048 字符，当前 `diag_log` 通常生成 500~800 字符
- DLL 文件需放置在 C# exe 同目录或系统 PATH 中
