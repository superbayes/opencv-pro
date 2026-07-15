# OpenCV Pro - 液流检测系统

基于 OpenCV 的工业自动化液流检测系统，提供高精度的液流角度识别和谷底定位功能。

## 项目概述

本项目是一个用于工业自动化检测的液流识别系统，通过图像处理算法自动检测液流的最佳旋转角度和两个谷底位置。系统采用 C++ 实现核心算法，通过 DLL 导出接口供 C# 调用，适用于管道液流检测等工业场景。

## 核心功能

- **液流角度检测**：自动计算液流的最佳旋转角度
- **双谷底定位**：精确定位液流图像中的两个谷底位置
- **高阶轮廓特征提取**：提供 9 种轮廓特征提取算法
  - 傅里叶描述子 (Fourier Descriptors)
  - 椭圆傅里叶描述子 (Elliptical Fourier Descriptors)
  - Zernike 矩 (Zernike Moments)
  - 质心距离签名 (Centroid Distance Signature)
  - 复数坐标签名 (Complex Coordinate Signature)
  - 切向角签名 (Tangent Angle Signature)
  - D1/D2 距离分布 (D1/D2 Distribution)
  - 曲率轮廓 (Curvature Profile)
- **图像预处理**：双边滤波、二值化、连通域分析
- **诊断日志**：详细的处理过程日志输出

## 项目结构

```
opencv-pro/
├── opencv-pro-develop/           # C++ 核心算法开发项目
│   ├── include/                  # 头文件
│   │   ├── ProcessLiquidFlow.hpp    # 液流处理主函数
│   │   ├── ContourUtils.hpp         # 轮廓工具函数
│   │   ├── ContourHighOrderFeatures.hpp  # 高阶轮廓特征
│   │   └── LiquidFlowResult.h       # 结果结构体定义
│   └── src/
│       └── main.cpp               # 主程序入口
│
├── opencv-pro-dll/               # C++ DLL 导出项目
│   ├── ProcessLiquidFlowDLL.h    # DLL 导出接口
│   └── ProcessLiquidFlowDLL.cpp  # DLL 实现
│
├── opencv-pro-csharp-test/       # C# 测试项目
│   ├── Program.cs                # 测试程序主入口
│   ├── LiquidFlowNative.cs       # P/Invoke 封装
│   └── opencv-pro-csharp-test.csproj
│
├── docs/                         # 文档目录
├── plans/                        # 计划文档
├── opencv-pro.slnx              # Visual Studio 解决方案
└── README.md                     # 本文件
```

## 技术栈

### 开发环境
- **IDE**: Visual Studio 2022
- **C++ 标准**: C++20
- **平台**: Windows x64
- **工具链**: MSVC v145

### 核心依赖
- **OpenCV**: 图像处理核心库
- **.NET 8.0**: C# 运行时
- **System.Drawing.Common**: C# 图像处理

## 快速开始

### 前置要求

1. 安装 Visual Studio 2022
2. 安装 OpenCV（确保配置正确）
3. 安装 .NET 8.0 SDK

### 构建步骤

#### 1. 构建 C++ DLL 项目

```bash
# Debug 配置
msbuild opencv-pro-dll\opencv-pro-dll.vcxproj /p:Configuration=Debug /p:Platform=x64

# Release 配置
msbuild opencv-pro-dll\opencv-pro-dll.vcxproj /p:Configuration=Release /p:Platform=x64
```

#### 2. 复制 DLL 到 C# 项目

```bash
# Debug 配置
copy x64\Debug\S4-OpticalPathRecognitionALgorithm.dll opencv-pro-csharp-test\bin\Debug\net8.0\opencv-pro-dll.dll

# Release 配置
copy x64\Release\S4-OpticalPathRecognitionALgorithm.dll opencv-pro-csharp-test\bin\Release\net8.0\opencv-pro-dll.dll
```

#### 3. 运行 C# 测试程序

```bash
cd opencv-pro-csharp-test
dotnet run
```

## API 接口

### DLL 导出函数

```cpp
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
```

### 返回值

- `0`: 成功
- `-1`: 参数错误
- `-2`: OpenCV 异常
- `-3`: 未知异常

### 结果结构体

```cpp
#pragma pack(push, 1)
typedef struct {
    float best_angle;   // 最佳旋转角度 (单位: 度)
    float best_idx1;    // 谷底1 在全局图像中的 x 坐标
    float best_idx2;    // 谷底2 在全局图像中的 x 坐标
    int   success;      // 1 = 检测成功, 0 = 检测失败
} LiquidFlowResult;
#pragma pack(pop)
```

## C# 使用示例

```csharp
// 调用液流检测
int ret = LiquidFlowDetector.DetectFromFile(
    imagePath,
    savePath,
    out LiquidFlowResult result,
    out string diagLog
);

// 输出检测结果
if (result.Success == 1)
{
    Console.WriteLine($"最佳旋转角度: {result.BestAngle:F2} 度");
    Console.WriteLine($"谷底1 X坐标: {result.BestIdx1:F1}");
    Console.WriteLine($"谷底2 X坐标: {result.BestIdx2:F1}");
}
```

## 算法原理

### 液流检测流程

1. **图像预处理**
   - 灰度化
   - 双边滤波去噪
   - 自适应二值化

2. **轮廓分析**
   - 连通域提取
   - 轮廓筛选
   - 高阶特征匹配

3. **角度优化**
   - 多角度旋转扫描
   - 投影分析
   - 双谷底检测

4. **结果输出**
   - 最佳角度
   - 谷底坐标
   - 诊断日志

### 双谷底检测算法

在投影向量中寻找两个局部极小值，约束条件：
- 搜索范围：中间 50% 列区域
- 谷底间距：10-60 像素
- 返回：按 x 坐标升序排列的两个谷底

## 配置说明

### DLL 路径配置

C# 项目期望 DLL 位于以下位置之一：

1. **项目输出目录**: 与可执行文件同级目录
   - `bin/Debug/net8.0/opencv-pro-dll.dll`
   - `bin/Release/net8.0/opencv-pro-dll.dll`

2. **解决方案输出目录**: 
   - `$(SolutionDir)x64\Debug\opencv-pro-dll.dll`
   - `$(SolutionDir)x64\Release\opencv-pro-dll.dll`

可在 [`LiquidFlowNative.cs`](opencv-pro-csharp-test/LiquidFlowNative.cs:36) 中修改 `DllImport` 属性自定义路径。

## 文档

- [C# 测试项目配置说明](opencv-pro-csharp-test/README.md)
- [高阶轮廓特征说明](opencv-pro-develop/include/ContourHighOrderFeatures.md)
- [C# 互操作计划](plans/ProcessLiquidFlow-CSharp-Interop.md)

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题或建议，请通过 Issue 联系。
