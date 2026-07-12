# C# 测试项目配置说明

## 项目概述

`opencv-pro-csharp-test` 是一个 C# 控制台应用程序，用于测试 `opencv-pro-dll` 生成的 DLL 中的 `ProcessLiquidFlow` 函数。

## 项目结构

```
opencv-pro-csharp-test/
├── opencv-pro-csharp-test.csproj  # 项目文件
├── Program.cs                       # 主测试程序
└── LiquidFlowNative.cs              # P/Invoke 封装类
```

## 依赖项

### NuGet 包
- `System.Drawing.Common` v8.0.0 - 用于图像处理

### DLL 依赖
- `opencv-pro-dll.dll` - 由 `opencv-pro-dll` 项目生成的 C++ DLL

## 构建配置

### 平台
- **目标平台**: x64
- **目标框架**: .NET 8.0

### 配置
- Debug / Release

## DLL 路径配置

### 输出路径
C# 项目期望 DLL 位于以下位置之一：

1. **项目输出目录**: 与可执行文件同级目录
   - `bin/Debug/net8.0/opencv-pro-dll.dll`
   - `bin/Release/net8.0/opencv-pro-dll.dll`

2. **解决方案输出目录**: 
   - `$(SolutionDir)x64\Debug\opencv-pro-dll.dll`
   - `$(SolutionDir)x64\Release\opencv-pro-dll.dll`

### 手动配置 DLL 路径

如果 DLL 位于其他位置，可以在 [`LiquidFlowNative.cs`](opencv-pro-csharp-test/LiquidFlowNative.cs:36) 中修改 `DllImport` 属性：

```csharp
// 方式1: 使用完整路径
[DllImport(@"E:\01wk\vs2026\opencv-pro\x64\Debug\opencv-pro-dll.dll", CallingConvention = CallingConvention.StdCall)]

// 方式2: 使用相对路径
[DllImport(@"..\..\..\x64\Debug\opencv-pro-dll.dll", CallingConvention = CallingConvention.StdCall)]

// 方式3: 仅使用文件名（DLL 必须在 PATH 或应用程序目录中）
[DllImport("opencv-pro-dll.dll", CallingConvention = CallingConvention.StdCall)]
```

## 构建步骤

### 1. 构建 C++ DLL 项目

在 Visual Studio 中：

```bash
# 或使用命令行
msbuild opencv-pro-dll\opencv-pro-dll.vcxproj /p:Configuration=Debug /p:Platform=x64
```

### 2. 复制 DLL 到 C# 项目输出目录

```bash
# Debug 配置
copy x64\Debug\opencv-pro-dll.dll opencv-pro-csharp-test\bin\Debug\net8.0\

# Release 配置
copy x64\Release\opencv-pro-dll.dll opencv-pro-csharp-test\bin\Release\net8.0\
```

### 3. 构建 C# 测试项目

```bash
cd opencv-pro-csharp-test
dotnet build
```

## 运行测试

### 命令行运行

```bash
cd opencv-pro-csharp-test
dotnet run
```

### Visual Studio 中运行

1. 在 Visual Studio 中打开 `opencv-pro.slnx`
2. 将 `opencv-pro-csharp-test` 设置为启动项目
3. 按 F5 运行

### 输入参数

程序运行时会提示输入：

1. **测试图像路径**: 默认为 `../../image/test.jpg`
2. **灰度图像保存路径**: 默认为 `grayImage.png`

可以直接按 Enter 使用默认值。

## 输出说明

### 成功输出示例

```
========================================
  OpenCV Pro - 液流检测测试程序
========================================

请输入测试图像路径 (默认: .../image/test.jpg):

请输入灰度图像保存路径 (默认: .../grayImage.png):

开始处理...

返回码: 0

--- 诊断日志 ---
INFO: Image loaded: 640x480
INFO: Processing...
...

=== 检测成功 ===
  最佳旋转角度: 12.34 度
  谷底1 X坐标: 123.5
  谷底2 X坐标: 456.7
  灰度图像已保存至: .../grayImage.png
```

### 失败输出示例

```
返回码: -1

--- 诊断日志 ---
FAIL: Bitmap load error - Could not find file '...'

=== 检测失败 ===
  Success = 0
```

## 返回码说明

| 返回码 | 说明 |
|--------|------|
| 0 | 成功 |
| -1 | 参数错误 |
| -2 | OpenCV 异常 |
| -3 | 未知异常 |

## API 参考

### LiquidFlowNative.ProcessLiquidFlow

```csharp
[DllImport("opencv-pro-dll.dll", CallingConvention = CallingConvention.StdCall)]
public static extern int ProcessLiquidFlow(
    byte[]       imgData,      // 原始 BGR 像素数据
    int          width,        // 图像宽度
    int          height,       // 图像高度
    int          channels,     // 通道数 (3=BGR, 1=Gray)
    string       savePath,     // grayImage 保存路径
    ref LiquidFlowResult result, // [out] 检测结果
    StringBuilder diagLog,    // [out] 诊断日志缓冲区
    int          diagLogSize   // 缓冲区大小
);
```

### LiquidFlowDetector.DetectFromFile

```csharp
public static int DetectFromFile(
    string imagePath,          // 输入图像文件路径
    string savePath,           // grayImage 保存路径
    out LiquidFlowResult result, // [out] 检测结果
    out string diagLog         // [out] 诊断日志
);
```

## 故障排除

### DLL 加载失败

**错误**: `Unable to load DLL 'opencv-pro-dll.dll'`

**解决方案**:
1. 确保 DLL 已构建
2. 检查 DLL 是否在正确的输出目录
3. 确保平台配置匹配（x64）
4. 检查 DLL 的依赖项（如 OpenCV DLL）是否可用

### BadImageFormatException

**错误**: `An attempt was made to load a program with an incorrect format`

**解决方案**:
- 确保 C# 项目和 DLL 项目都配置为 x64 平台

### 图像加载失败

**错误**: `Bitmap load error`

**解决方案**:
- 检查图像路径是否正确
- 确保图像格式受支持（BMP, JPG, PNG 等）

## 扩展功能

### 添加批量测试

```csharp
// 在 Program.cs 中添加
string[] testImages = Directory.GetFiles("test_images", "*.jpg");
foreach (var imgPath in testImages)
{
    // 执行检测
}
```

### 添加性能测试

```csharp
var stopwatch = Stopwatch.StartNew();
// 执行检测
stopwatch.Stop();
Console.WriteLine($"耗时: {stopwatch.ElapsedMilliseconds} ms");
```

### 添加日志记录

```csharp
using (var writer = new StreamWriter("test_log.txt", true))
{
    writer.WriteLine($"[{DateTime.Now}] 测试完成");
}
```

## 相关文件

- [DLL 接口定义](../opencv-pro-dll/ProcessLiquidFlowDLL.h)
- [原始 P/Invoke 封装](../docs/LiquidFlowNative.cs)
- [C++ 开发项目](../opencv-pro-develop/)
- [C++ DLL 项目](../opencv-pro-dll/)