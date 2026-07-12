using System;
using System.IO;

namespace OpenCvProTest;

/// <summary>
/// 液流检测测试程序
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("========================================");
        Console.WriteLine("  OpenCV Pro - 液流检测测试程序");
        Console.WriteLine("========================================");
        Console.WriteLine();

        // 设置默认路径
        string defaultImagePath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "image", "test.jpg");
        string defaultSavePath = Path.Combine(AppContext.BaseDirectory, "grayImage.png");

        // 获取输入图像路径
        string imagePath = GetInputPath("请输入测试图像路径", defaultImagePath);
        
        // 获取灰度图像保存路径
        string savePath = GetInputPath("请输入灰度图像保存路径", defaultSavePath);

        // 确保保存目录存在
        string? saveDir = Path.GetDirectoryName(savePath);
        if (!string.IsNullOrEmpty(saveDir) && !Directory.Exists(saveDir))
        {
            Directory.CreateDirectory(saveDir);
            Console.WriteLine($"创建目录: {saveDir}");
        }

        Console.WriteLine();
        Console.WriteLine("开始处理...");
        Console.WriteLine();

        // 调用液流检测
        int ret = LiquidFlowDetector.DetectFromFile(
            imagePath,
            savePath,
            out LiquidFlowResult result,
            out string diagLog
        );

        // 输出返回码
        Console.WriteLine($"返回码: {ret}");
        Console.WriteLine();

        // 输出诊断日志
        Console.WriteLine("--- 诊断日志 ---");
        Console.WriteLine(diagLog);
        Console.WriteLine();

        // 输出检测结果
        if (result.Success == 1)
        {
            Console.WriteLine("=== 检测成功 ===");
            Console.WriteLine($"  最佳旋转角度: {result.BestAngle:F2} 度");
            Console.WriteLine($"  谷底1 X坐标: {result.BestIdx1:F1}");
            Console.WriteLine($"  谷底2 X坐标: {result.BestIdx2:F1}");
            Console.WriteLine($"  灰度图像已保存至: {savePath}");
        }
        else
        {
            Console.WriteLine("=== 检测失败 ===");
            Console.WriteLine($"  Success = {result.Success}");
        }

        Console.WriteLine();
        Console.WriteLine("按任意键退出...");
        Console.ReadKey();
    }

    /// <summary>
    /// 获取用户输入路径
    /// </summary>
    static string GetInputPath(string prompt, string defaultPath)
    {
        Console.WriteLine($"{prompt} (默认: {defaultPath}):");
        string? input = Console.ReadLine();

        if (string.IsNullOrWhiteSpace(input))
        {
            return defaultPath;
        }

        // 处理相对路径
        if (!Path.IsPathRooted(input))
        {
            return Path.Combine(AppContext.BaseDirectory, input);
        }

        return input;
    }
}