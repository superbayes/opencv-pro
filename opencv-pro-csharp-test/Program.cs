using System;

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

        // 输入图像路径
        string imagePath = @"..\..\..\..\image\test.jpg";
        // 灰度图像保存路径
        string savePath = "grayImage.png";

        Console.WriteLine("开始处理...");

        // 调用液流检测
        int ret = LiquidFlowDetector.DetectFromFile(
            imagePath,
            savePath,
            out LiquidFlowResult result,
            out string diagLog
        );

        // 输出返回码
        Console.WriteLine($"返回码: {ret}");


        // 输出诊断日志
        Console.WriteLine("--- 诊断日志 ---");
        Console.WriteLine(diagLog);

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

}