using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;

namespace OpenCvProTest;

/// <summary>
/// 液流检测结果结构体 —— 与 C++ LiquidFlowResult 一一对应
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct LiquidFlowResult
{
    public float BestAngle;   // 最佳旋转角度 (度)
    public float BestIdx1;    // 谷底1 在全局图像中的 x 坐标
    public float BestIdx2;    // 谷底2 在全局图像中的 x 坐标
    public int   Success;     // 1=成功, 0=失败
}

/// <summary>
/// ProcessLiquidFlow P/Invoke 封装
/// </summary>
public static class LiquidFlowNative
{
    /// <summary>
    /// 调用 C++ DLL 进行液流检测
    /// </summary>
    /// <param name="imgData">原始 BGR 像素数据 (byte[])</param>
    /// <param name="width">图像宽度</param>
    /// <param name="height">图像高度</param>
    /// <param name="channels">通道数 (3=BGR, 1=Gray)</param>
    /// <param name="savePath">grayImage 保存路径 (完整路径, 含文件名)</param>
    /// <param name="result">[out] 检测结果</param>
    /// <param name="diagLog">[out] 诊断日志缓冲区</param>
    /// <param name="diagLogSize">缓冲区大小</param>
    /// <returns>0=成功, -1=参数错误, -2=处理异常</returns>
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

/// <summary>
/// 液流检测高级封装 —— 提供更友好的 C# API
/// </summary>
public static class LiquidFlowDetector
{
    /// <summary>
    /// 从 Bitmap 对象执行液流检测
    /// </summary>
    /// <param name="bmp">输入图像 (System.Drawing.Bitmap)</param>
    /// <param name="savePath">grayImage 保存路径</param>
    /// <param name="result">[out] 检测结果</param>
    /// <param name="diagLog">[out] 诊断日志</param>
    /// <returns>0=成功, 负值=失败码</returns>
    public static int Detect(Bitmap bmp, string savePath, out LiquidFlowResult result, out string diagLog)
    {
        result = default;
        diagLog = "";

        if (bmp == null)
            return -1;

        // 1. 从 Bitmap 提取原始像素数据 (确保 BGR 格式)
        Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
        BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly,
            bmp.PixelFormat == PixelFormat.Format8bppIndexed
                ? PixelFormat.Format8bppIndexed
                : PixelFormat.Format24bppRgb);

        int channels = (bmp.PixelFormat == PixelFormat.Format8bppIndexed) ? 1 : 3;
        int stride = bmpData.Stride;
        int dataLen = stride * bmp.Height;

        byte[] imgData = new byte[dataLen];
        Marshal.Copy(bmpData.Scan0, imgData, 0, dataLen);
        bmp.UnlockBits(bmpData);

        // 2. 调用原生 DLL
        var sb = new StringBuilder(4096);
        var res = new LiquidFlowResult();

        int ret = LiquidFlowNative.ProcessLiquidFlow(
            imgData,
            bmp.Width,
            bmp.Height,
            channels,
            savePath ?? "",
            ref res,
            sb,
            sb.Capacity
        );

        result = res;
        diagLog = sb.ToString();
        return ret;
    }

    /// <summary>
    /// 从图像文件路径执行液流检测
    /// </summary>
    /// <param name="imagePath">输入图像文件路径</param>
    /// <param name="savePath">grayImage 保存路径</param>
    /// <param name="result">[out] 检测结果</param>
    /// <param name="diagLog">[out] 诊断日志</param>
    /// <returns>0=成功, 负值=失败码</returns>
    public static int DetectFromFile(string imagePath, string savePath,
        out LiquidFlowResult result, out string diagLog)
    {
        result = default;
        diagLog = "";

        try
        {
            using (var bmp = new Bitmap(imagePath))
            {
                return Detect(bmp, savePath, out result, out diagLog);
            }
        }
        catch (Exception ex)
        {
            diagLog = $"FAIL: Bitmap load error - {ex.Message}";
            return -1;
        }
    }
}