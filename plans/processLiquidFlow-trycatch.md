# 计划：processLiquidFlowImage 轻量级 try-catch 加固

## 目标

在 [`processLiquidFlowImage`](opencv-pro-develop/include/ProcessLiquidFlow.hpp:88) 函数外层包裹 try-catch，捕获 OpenCV 异常和标准库异常，防止在工业现场多变光照条件下因偶发异常导致程序崩溃。**保持现有管线逻辑不变**。

## 修改文件

- `opencv-pro-develop/include/ProcessLiquidFlow.hpp`

## 具体修改

### 当前代码结构 (L88-L239)

```cpp
inline void processLiquidFlowImage(const cv::Mat& image)
{
    // 灰度化 & 双边滤波
    // 二值化
    // 连通域分析
    // D1特征匹配
    // ROI旋转扫描寻谷
    // 可视化
}
```

### 修改后结构

```cpp
inline void processLiquidFlowImage(const cv::Mat& image)
{
    try
    {
        // ======== 原有全部管线代码，完全不动 ========
        // 灰度化 & 双边滤波
        // 二值化
        // 连通域分析
        // D1特征匹配
        // ROI旋转扫描寻谷
        // 可视化
        // ============================================
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[processLiquidFlowImage] OpenCV Exception: " << e.what() << std::endl;
        std::cerr << "  err code=" << e.code << " func=" << e.func
                  << " file=" << e.file << " line=" << e.line << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[processLiquidFlowImage] std::exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[processLiquidFlowImage] Unknown exception caught." << std::endl;
    }
}
```

## 异常覆盖范围

| catch 子句 | 覆盖的典型异常 |
|------------|---------------|
| `cv::Exception` | `cv::threshold` 空Mat、`cv::warpAffine` 参数异常、`cv::connectedComponentsWithStats` 内存不足等 |
| `std::exception` | `featureCosineSimilarity` 尺寸不匹配抛出的 `std::runtime_error`、`ContourUtils::fitEllipse` 点数不足等 |
| `...` | 其他未知异常（如 `std::bad_alloc` 等非标准派生类） |

## 不涉及

- ❌ 不修改任何管线算法参数
- ❌ 不添加 CLAHE / OTSU / 自适应策略
- ❌ 不修改 `processLiquidFlows()` 调用方
- ❌ 不改变函数签名
- ❌ 不引入新的头文件依赖（`<exception>`, `<stdexcept>` 已间接包含）

## 实施步骤

1. 在 `processLiquidFlowImage` 函数体开头（L89 之前）插入 `try {`
2. 在函数体末尾（L239 `}` 之前）插入 catch 三个子句
3. 重新缩进原有代码块（增加一级缩进）
