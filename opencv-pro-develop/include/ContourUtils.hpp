#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ContourUtils {

// ============================================================================
// 1. 基础几何属性
// ============================================================================

/// 轮廓面积 (包装 cv::contourArea)
inline double area(const std::vector<cv::Point>& contour)
{
    return cv::contourArea(contour);
}

/// 轮廓周长 (包装 cv::arcLength, closed=true)
inline double perimeter(const std::vector<cv::Point>& contour, bool closed = true)
{
    return cv::arcLength(contour, closed);
}

/// 正外接矩形
inline cv::Rect boundingRect(const std::vector<cv::Point>& contour)
{
    return cv::boundingRect(contour);
}

/// 最小旋转矩形
inline cv::RotatedRect minAreaRect(const std::vector<cv::Point>& contour)
{
    return cv::minAreaRect(contour);
}

/// 最小外接圆, 返回 (center, radius)
inline std::pair<cv::Point2f, float> minEnclosingCircle(const std::vector<cv::Point>& contour)
{
    cv::Point2f center;
    float radius = 0.0f;
    cv::minEnclosingCircle(contour, center, radius);
    return { center, radius };
}

/// 拟合椭圆 (最少5个点)
inline cv::RotatedRect fitEllipse(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 5) {
        throw std::runtime_error("fitEllipse requires at least 5 points");
    }
    return cv::fitEllipse(contour);
}

/// 质心
inline cv::Point2d centroid(const std::vector<cv::Point>& contour)
{
    cv::Moments m = cv::moments(contour, true);
    if (std::abs(m.m00) < 1e-9) {
        return { 0.0, 0.0 };
    }
    return { m.m10 / m.m00, m.m01 / m.m00 };
}

/// 图像矩
inline cv::Moments moments(const std::vector<cv::Point>& contour)
{
    return cv::moments(contour, true);
}

// ============================================================================
// 2. 形状描述符
// ============================================================================

/// 圆度: 4π * area / perimeter²  (完美圆 = 1.0)
inline double circularity(const std::vector<cv::Point>& contour)
{
    double a = cv::contourArea(contour);
    double p = cv::arcLength(contour, true);
    if (p < 1e-9) return 0.0;
    return (4.0 * CV_PI * a) / (p * p);
}

/// 矩形度: area / boundingRect.area
inline double rectangularity(const std::vector<cv::Point>& contour)
{
    double a = cv::contourArea(contour);
    cv::Rect r = cv::boundingRect(contour);
    double rectArea = static_cast<double>(r.width) * static_cast<double>(r.height);
    if (rectArea < 1e-9) return 0.0;
    return a / rectArea;
}

/// 宽高比 (boundingRect width / height)
inline double aspectRatio(const std::vector<cv::Point>& contour)
{
    cv::Rect r = cv::boundingRect(contour);
    if (r.height == 0) return 0.0;
    return static_cast<double>(r.width) / static_cast<double>(r.height);
}

/// 扩展度: contourArea / convexHullArea  (<= 1.0)
inline double solidity(const std::vector<cv::Point>& contour)
{
    double a = cv::contourArea(contour);
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double hullArea = cv::contourArea(hull);
    if (hullArea < 1e-9) return 0.0;
    return a / hullArea;
}

/// 等效直径: sqrt(4 * area / π)
inline double equivalentDiameter(const std::vector<cv::Point>& contour)
{
    double a = cv::contourArea(contour);
    return std::sqrt(4.0 * a / CV_PI);
}

/// 偏心率 (基于拟合椭圆的长短轴比值, 或基于矩)
inline double eccentricity(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 5) {
        // 如果没有足够点拟合椭圆, 回退到基于 boundingRect 的近似
        cv::Rect r = cv::boundingRect(contour);
        double w = static_cast<double>(r.width);
        double h = static_cast<double>(r.height);
        if (w < 1e-9 && h < 1e-9) return 0.0;
        double major = std::max(w, h);
        double minor = std::min(w, h);
        if (major < 1e-9) return 0.0;
        return std::sqrt(1.0 - (minor * minor) / (major * major));
    }
    cv::RotatedRect ellipse = cv::fitEllipse(contour);
    double major = std::max(ellipse.size.width, ellipse.size.height);
    double minor = std::min(ellipse.size.width, ellipse.size.height);
    if (major < 1e-9) return 0.0;
    return std::sqrt(1.0 - (minor * minor) / (major * major));
}

/// 最小旋转矩形的宽高比
inline double minRectAspectRatio(const std::vector<cv::Point>& contour)
{
    cv::RotatedRect r = cv::minAreaRect(contour);
    double w = static_cast<double>(r.size.width);
    double h = static_cast<double>(r.size.height);
    if (w < 1e-9 && h < 1e-9) return 0.0;
    double major = std::max(w, h);
    double minor = std::min(w, h);
    if (minor < 1e-9) return 0.0;
    return major / minor;
}

// ============================================================================
// 2.5 形状相似度综合评分 (0.0 ~ 1.0, 越接近 1 越相似)
// ============================================================================

/// 圆形相似度: 综合圆度、最小外接圆面积比、偏心率
inline double circleSimilarity(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 5) return 0.0;

    double a = cv::contourArea(contour);
    double p = cv::arcLength(contour, true);
    if (a < 1e-9 || p < 1e-9) return 0.0;

    // 1) 圆度: 4π·area / perimeter² (完美圆 = 1)
    double circ = (4.0 * CV_PI * a) / (p * p);
    if (circ > 1.0) circ = 1.0;                    // 截断, 防止数值噪声导致 >1

    // 2) 最小外接圆面积比: contourArea / enclosingCircleArea
    cv::Point2f center;
    float radius = 0.0f;
    cv::minEnclosingCircle(contour, center, radius);
    double encArea = CV_PI * static_cast<double>(radius) * static_cast<double>(radius);
    double encRatio = (encArea > 1e-9) ? (a / encArea) : 0.0;
    if (encRatio > 1.0) encRatio = 1.0;

    // 3) 偏心率补偿: 圆形偏心率 → 0, 所以 1-ecc ≈ 1
    cv::RotatedRect ellipse = cv::fitEllipse(contour);
    double major = static_cast<double>(std::max(ellipse.size.width, ellipse.size.height));
    double minor = static_cast<double>(std::min(ellipse.size.width, ellipse.size.height));
    double ecc = 0.0;
    if (major > 1e-9) {
        ecc = std::sqrt(1.0 - (minor * minor) / (major * major));
    }
    double eccScore = 1.0 - ecc;                      // 偏心率越低越好

    // 加权综合
    constexpr double wCirc    = 0.50;
    constexpr double wEnc     = 0.45;
    constexpr double wEcc     = 0.05;
    double score = wCirc * circ + wEnc * encRatio + wEcc * eccScore;

    return std::clamp(score, 0.0, 1.0);
}

/// 长方形相似度: 综合矩形度、角点数目、最小旋转矩形面积比
inline double rectangleSimilarity(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 4) return 0.0;

    double a = cv::contourArea(contour);
    if (a < 1e-9) return 0.0;

    // 1) 矩形度: contourArea / boundingRect.area
    cv::Rect br = cv::boundingRect(contour);
    double brArea = static_cast<double>(br.width) * static_cast<double>(br.height);
    double rect = (brArea > 1e-9) ? (a / brArea) : 0.0;
    if (rect > 1.0) rect = 1.0;

    // 2) 角点数目得分: 多边形逼近顶点数接近 4 = 高分
    double peri = cv::arcLength(contour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.02 * peri, true);
    int numVertices = static_cast<int>(approx.size());
    double vertexScore = 1.0;
    if (numVertices == 3) {
        vertexScore = 0.70;
    } else if (numVertices == 4) {
        vertexScore = 1.00;
    } else if (numVertices == 5) {
        vertexScore = 0.75;
    } else if (numVertices == 6) {
        vertexScore = 0.50;
    } else if (numVertices > 6) {
        vertexScore = std::max(0.1, 1.0 - 0.08 * (numVertices - 4));
    } else {
        vertexScore = 0.30;
    }

    // 3) 最小旋转矩形面积比: contourArea / minAreaRect.area
    cv::RotatedRect mr = cv::minAreaRect(contour);
    double mrArea = static_cast<double>(mr.size.width) * static_cast<double>(mr.size.height);
    double mrRatio = (mrArea > 1e-9) ? (a / mrArea) : 0.0;
    if (mrRatio > 1.0) mrRatio = 1.0;

    // 加权综合
    constexpr double wRect      = 0.50;
    constexpr double wVertex    = 0.30;
    constexpr double wMR        = 0.20;
    double score = wRect * rect + wVertex * vertexScore + wMR * mrRatio;

    return std::clamp(score, 0.0, 1.0);
}

// ============================================================================
// 2.6 曲率分析 — 圆形 vs 长方形
// ============================================================================

/// 离散轮廓曲率计算
/// 对轮廓上每个点，用前后 k 个邻点构造两个向量，
/// 通过向量夹角除以弧长近似该点的曲率  κ ≈ Δθ / Δs
/// @param contour  输入轮廓点序列
/// @param k        平滑半径 (点数), 默认 5。k 越大曲率越平滑但越不敏感
/// @return         各点曲率值 (与 contour 等长), 值域 >= 0
inline std::vector<double> computeCurvature(const std::vector<cv::Point>& contour, int k = 5)
{
    const int n = static_cast<int>(contour.size());
    if (n < 2 * k + 1) {
        return std::vector<double>(n, 0.0);
    }

    std::vector<double> curv(n, 0.0);

    for (int i = 0; i < n; ++i) {
        // 前点 (闭合取模)
        int prev = (i - k + n) % n;
        // 后点
        int next = (i + k) % n;

        const auto& pi = contour[i];
        const auto& pp = contour[prev];
        const auto& pn = contour[next];

        // 向量 v1 = pi - pp,  v2 = pn - pi
        double v1x = static_cast<double>(pi.x - pp.x);
        double v1y = static_cast<double>(pi.y - pp.y);
        double v2x = static_cast<double>(pn.x - pi.x);
        double v2y = static_cast<double>(pn.y - pi.y);

        double len1 = std::sqrt(v1x * v1x + v1y * v1y);
        double len2 = std::sqrt(v2x * v2x + v2y * v2y);

        if (len1 < 1e-9 || len2 < 1e-9) {
            curv[i] = 0.0;
            continue;
        }

        // cosθ = (v1·v2) / (|v1|·|v2|)
        double cosTheta = (v1x * v2x + v1y * v2y) / (len1 * len2);
        // 数值裁剪防越界
        cosTheta = std::clamp(cosTheta, -1.0, 1.0);
        double theta = std::acos(cosTheta);        // 方向变化角 [0, π]

        // 弧长 ≈ (len1 + len2) / 2
        double ds = (len1 + len2) * 0.5;
        if (ds > 1e-9) {
            curv[i] = theta / ds;
        } else {
            curv[i] = 0.0;
        }
    }

    return curv;
}

/// 曲率统计: 均值、标准差、变异系数、低曲率占比、高曲率占比
struct CurvatureStats {
    double mean    = 0.0;   // 曲率均值
    double stddev  = 0.0;   // 曲率标准差
    double cv      = 0.0;   // 变异系数  σ / |μ|
    double lowRatio = 0.0;  // 低曲率点占比 (κ < lowThresh, 近似直线)
    double highRatio = 0.0; // 高曲率点占比 (κ > highThresh, 近似角点)
};

/// 计算曲率统计信息
/// @param curv         曲率序列
/// @param lowThresh    低曲率阈值 (低于此值视为直线段)
/// @param highThresh   高曲率阈值 (高于此值视为角点)
inline CurvatureStats curvatureStatistics(const std::vector<double>& curv,
                                          double lowThresh = 0.02,
                                          double highThresh = 0.15)
{
    CurvatureStats stats;
    if (curv.empty()) return stats;

    const int n = static_cast<int>(curv.size());

    // 均值
    double sum = 0.0;
    for (double c : curv) sum += c;
    stats.mean = sum / n;

    // 标准差 & 低/高比率
    double sumSq = 0.0;
    int lowCount = 0;
    int highCount = 0;
    for (double c : curv) {
        double diff = c - stats.mean;
        sumSq += diff * diff;
        if (c < lowThresh)  ++lowCount;
        if (c > highThresh) ++highCount;
    }

    double variance = (n > 1) ? (sumSq / static_cast<double>(n)) : 0.0;
    stats.stddev = std::sqrt(variance);
    stats.cv = (std::abs(stats.mean) > 1e-9) ? (stats.stddev / std::abs(stats.mean)) : 0.0;
    stats.lowRatio  = static_cast<double>(lowCount)  / n;
    stats.highRatio = static_cast<double>(highCount) / n;

    return stats;
}

/// 基于曲率均匀度的圆形相似度 (0.0~1.0)
/// 原理: 圆的曲率处处相等, 变异系数(CV)趋近于 0
///       CV 越大说明曲率越不均匀, 越不像圆
inline double circleSimilarityByCurvature(const std::vector<cv::Point>& contour, int k = 5)
{
    if (contour.size() < 2 * k + 1) return 0.0;

    std::vector<double> curv = computeCurvature(contour, k);
    CurvatureStats stats = curvatureStatistics(curv);

    // 变异系数 → 圆形评分:  CV=0 时得分 1.0, CV 越大得分越低
    double cv = stats.cv;
    // 使用指数衰减: exp(-α·CV²),  α 控制衰减速度
    constexpr double alpha = 15.0;
    double cvScore = std::exp(-alpha * cv * cv);

    // 补充: 低曲率占比过高也不行 (圆不应该有大量接近直线的段)
    double lowPenalty = 1.0 - stats.lowRatio;

    double score = cvScore * lowPenalty;
    return std::clamp(score, 0.0, 1.0);
}

/// 基于曲率双峰分布的矩形相似度 (0.0~1.0)
/// 原理: 长方形四边曲率≈0 (低), 四角曲率很高 (高),
///       曲率分布呈双峰: 大量低值 + 少量极高值
///       表现为: 低曲率占比高、高曲率占比低但存在、标准差大
inline double rectangleSimilarityByCurvature(const std::vector<cv::Point>& contour, int k = 5)
{
    if (contour.size() < 2 * k + 1) return 0.0;

    std::vector<double> curv = computeCurvature(contour, k);
    CurvatureStats stats = curvatureStatistics(curv);

    // 1) 低曲率占比得分: 矩形的边占大部分, 期望 > 0.6
    double lowScore = stats.lowRatio;
    // 饱和映射: > 0.7 接近满分
    lowScore = std::min(1.0, lowScore / 0.70);

    // 2) 高曲率存在性: 矩形必须有几个尖峰, 但不能太多
    double highScore = 0.0;
    if (stats.highRatio > 0.02 && stats.highRatio < 0.50) {
        // 理想矩形: 4 个角 / 总点数 ≈ 小比例
        highScore = 1.0;
    } else if (stats.highRatio >= 0.50) {
        highScore = 1.0 - (stats.highRatio - 0.50);  // 太多高曲率 → 减分
    } else {
        highScore = stats.highRatio / 0.02;  // < 0.02 线性衰减
    }
    highScore = std::clamp(highScore, 0.0, 1.0);

    // 3) 变异系数得分: 矩形 CV 应该较大 (不均匀)
    double cvScore = std::min(1.0, stats.cv / 2.0);  // CV > 2 满分

    // 4) 峰度近似: 低曲率占比高 + 存在少量极端值 → 典型矩形
    double bimodalScore = lowScore * highScore;

    // 加权
    constexpr double wLow      = 0.40;
    constexpr double wHigh     = 0.25;
    constexpr double wCV       = 0.15;
    constexpr double wBimodal  = 0.20;
    double score = wLow * lowScore + wHigh * highScore + wCV * cvScore + wBimodal * bimodalScore;

    return std::clamp(score, 0.0, 1.0);
}

// ============================================================================
// 3. Hu 矩与形状匹配
// ============================================================================

/// 计算 Hu 矩 (7 个不变矩)
inline std::vector<double> huMoments(const std::vector<cv::Point>& contour)
{
    cv::Moments m = cv::moments(contour, true);
    if (std::abs(m.m00) < 1e-9) {
        return std::vector<double>(7, 0.0);
    }
    double hu[7];
    cv::HuMoments(m, hu);
    return { hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6] };
}

/// 基于 Hu 矩的形状匹配 (值越小越相似, 0 = 完全相同)
inline double matchShapes(const std::vector<cv::Point>& contour1,
                          const std::vector<cv::Point>& contour2,
                          int method = cv::CONTOURS_MATCH_I1)
{
    return cv::matchShapes(contour1, contour2, method, 0.0);
}

/// 计算形状距离 (是对数绝对值归一化的欧式距离, 便于比较)
inline double shapeDistance(const std::vector<cv::Point>& contour1,
                            const std::vector<cv::Point>& contour2)
{
    std::vector<double> hu1 = huMoments(contour1);
    std::vector<double> hu2 = huMoments(contour2);

    double dist = 0.0;
    for (int i = 0; i < 7; i++) {
        double m1 = std::abs(hu1[i]);
        double m2 = std::abs(hu2[i]);
        double numer = m1 - m2;
        double denom = std::max(m1, m2);
        if (denom > 1e-9) {
            dist += std::abs(numer / denom);
        }
    }
    return dist / 7.0;
}

// ============================================================================
// 4. 凸包 & 多边形逼近
// ============================================================================

/// 凸包
inline std::vector<cv::Point> convexHull(const std::vector<cv::Point>& contour, bool clockwise = false)
{
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull, clockwise, true);
    return hull;
}

/// 凸包 (返回索引)
inline std::vector<int> convexHullIndices(const std::vector<cv::Point>& contour, bool clockwise = false)
{
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices, clockwise, false);
    return hullIndices;
}

/// 凸性缺陷
inline std::vector<cv::Vec4i> convexityDefects(const std::vector<cv::Point>& contour)
{
    std::vector<int> hullIndices = convexHullIndices(contour);
    if (hullIndices.size() < 3) {
        return {};
    }
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    return defects;
}

/// 多边形逼近 (自动计算 epsilon = fraction * perimeter)
inline std::vector<cv::Point> approxPolyDP(const std::vector<cv::Point>& contour,
                                           double epsilonFraction = 0.01, bool closed = true)
{
    double peri = cv::arcLength(contour, closed);
    double epsilon = epsilonFraction * peri;
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, epsilon, closed);
    return approx;
}

/// 多边形逼近 (指定 epsilon 绝对值)
inline std::vector<cv::Point> approxPolyDPExact(const std::vector<cv::Point>& contour,
                                                double epsilon, bool closed = true)
{
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, epsilon, closed);
    return approx;
}

/// 判断轮廓是否为凸
inline bool isContourConvex(const std::vector<cv::Point>& contour)
{
    return cv::isContourConvex(contour);
}

// ============================================================================
// 5. 特征点
// ============================================================================

/// 极端点: 返回 { topmost, bottommost, leftmost, rightmost }
inline std::tuple<cv::Point, cv::Point, cv::Point, cv::Point>
extremePoints(const std::vector<cv::Point>& contour)
{
    if (contour.empty()) {
        return {};
    }
    cv::Point top    = *std::min_element(contour.begin(), contour.end(),
        [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    cv::Point bottom = *std::max_element(contour.begin(), contour.end(),
        [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
    cv::Point left   = *std::min_element(contour.begin(), contour.end(),
        [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
    cv::Point right  = *std::max_element(contour.begin(), contour.end(),
        [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
    return { top, bottom, left, right };
}

/// 轮廓上的角点检测 (基于多边形逼近的顶点)
inline std::vector<cv::Point> cornerPoints(const std::vector<cv::Point>& contour,
                                           double epsilonFraction = 0.02, bool closed = true)
{
    return approxPolyDP(contour, epsilonFraction, closed);
}

// ============================================================================
// 6. 方向分析
// ============================================================================

/// 拟合直线 (最小二乘), 返回 (vx, vy, x0, y0)
inline cv::Vec4f fitLine(const std::vector<cv::Point>& contour)
{
    cv::Vec4f line;
    std::vector<cv::Point2f> pts;
    pts.reserve(contour.size());
    for (const auto& p : contour) {
        pts.emplace_back(cv::Point2f(p.x, p.y));
    }
    cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
    return line;
}

/// 主轴方向角度 (基于最小旋转矩形)
inline double orientation(const std::vector<cv::Point>& contour)
{
    cv::RotatedRect r = cv::minAreaRect(contour);
    double angle = r.angle;
    // 归一化: 当 width < height 时, OpenCV 返回 angle + 90
    if (r.size.width < r.size.height) {
        angle += 90.0;
    }
    return angle;
}

/// 基于矩计算方向角度 (0~180 度)
inline double orientationByMoments(const std::vector<cv::Point>& contour)
{
    cv::Moments m = cv::moments(contour, true);
    if (std::abs(m.m00) < 1e-9) return 0.0;
    double mu20 = m.mu20 / m.m00;
    double mu02 = m.mu02 / m.m00;
    double mu11 = m.mu11 / m.m00;
    if (std::abs(mu20 - mu02) < 1e-9) return 0.0;
    double theta = 0.5 * std::atan2(2.0 * mu11, mu20 - mu02);
    return theta * 180.0 / CV_PI;
}

// ============================================================================
// 7. 轮廓点信息
// ============================================================================

/// 获取轮廓点数
inline size_t pointCount(const std::vector<cv::Point>& contour)
{
    return contour.size();
}

/// 轮廓点的质心
inline cv::Point2d meanPoint(const std::vector<cv::Point>& contour)
{
    if (contour.empty()) return { 0.0, 0.0 };
    double sx = 0.0, sy = 0.0;
    for (const auto& p : contour) {
        sx += p.x;
        sy += p.y;
    }
    return { sx / contour.size(), sy / contour.size() };
}

/// 是否闭合 (首尾点相同或非常接近)
inline bool isClosed(const std::vector<cv::Point>& contour, int tolerance = 1)
{
    if (contour.size() < 2) return false;
    const auto& first = contour.front();
    const auto& last  = contour.back();
    return (std::abs(first.x - last.x) <= tolerance &&
            std::abs(first.y - last.y) <= tolerance);
}

} // namespace ContourUtils