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