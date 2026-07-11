#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <random>
#include <stdexcept>

namespace ContourHighOrder {

// ============================================================================
// 内部工具 — 等弧长重采样 (每个特征函数内部自行调用，不对外暴露)
// ============================================================================
namespace {

/// 计算轮廓总弧长 (闭合)
inline double arcLength_(const std::vector<cv::Point>& contour)
{
    double total = 0.0;
    int n = static_cast<int>(contour.size());
    for (int i = 0; i < n; ++i) {
        const auto& p0 = contour[i];
        const auto& p1 = contour[(i + 1) % n];
        double dx = static_cast<double>(p1.x - p0.x);
        double dy = static_cast<double>(p1.y - p0.y);
        total += std::sqrt(dx * dx + dy * dy);
    }
    return total;
}

/// 等弧长重采样，返回指定数量采样点 (闭合)
inline std::vector<cv::Point2d> resampleContour_(
    const std::vector<cv::Point>& contour, int numSamples)
{
    int n = static_cast<int>(contour.size());
    if (n < 3) {
        std::vector<cv::Point2d> out(numSamples);
        if (n == 0) return out;
        double x = static_cast<double>(contour[0].x);
        double y = static_cast<double>(contour[0].y);
        for (auto& p : out) p = { x, y };
        return out;
    }

    double totalLen = arcLength_(contour);
    double step = totalLen / numSamples;

    std::vector<cv::Point2d> result;
    result.reserve(numSamples);

    double accumulated = 0.0;
    int segIdx = 0;

    for (int i = 0; i < numSamples; ++i) {
        double target = i * step;
        // 前进到包含 target 的线段
        while (segIdx < n) {
            int nextIdx = (segIdx + 1) % n;
            double dx = static_cast<double>(contour[nextIdx].x - contour[segIdx].x);
            double dy = static_cast<double>(contour[nextIdx].y - contour[segIdx].y);
            double segLen = std::sqrt(dx * dx + dy * dy);
            if (accumulated + segLen >= target) {
                // 在当前线段上插值
                double local = target - accumulated;
                double t = (segLen > 1e-9) ? (local / segLen) : 0.0;
                t = std::clamp(t, 0.0, 1.0);
                result.emplace_back(
                    static_cast<double>(contour[segIdx].x) + t * dx,
                    static_cast<double>(contour[segIdx].y) + t * dy);
                break;
            }
            accumulated += segLen;
            segIdx = (segIdx + 1) % n;
        }
    }

    return result;
}

/// 计算质心
inline cv::Point2d centroid_(const std::vector<cv::Point2d>& pts)
{
    if (pts.empty()) return { 0.0, 0.0 };
    double sx = 0.0, sy = 0.0;
    for (const auto& p : pts) {
        sx += p.x;
        sy += p.y;
    }
    return { sx / pts.size(), sy / pts.size() };
}

/// L2 归一化 (in-place)
inline void l2Normalize_(std::vector<double>& v)
{
    double sumSq = 0.0;
    for (double x : v) sumSq += x * x;
    double norm = std::sqrt(sumSq);
    if (norm > 1e-12) {
        for (double& x : v) x /= norm;
    }
}

/// 求阶乘
inline double factorial_(int n)
{
    double result = 1.0;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}

/// Zernike 径向多项式 R_nm(ρ)
inline double zernikeRadial_(int n, int m, double rho)
{
    if ((n - m) % 2 != 0) return 0.0;
    double sum = 0.0;
    int mAbs = std::abs(m);
    for (int s = 0; s <= (n - mAbs) / 2; ++s) {
        int k = n - 2 * s;
        double num = ((s % 2 == 0) ? 1.0 : -1.0) * factorial_(k);
        double den = factorial_(s) *
                     factorial_((n + mAbs) / 2 - s) *
                     factorial_((n - mAbs) / 2 - s);
        sum += (num / den) * std::pow(rho, static_cast<double>(k));
    }
    return sum;
}

} // anonymous namespace

// ============================================================================
// 1. fourierDescriptors — 频域傅里叶描述子
//    原理: 轮廓 → 等弧长重采样 → 复数坐标序列 → DFT → 前 N 个谐波幅值
//    尺度归一化 (除以 DC 分量) 实现平移/旋转/缩放不变性
//    固定维度: 10
// ============================================================================

inline std::vector<double> fourierDescriptors(const std::vector<cv::Point>& contour,
                                              int numDescriptors = 10)
{
    if (contour.size() < 3) return std::vector<double>(numDescriptors, 0.0);

    // 等弧长重采样 → 128 点
    constexpr int RESAMPLE_N = 128;
    std::vector<cv::Point2d> pts = resampleContour_(contour, RESAMPLE_N);

    // 构建复数信号 z[k] = x[k] + j·y[k]
    cv::Mat signal(RESAMPLE_N, 1, CV_64FC2);
    for (int i = 0; i < RESAMPLE_N; ++i) {
        signal.at<cv::Vec2d>(i, 0) = cv::Vec2d(pts[i].x, pts[i].y);
    }

    // DFT
    cv::Mat dftResult;
    cv::dft(signal, dftResult, cv::DFT_COMPLEX_OUTPUT);

    // 提取前 numDescriptors 个非 DC 系数的幅值
    std::vector<double> desc(numDescriptors, 0.0);
    double dc = std::sqrt(dftResult.at<cv::Vec2d>(0, 0)[0] * dftResult.at<cv::Vec2d>(0, 0)[0] +
                          dftResult.at<cv::Vec2d>(0, 0)[1] * dftResult.at<cv::Vec2d>(0, 0)[1]);
    if (dc < 1e-12) return desc;

    for (int i = 0; i < numDescriptors; ++i) {
        int idx = i + 1;
        double real = dftResult.at<cv::Vec2d>(idx, 0)[0];
        double imag = dftResult.at<cv::Vec2d>(idx, 0)[1];
        desc[i] = std::sqrt(real * real + imag * imag) / dc;
    }

    return desc;
}

// ============================================================================
// 2. ellipticalFourierDescriptors — 椭圆傅里叶描述子
//    原理: 等弧长重采样 → 对 x(t) 和 y(t) 分别做椭圆傅里叶展开
//          每阶谐波产生 4 个系数: a_n, b_n, c_n, d_n
//    固定维度: 32 (8 阶 × 4 系数)
// ============================================================================

inline std::vector<double> ellipticalFourierDescriptors(const std::vector<cv::Point>& contour,
                                                        int numHarmonics = 8)
{
    if (contour.size() < 3) return std::vector<double>(numHarmonics * 4, 0.0);

    constexpr int RESAMPLE_N = 256;
    std::vector<cv::Point2d> pts = resampleContour_(contour, RESAMPLE_N);
    int N = RESAMPLE_N;

    // 预处理: 使轮廓闭合 (首尾一致，这里重采样已保证)
    double T = static_cast<double>(N);

    // 计算各阶系数
    // a_n = (T / (2π²n²)) * Σ Δx_p * (cos(2πn t_p/T) - cos(2πn t_{p-1}/T)) / Δt_p
    // 简化实现: 使用离散傅里叶直接计算
    std::vector<double> result(numHarmonics * 4, 0.0);

    // 构建 x(t) 和 y(t) 序列
    std::vector<double> xSeq(N), ySeq(N);
    for (int i = 0; i < N; ++i) {
        xSeq[i] = pts[i].x;
        ySeq[i] = pts[i].y;
    }

    // 去均值
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < N; ++i) { mx += xSeq[i]; my += ySeq[i]; }
    mx /= N; my /= N;
    for (int i = 0; i < N; ++i) { xSeq[i] -= mx; ySeq[i] -= my; }

    // 通过 DFT 计算傅里叶系数
    // x(t) ≈ a0/2 + Σ(a_n cos(2πn t/T) + b_n sin(2πn t/T))
    // y(t) ≈ c0/2 + Σ(c_n cos(2πn t/T) + d_n sin(2πn t/T))
    for (int n = 1; n <= numHarmonics; ++n) {
        double a_n = 0.0, b_n = 0.0, c_n = 0.0, d_n = 0.0;
        for (int t = 0; t < N; ++t) {
            double theta = 2.0 * CV_PI * n * t / T;
            a_n += xSeq[t] * std::cos(theta);
            b_n += xSeq[t] * std::sin(theta);
            c_n += ySeq[t] * std::cos(theta);
            d_n += ySeq[t] * std::sin(theta);
        }
        a_n *= 2.0 / T;
        b_n *= 2.0 / T;
        c_n *= 2.0 / T;
        d_n *= 2.0 / T;

        int base = (n - 1) * 4;
        result[base + 0] = a_n;
        result[base + 1] = b_n;
        result[base + 2] = c_n;
        result[base + 3] = d_n;
    }

    // 归一化: 除以第一阶椭圆的大小，实现尺度不变
    double norm = std::sqrt(result[0] * result[0] + result[1] * result[1] +
                            result[2] * result[2] + result[3] * result[3]);
    if (norm > 1e-12) {
        for (double& v : result) v /= norm;
    }

    return result;
}

// ============================================================================
// 3. zernikeMoments — Zernike 正交矩
//    原理: 等弧长重采样 → 映射到单位圆 → 计算 0~order 各阶 Zernike 矩幅值
//          单位圆上的正交分解，天然旋转不变 (幅值不变，仅相位变化)
//    固定维度: 25 (order=6 时的非冗余矩幅值数量)
// ============================================================================

inline std::vector<double> zernikeMoments(const std::vector<cv::Point>& contour,
                                          int order = 6)
{
    if (contour.size() < 5) {
        int count = (order + 1) * (order + 2) / 2;
        return std::vector<double>(count, 0.0);
    }

    constexpr int RESAMPLE_N = 256;
    std::vector<cv::Point2d> pts = resampleContour_(contour, RESAMPLE_N);
    cv::Point2d c = centroid_(pts);

    // 映射到单位圆: 用最大距离归一化
    double maxDist = 0.0;
    for (const auto& p : pts) {
        double dx = p.x - c.x, dy = p.y - c.y;
        double d = std::sqrt(dx * dx + dy * dy);
        if (d > maxDist) maxDist = d;
    }
    if (maxDist < 1e-9) {
        int count = (order + 1) * (order + 2) / 2;
        return std::vector<double>(count, 0.0);
    }

    // 收集结果: 对每个有效的 (n,m) 对，n = 0..order, m = -n..n 步长2
    std::vector<double> moments;

    for (int n = 0; n <= order; ++n) {
        for (int m = -n; m <= n; m += 2) {
            // Zernike 矩: A_nm = (n+1)/π * Σ_x Σ_y f(x,y) V*_nm(x,y)
            // 对于轮廓: 仅在轮廓点上 f=1
            // V_nm(ρ,θ) = R_nm(ρ) · e^{jmθ}
            double realSum = 0.0, imagSum = 0.0;

            for (const auto& p : pts) {
                double dx = (p.x - c.x) / maxDist;
                double dy = (p.y - c.y) / maxDist;
                double rho = std::sqrt(dx * dx + dy * dy);
                if (rho > 1.0) rho = 1.0;  // 裁剪到单位圆内

                double theta = std::atan2(dy, dx);
                double R = zernikeRadial_(n, m, rho);

                // V*_nm = R_nm(ρ) · e^{-jmθ}
                double cosTerm = std::cos(static_cast<double>(m) * theta);
                double sinTerm = -std::sin(static_cast<double>(m) * theta);  // 共轭
                realSum += R * cosTerm;
                imagSum += R * sinTerm;
            }

            double coeff = static_cast<double>(n + 1) / CV_PI;
            realSum *= coeff / RESAMPLE_N;
            imagSum *= coeff / RESAMPLE_N;

            double mag = std::sqrt(realSum * realSum + imagSum * imagSum);
            moments.push_back(mag);
        }
    }

    // 用 |Z_00| 做尺度归一化
    double z00 = moments.empty() ? 1.0 : moments[0];
    if (z00 > 1e-12) {
        for (double& v : moments) v /= z00;
    }

    return moments;
}

// ============================================================================
// 4. centroidDistanceSignature — 质心距离一维签名
//    原理: 等弧长重采样 → 计算每点到质心的距离 → 均值归一化
//          本质上是将 2D 形状压缩为 1D 距离函数 r(t)
//    固定维度: 64
// ============================================================================

inline std::vector<double> centroidDistanceSignature(const std::vector<cv::Point>& contour,
                                                     int numSamples = 64)
{
    if (contour.size() < 3) return std::vector<double>(numSamples, 0.0);

    std::vector<cv::Point2d> pts = resampleContour_(contour, numSamples);
    cv::Point2d c = centroid_(pts);

    std::vector<double> signature(numSamples, 0.0);
    double sum = 0.0;
    for (int i = 0; i < numSamples; ++i) {
        double dx = pts[i].x - c.x;
        double dy = pts[i].y - c.y;
        signature[i] = std::sqrt(dx * dx + dy * dy);
        sum += signature[i];
    }

    // 均值归一化
    double meanVal = sum / numSamples;
    if (meanVal > 1e-12) {
        for (int i = 0; i < numSamples; ++i) {
            signature[i] /= meanVal;
        }
    }

    return signature;
}

// ============================================================================
// 5. complexCoordinateSignature — 复数坐标一维签名
//    原理: 等弧长重采样 → z(t) = x(t) + j·y(t)，以质心为原点
//          返回向量按 [Re(z₀), Im(z₀), Re(z₁), Im(z₁), ...] 交替排列
//    固定维度: 128 (64 个复数点 × 2 分量)
// ============================================================================

inline std::vector<double> complexCoordinateSignature(const std::vector<cv::Point>& contour,
                                                      int numSamples = 64)
{
    if (contour.size() < 3) return std::vector<double>(numSamples * 2, 0.0);

    std::vector<cv::Point2d> pts = resampleContour_(contour, numSamples);
    cv::Point2d c = centroid_(pts);

    // 计算尺度 (最大距离，用于归一化)
    double maxDist = 0.0;
    for (const auto& p : pts) {
        double dx = p.x - c.x;
        double dy = p.y - c.y;
        double d = std::sqrt(dx * dx + dy * dy);
        if (d > maxDist) maxDist = d;
    }

    std::vector<double> sig(numSamples * 2, 0.0);
    if (maxDist < 1e-12) return sig;

    for (int i = 0; i < numSamples; ++i) {
        sig[2 * i + 0] = (pts[i].x - c.x) / maxDist;  // 实部
        sig[2 * i + 1] = (pts[i].y - c.y) / maxDist;  // 虚部
    }

    return sig;
}

// ============================================================================
// 6. tangentAngleSignature — 切向角累积一维签名
//    原理: 等弧长重采样 → 计算逐点切向角 → 累积 → 去线性趋势
//          累积切向角函数 Θ(t) 描述形状的"转弯"累积量
//    固定维度: 64
// ============================================================================

inline std::vector<double> tangentAngleSignature(const std::vector<cv::Point>& contour,
                                                 int numSamples = 64)
{
    if (contour.size() < 3) return std::vector<double>(numSamples, 0.0);

    std::vector<cv::Point2d> pts = resampleContour_(contour, numSamples);

    // 计算每个采样点的切向角
    std::vector<double> angles(numSamples, 0.0);
    for (int i = 0; i < numSamples; ++i) {
        int next = (i + 1) % numSamples;
        double dx = pts[next].x - pts[i].x;
        double dy = pts[next].y - pts[i].y;
        angles[i] = std::atan2(dy, dx);
    }

    // 累积切向角 (处理相位跳变)
    std::vector<double> cumulative(numSamples, 0.0);
    cumulative[0] = angles[0];
    for (int i = 1; i < numSamples; ++i) {
        double diff = angles[i] - angles[i - 1];
        // 相位展开: 限制在 [-π, π]
        while (diff > CV_PI)  diff -= 2.0 * CV_PI;
        while (diff < -CV_PI) diff += 2.0 * CV_PI;
        cumulative[i] = cumulative[i - 1] + diff;
    }

    // 去线性趋势 (闭合轮廓总转角 = ±2π，去除均匀旋转分量)
    double totalTurn = cumulative[numSamples - 1] - cumulative[0];
    double slope = totalTurn / static_cast<double>(numSamples - 1);
    for (int i = 0; i < numSamples; ++i) {
        cumulative[i] -= slope * i;
    }

    // 归一化到 [-1, 1]
    double maxAbs = 0.0;
    for (double v : cumulative) {
        double a = std::abs(v);
        if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs > 1e-9) {
        for (double& v : cumulative) v /= maxAbs;
    }

    return cumulative;
}

// ============================================================================
// 7. d1Distribution — Osada D1 形状分布
//    原理: 基于质心到轮廓点的距离统计直方图
//          核心理念: 形状 → 距离分布 → 直方图特征 → 形状检索
//    固定维度: 32
// ============================================================================

inline std::vector<double> d1Distribution(const std::vector<cv::Point>& contour,
                                          int numBins = 32)
{
    if (contour.size() < 3) return std::vector<double>(numBins, 0.0);

    constexpr int RESAMPLE_N = 256;
    std::vector<cv::Point2d> pts = resampleContour_(contour, RESAMPLE_N);
    cv::Point2d c = centroid_(pts);

    // 计算所有质心距离
    std::vector<double> distances(RESAMPLE_N);
    double maxDist = 0.0;
    for (int i = 0; i < RESAMPLE_N; ++i) {
        double dx = pts[i].x - c.x;
        double dy = pts[i].y - c.y;
        distances[i] = std::sqrt(dx * dx + dy * dy);
        if (distances[i] > maxDist) maxDist = distances[i];
    }

    // 填充直方图
    std::vector<double> hist(numBins, 0.0);
    if (maxDist < 1e-12) return hist;

    for (double d : distances) {
        double normD = d / maxDist;  // [0, 1]
        int bin = static_cast<int>(normD * numBins);
        bin = std::clamp(bin, 0, numBins - 1);
        hist[bin] += 1.0;
    }

    // 归一化: 转为概率密度
    double total = static_cast<double>(RESAMPLE_N);
    for (double& v : hist) v /= total;

    return hist;
}

// ============================================================================
// 8. d2Distribution — Osada D2 形状分布
//    原理: 基于轮廓上随机点对之间的欧氏距离统计直方图
//          对遮挡和噪声比 D1 更鲁棒
//    固定维度: 32
// ============================================================================

inline std::vector<double> d2Distribution(const std::vector<cv::Point>& contour,
                                          int numBins = 32)
{
    if (contour.size() < 4) return std::vector<double>(numBins, 0.0);

    constexpr int RESAMPLE_N = 256;
    constexpr int SAMPLE_PAIRS = 5000;
    std::vector<cv::Point2d> pts = resampleContour_(contour, RESAMPLE_N);

    // 随机采样点对
    std::mt19937 rng(42);  // 固定种子保证可重复性
    std::uniform_int_distribution<int> dist(0, RESAMPLE_N - 1);

    std::vector<double> pairDistances(SAMPLE_PAIRS);
    double maxDist = 0.0;
    int validPairs = 0;

    for (int k = 0; k < SAMPLE_PAIRS; ++k) {
        int i = dist(rng);
        int j = dist(rng);
        if (i == j) continue;
        double dx = pts[i].x - pts[j].x;
        double dy = pts[i].y - pts[j].y;
        double d = std::sqrt(dx * dx + dy * dy);
        pairDistances[validPairs] = d;
        if (d > maxDist) maxDist = d;
        ++validPairs;
    }

    // 填充直方图
    std::vector<double> hist(numBins, 0.0);
    if (maxDist < 1e-12 || validPairs == 0) return hist;

    for (int k = 0; k < validPairs; ++k) {
        double normD = pairDistances[k] / maxDist;  // [0, 1]
        int bin = static_cast<int>(normD * numBins);
        bin = std::clamp(bin, 0, numBins - 1);
        hist[bin] += 1.0;
    }

    // 归一化
    for (double& v : hist) v /= static_cast<double>(validPairs);

    return hist;
}

// ============================================================================
// 9. curvatureProfile — 离散曲率轮廓
//    原理: 等弧长重采样 → 对每个采样点用前后邻点计算离散曲率 κ ≈ Δθ/Δs
//          曲率描述了轮廓各处的弯曲程度
//    固定维度: 64
// ============================================================================

inline std::vector<double> curvatureProfile(const std::vector<cv::Point>& contour,
                                            int numSamples = 64)
{
    if (contour.size() < 5) return std::vector<double>(numSamples, 0.0);

    std::vector<cv::Point2d> pts = resampleContour_(contour, numSamples);

    int n = numSamples;
    int k = 3;  // 邻域半径

    std::vector<double> curv(n, 0.0);
    double maxCurv = 0.0;

    for (int i = 0; i < n; ++i) {
        int prev = (i - k + n) % n;
        int next = (i + k) % n;

        double v1x = pts[i].x - pts[prev].x;
        double v1y = pts[i].y - pts[prev].y;
        double v2x = pts[next].x - pts[i].x;
        double v2y = pts[next].y - pts[i].y;

        double len1 = std::sqrt(v1x * v1x + v1y * v1y);
        double len2 = std::sqrt(v2x * v2x + v2y * v2y);

        if (len1 < 1e-9 || len2 < 1e-9) {
            curv[i] = 0.0;
            continue;
        }

        double cosTheta = (v1x * v2x + v1y * v2y) / (len1 * len2);
        cosTheta = std::clamp(cosTheta, -1.0, 1.0);
        double theta = std::acos(cosTheta);

        double ds = (len1 + len2) * 0.5;
        if (ds > 1e-9) {
            curv[i] = theta / ds;
        }

        if (curv[i] > maxCurv) maxCurv = curv[i];
    }

    // 尺度归一化
    if (maxCurv > 1e-12) {
        for (double& v : curv) v /= maxCurv;
    }

    return curv;
}

// ============================================================================
// 10. featureDistance — 欧氏距离
//     原理: 两个特征向量之间的 L2 欧氏距离
//     用途: 形状匹配，距离越小表示越相似
// ============================================================================

inline double featureDistance(const std::vector<double>& v1,
                              const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        throw std::runtime_error("featureDistance: vectors must have the same size");
    }
    if (v1.empty()) return 0.0;

    double sumSq = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sumSq += diff * diff;
    }
    return std::sqrt(sumSq);
}

// ============================================================================
// 11. featureCosineSimilarity — 余弦相似度
//     原理: cos(θ) = (v1·v2) / (|v1|·|v2|)，值域 [-1, 1]
//     用途: 形状匹配，越接近 1 表示方向越一致 (对特征尺度不敏感)
// ============================================================================

inline double featureCosineSimilarity(const std::vector<double>& v1,
                                      const std::vector<double>& v2)
{
    if (v1.size() != v2.size()) {
        throw std::runtime_error("featureCosineSimilarity: vectors must have the same size");
    }
    if (v1.empty()) return 0.0;

    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot  += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    double denom = std::sqrt(norm1) * std::sqrt(norm2);
    if (denom < 1e-12) return 0.0;
    return dot / denom;
}

} // namespace ContourHighOrder