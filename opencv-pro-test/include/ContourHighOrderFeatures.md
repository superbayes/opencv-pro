# ContourHighOrderFeatures 说明文档

## 概述

命名空间 `ContourHighOrder` 提供 **9 种高阶轮廓特征提取函数** + **2 种向量比较函数**。

- **所有特征函数自包含** — 内部自行完成等弧长重采样与归一化，无需外界预处理
- **零外部依赖** — 仅依赖 OpenCV 与 C++ 标准库，不依赖 `ContourUtils.hpp`
- **固定最佳维度** — 每个函数都有推荐的默认输出维度，直接调用无参版本即可
- **统一接口** — 输入 `std::vector<cv::Point>`，输出 `std::vector<double>`

```cpp
#include "ContourHighOrderFeatures.hpp"

std::vector<cv::Point> contour = /* 用户输入的轮廓点集 */;

auto fd  = ContourHighOrder::fourierDescriptors(contour);             // 10 维
auto efd = ContourHighOrder::ellipticalFourierDescriptors(contour);   // 32 维
auto zm  = ContourHighOrder::zernikeMoments(contour);                 // 25 维
auto sig = ContourHighOrder::centroidDistanceSignature(contour);      // 64 维

// 比较两个特征向量
double dist = ContourHighOrder::featureDistance(fd1, fd2);
double sim  = ContourHighOrder::featureCosineSimilarity(fd1, fd2);
```

---

## 函数清单

| # | 函数名 | 原理简介 | 维度 | 适用用途 |
|---|--------|---------|:----:|---------|
| 1 | `fourierDescriptors` | 轮廓→复信号→DFT→取前10阶谐波幅值，DC归一化 | **10** | **默认首选**，通用形状分类、旋转不变匹配 |
| 2 | `ellipticalFourierDescriptors` | x(t)/y(t) 分别傅里叶展开，8阶×4系数=a_n,b_n,c_n,d_n | **32** | 复杂形状重建与识别，对不规则轮廓鲁棒 |
| 3 | `zernikeMoments` | 单位圆上 Zernike 正交矩分解，order=6，取幅值 | **25** | **旋转不变识别首选**，抗噪强，MPEG-7 标准 |
| 4 | `centroidDistanceSignature` | 64 点等弧长采样，每点到质心的距离，均值归一化 | **64** | 形状匹配、局部膨胀/收缩检测 |
| 5 | `complexCoordinateSignature` | 64 点复数坐标 z(t)=x+jy，质心原点，最大距离归一化 | **128** | 轮廓重建，保留完整位置信息 |
| 6 | `tangentAngleSignature` | 64 点累积切向角序列，去线性趋势 | **64** | 角点检测、形状分段、拐点分析 |
| 7 | `d1Distribution` | 质心到轮廓点距离的32-bin概率直方图 | **32** | 形状检索、统计形状差异分析 |
| 8 | `d2Distribution` | 轮廓上随机点对距离的32-bin概率直方图 | **32** | 形状检索，对遮挡和噪声比 D1 更鲁棒 |
| 9 | `curvatureProfile` | 64 点离散曲率序列 κ≈Δθ/Δs，最大曲率归一化 | **64** | 局部形状特征、缺陷检测、凹凸分析 |
| 10 | `featureDistance` | 两向量 L2 归一化后欧氏距离 ÷ 2，值域 [0, 1] | **1** | d=0 完全相同，d=1 正交/相反 |
| 11 | `featureCosineSimilarity` | 余弦相似度映射到 [0, 1]，sim = (cosθ+1)/2 | **1** | sim=0 相反，sim=0.5 正交，sim=1 相同 |

---

## 各函数详细说明

### 1. `fourierDescriptors` — 频域傅里叶描述子

```cpp
std::vector<double> fourierDescriptors(const std::vector<cv::Point>& contour,
                                       int numDescriptors = 10);
```

**算法步骤：**
1. 等弧长重采样 → 128 点
2. 构建复数信号 z[k] = x[k] + j·y[k]
3. OpenCV `cv::dft()` 做 DFT
4. 取前 `numDescriptors` 个非 DC 系数的幅值
5. 除以 DC 幅值，实现平移、旋转、缩放不变性

**维度选择理由：** 前 10 个谐波已捕获约 95% 的形状能量，是学术界和工业界的经典取值。

**典型应用：** 通用形状分类、字符识别、工件匹配。

---

### 2. `ellipticalFourierDescriptors` — 椭圆傅里叶描述子

```cpp
std::vector<double> ellipticalFourierDescriptors(const std::vector<cv::Point>& contour,
                                                 int numHarmonics = 8);
```

**算法步骤：**
1. 等弧长重采样 → 256 点
2. 分别对 x(t)、y(t) 做傅里叶级数展开
3. 每阶谐波产生 4 个系数 (a_n, b_n, c_n, d_n)
4. 以第 1 阶椭圆大小做尺度归一化

**维度选择理由：** 8 阶谐波 (32 系数) 足以区分绝大多数工业零件形状，同时保持计算高效。

**典型应用：** 叶片形状分析、生物形态学、复杂不规则轮廓重建。

---

### 3. `zernikeMoments` — Zernike 正交矩

```cpp
std::vector<double> zernikeMoments(const std::vector<cv::Point>& contour,
                                   int order = 6);
```

**算法步骤：**
1. 等弧长重采样 → 256 点
2. 以质心为原点，映射到单位圆
3. 计算 0~order 各阶 Zernike 矩 A_nm
4. 取各阶矩的幅值 |A_nm|（天然旋转不变）
5. 用 |Z_00| 做尺度归一化

**维度选择理由：** order=6 产生 28 个 (n,m) 对，但 n=0,m=0 归一化后恒为 1，实际有效 27 个。保留全部 28 个以保持完整；其中 3 个 (m≠0 的零值对) 实际为 0。有效非零维度约 25 个。这是 MPEG-7 形状描述符的标准选取。

**典型应用：** 旋转不变形状识别、商标检索、医学图像分析。

---

### 4. `centroidDistanceSignature` — 质心距离一维签名

```cpp
std::vector<double> centroidDistanceSignature(const std::vector<cv::Point>& contour,
                                              int numSamples = 64);
```

**算法步骤：**
1. 等弧长重采样 → `numSamples` 点
2. 计算质心
3. 每点到质心的欧氏距离
4. 均值归一化

**维度选择理由：** 64 点足够以 5.6° 分辨率重建绝大多数常见轮廓，维度适中。

**典型应用：** 形状的局部膨胀/收缩检测、简单形状的相似度比较。

---

### 5. `complexCoordinateSignature` — 复数坐标一维签名

```cpp
std::vector<double> complexCoordinateSignature(const std::vector<cv::Point>& contour,
                                               int numSamples = 64);
```

**算法步骤：**
1. 等弧长重采样 → `numSamples` 点
2. 以质心为原点
3. 最大距离归一化到单位圆
4. 返回 [Re(z₀), Im(z₀), Re(z₁), Im(z₁), …] 交替排列

**维度选择理由：** 64 复数点 × 2 分量 = 128 维，与质心距离签名采样分辨率对齐。

**典型应用：** 轮廓的完整重建、需要保留相位信息的场景。

---

### 6. `tangentAngleSignature` — 切向角累积一维签名

```cpp
std::vector<double> tangentAngleSignature(const std::vector<cv::Point>& contour,
                                          int numSamples = 64);
```

**算法步骤：**
1. 等弧长重采样 → `numSamples` 点
2. 逐点计算切向角 θ(t) = atan2(dy, dx)
3. 相位展开 (unwrap) 得到累积切向角 Θ(t)
4. 去除线性趋势（闭合轮廓的恒定旋转分量）
5. 归一化到 [-1, 1]

**维度选择理由：** 64 点与质心距离签名匹配，适合联合使用。

**典型应用：** 角点检测、多边形近似评估、形状曲率突变检测。

---

### 7. `d1Distribution` — Osada D1 形状分布

```cpp
std::vector<double> d1Distribution(const std::vector<cv::Point>& contour,
                                   int numBins = 32);
```

**算法步骤：**
1. 等弧长重采样 → 256 点
2. 计算质心到每个采样点的距离
3. 最大距离归一化到 [0, 1]
4. 统计 32-bin 直方图
5. 转为概率密度分布

**维度选择理由：** 32 bins 是 Osada 原论文推荐值，在平滑性和分辨力之间最佳平衡。

**典型应用：** 基于统计的形状检索、对局部形变鲁棒的匹配。

---

### 8. `d2Distribution` — Osada D2 形状分布

```cpp
std::vector<double> d2Distribution(const std::vector<cv::Point>& contour,
                                   int numBins = 32);
```

**算法步骤：**
1. 等弧长重采样 → 256 点
2. 随机采样 5000 对轮廓点
3. 计算每对点之间的欧氏距离
4. 最大距离归一化到 [0, 1]
5. 统计 32-bin 直方图
6. 转为概率密度分布

**维度选择理由：** 32 bins，与 D1 对齐；5000 对采样保证统计显著性，固定随机种子 (seed=42) 确保可重复。

**典型应用：** 对遮挡、噪声鲁棒的形状检索，3D 物体二维投影匹配。

---

### 9. `curvatureProfile` — 离散曲率轮廓

```cpp
std::vector<double> curvatureProfile(const std::vector<cv::Point>& contour,
                                     int numSamples = 64);
```

**算法步骤：**
1. 等弧长重采样 → `numSamples` 点
2. 对每个采样点，用前后 k=3 邻点构造两向量
3. 计算离散曲率 κ ≈ Δθ / Δs（夹角 / 弧长）
4. 最大曲率归一化

**维度选择理由：** 64 点在曲率分辨率和噪声平滑之间取得最佳平衡，k=3 提供适度的局部平滑。

**典型应用：** 局部形状特征、凹陷/凸起检测、表面缺陷分析。

---

### 10. `featureDistance` — 归一化欧氏距离

```cpp
double featureDistance(const std::vector<double>& v1,
                       const std::vector<double>& v2);
```

**原理：** 先将两向量 L2 归一化为单位向量 u=v1/|v1|, w=v2/|v2|，再计算 `d = |u - w| / 2`，值域 **[0, 1]**。

- d = 0 → 完全相同
- d = 1 → 正交或方向完全相反

与 `featureCosineSimilarity` 满足互补关系：**sim + d² = 1**。对特征向量的整体尺度不敏感。

**用途：** 衡量两个特征向量的差异程度，值越小越相似。要求两向量维度相同。

---

### 11. `featureCosineSimilarity` — 归一化余弦相似度

```cpp
double featureCosineSimilarity(const std::vector<double>& v1,
                               const std::vector<double>& v2);
```

**原理：** cos(θ) = (v1 · v2) / (|v1| × |v2|)，映射为 `sim = (cosθ + 1) / 2`，值域 **[0, 1]**。

- sim = 0 → 方向完全相反
- sim = 0.5 → 正交
- sim = 1 → 完全相同

与 `featureDistance` 满足互补关系：**sim + d² = 1**。对特征向量的整体尺度不敏感。

**用途：** 衡量两个特征向量的方向一致性。越接近 1 越相似。要求两向量维度相同。

---

## 与 ContourUtils 的定位对比

| 特性 | `ContourUtils` | `ContourHighOrder` |
|------|---------------|-------------------|
| 目标 | 基础几何属性、形状评分 | 高阶特征向量化 |
| 输出 | 标量 (面积、圆度等) | `std::vector<double>` |
| 方法 | 几何公式、Hu 矩 | 频域 DFT、正交矩、签名、分布 |
| 典型用途 | 单轮廓属性查询、形状分类判断 | 多轮廓匹配、机器学习输入 |
| 依赖 | 仅 OpenCV | 仅 OpenCV（不依赖 ContourUtils） |

两者互补：先用 `ContourUtils` 判断形状类型，再用 `ContourHighOrder` 提取特征向量做精细匹配或送入分类器。

---

## 使用建议

| 场景 | 推荐特征 |
|------|---------|
| 通用形状识别 | `fourierDescriptors` (10 维，轻量快速) |
| 旋转不变严格要求 | `zernikeMoments` (25 维) |
| 复杂不规则轮廓 | `ellipticalFourierDescriptors` (32 维) |
| 形状检索 (对遮挡鲁棒) | `d2Distribution` (32 维) |
| 局部形变检测 | `centroidDistanceSignature` + `curvatureProfile` |
| 组合特征 (高精度) | 拼接 `fourierDescriptors` + `zernikeMoments` + `d2Distribution` |