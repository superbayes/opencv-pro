#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ContourUtils.hpp"
#include "ContourHighOrderFeatures.hpp"


// findTwoValleys: 在 reduceMat（1×N 投影向量）中寻找两个谷底（局部极小值）
// 搜索范围: 中间 50% 列区域
// 约束: 两谷间距 ∈ (10, 60) 像素
// 返回: cv::Vec<float,5>{idx1, val1, idx2, val2, angle}, idx1 < idx2;
//       失败返回 {-1, 0, -1, 0, angle}
inline cv::Vec<float,5> findTwoValleys(const cv::Mat& reduceMat, double angle)
{
    int N = reduceMat.cols;
    if (N < 4) return cv::Vec<float,5>(-1, 0, -1, 0, (float)angle);

    // 1. 限定中间50%搜索范围
    int scanLeft  = N / 4;
    int scanRight = N * 3 / 4;
    if (scanRight <= scanLeft + 1) return cv::Vec<float,5>(-1, 0, -1, 0, (float)angle);

    const uchar* row = reduceMat.ptr<uchar>(0);

    // 2. 收集扫描范围内所有局部极小值 (idx, val)
    std::vector<std::pair<int, int>> minima; // {idx, val}
    for (int i = scanLeft + 1; i < scanRight - 1; ++i)
    {
        uchar v = row[i];
        if (v <= row[i - 1] && v <= row[i + 1])
        {
            // 跳过平坦区重复: 若与前一个极小值相邻且同值则跳过
            if (!minima.empty() && minima.back().first == i - 1 && minima.back().second == v)
                continue;
            minima.push_back({i, (int)v});
        }
    }

    if (minima.size() < 2) return cv::Vec<float,5>(-1, 0, -1, 0, (float)angle);

    // 3. 找到全局最低谷 (valley1)
    int best1 = 0; // 在 minima 中的索引
    for (size_t k = 1; k < minima.size(); ++k)
    {
        if (minima[k].second < minima[best1].second)
            best1 = (int)k;
    }

    int idx1 = minima[best1].first;
    int val1 = minima[best1].second;

    // 4. 在 valley1 附近找次低谷: 距离 ∈ (10, 100), 值最小者
    int best2 = -1;
    int best2_val = 256;
    for (size_t k = 0; k < minima.size(); ++k)
    {
        if ((int)k == best1) continue;
        int dist = std::abs(minima[k].first - idx1);
        if (dist > 10 && dist < 60)
        {
            if (minima[k].second < best2_val)
            {
                best2_val = minima[k].second;
                best2 = (int)k;
            }
        }
    }

    if (best2 < 0) return cv::Vec<float,5>(-1, 0, -1, 0, (float)angle);

    int idx2 = minima[best2].first;
    int val2 = minima[best2].second;

    // 5. 按列索引升序返回
    if (idx1 < idx2)
        return cv::Vec<float,5>((float)idx1, (float)val1, (float)idx2, (float)val2, (float)angle);
    else
        return cv::Vec<float,5>((float)idx2, (float)val2, (float)idx1, (float)val1, (float)angle);
}

// processLiquidFlowImage: 对单张图像执行液流检测全流程
// 输入: BGR 彩色图像 Mat
// 处理: 灰度化→双边滤波→二值化→连通域分析→D1特征匹配→旋转扫描寻谷→可视化
inline void processLiquidFlowImage(const cv::Mat& image)
{
    std::ostringstream diag_log;
    if (!image.empty())
        diag_log << "===== processLiquidFlowImage Diagnostic =====\n"
                 << "image_size  = " << image.cols << "x" << image.rows << "\n";
    else {
        diag_log << "===== processLiquidFlowImage Diagnostic =====\n"
                 << "image_size  = (empty)\n"
                 << "STATUS: FAIL - empty input image\n";
        std::cerr << diag_log.str() << std::endl;
        return;
    }
    try
    {
        //生成灰度图 & 双边滤波
        cv::Mat grayImage,bilateraImage,binaryImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        cv::bilateralFilter(grayImage, bilateraImage, 9, 75, 75);

        //二值化
        cv::threshold(bilateraImage, binaryImage, 210, 255, cv::THRESH_BINARY);

        //基于binaryImage，使用连通域分析方法connectedComponentsWithStats，获取每个连通域的面积和中心坐标，外接矩形
        cv::Rect target_rect;
        cv::Mat labels,stats, centroids;
        int numComponents =
            cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids, 8);
        for (int idx = 1; idx < numComponents; idx++)
        {
            //获取stats[idx]的x,y,w,h,面积
            int x      = stats.at<int>(idx, cv::CC_STAT_LEFT);
            int y      = stats.at<int>(idx, cv::CC_STAT_TOP);
            int w      = stats.at<int>(idx, cv::CC_STAT_WIDTH);
            int h      = stats.at<int>(idx, cv::CC_STAT_HEIGHT);
            int area   = stats.at<int>(idx, cv::CC_STAT_AREA);

            if (area<800)
            {
                continue;
            }

            //计算当前blob的外层轮廓,依据labels
            cv::Mat blobMask = (labels == idx);
            blobMask.convertTo(blobMask, CV_8UC1);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(blobMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            //先对contours[0]进行简化,移除掉毛刺, 再获取凸包轮廓,再计算凸包的d1Distribution特征向量
            double sim = 0;
            cv::Mat visImage;
            std::vector<double> d1Feat;
            if (!contours.empty())
            {
                // a) 多边形逼近去毛刺 (epsilon = 0.5% 周长)
                std::vector<cv::Point> simplified = ContourUtils::approxPolyDP(contours[0], 0.005);

                // b) 获取凸包
                std::vector<cv::Point> hull = ContourUtils::convexHull(simplified);

                // c) 计算凸包的 D1 分布特征向量 (32 维直方图)
                d1Feat = ContourHighOrder::d1Distribution(hull,16);
                diag_log << "d1Feat      = ";
                for (size_t i = 0; i < d1Feat.size(); ++i) {
                    diag_log << std::fixed << std::setprecision(5) << d1Feat[i];
                    if (i + 1 < d1Feat.size()) diag_log << ", ";
                }
                diag_log << "\n";

                //创建D1 特征向量模板(16维)，用于储存模板向量
                std::vector<double> d1Template = { 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.30078, 0.19141, 0.17188, 0.22656, 0.10938 };
                //std::vector<double> d1Template = { 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.06641, 0.14062, 0.08203, 0.07422, 0.06250, 0.06250, 0.05469, 0.09375, 0.36328 };

                //计算d1Feat与d1Template01，d1Template02的余弦相似度
                sim = ContourHighOrder::featureCosineSimilarity(d1Feat, d1Template);
                diag_log << "[CosineSimilarity] Template01=" << std::fixed << std::setprecision(5) << sim << "\n";

                // d) 可视化: 在 blobMask 拷贝上绘制简化轮廓(蓝色)和凸包轮廓(红色)
                cv::cvtColor(blobMask, visImage, cv::COLOR_GRAY2BGR);
                cv::drawContours(visImage, std::vector<std::vector<cv::Point>>{simplified},
                                 -1, cv::Scalar(255, 0, 0), 2);   // 蓝色: 简化轮廓
                cv::drawContours(visImage, std::vector<std::vector<cv::Point>>{hull},
                                 -1, cv::Scalar(0, 0, 255), 2);   // 红色: 凸包轮廓
            }

            
            if (sim > 0.85)
            {
                target_rect = cv::Rect(x, y, w, h);
            }
            continue;
        }

        diag_log << "components  = " << numComponents << "\n";
        if (target_rect.area() > 0)
            diag_log << "target_rect = [x=" << target_rect.x << ", y=" << target_rect.y
                     << ", w=" << target_rect.width << ", h=" << target_rect.height << "]\n";
        else
            diag_log << "target_rect = (none matched, using full image)\n";

        //基于target_rect，检测倾斜的液流
        if (target_rect.area() == 0)
        {
            target_rect = cv::Rect(0, 0, binaryImage.cols, binaryImage.rows);
        }
        //可视化
        //cv::rectangle(grayImage, target_rect, cv::Scalar(255, 200, 0), 5);

        ///////////////////////////////////////////////////////////////////////////
        //开始计算roi区域中，液流特征
        cv::Mat roi_gray = grayImage(target_rect).clone();
        std::vector<cv::Vec<float,5>> valleyList;
        //在一个for循环中,对roi_gray,进行旋转,绕着中心点旋转,角度从-5到5,步长为0.25度,
        float best_angle = 0.0, best_idx1=0, best_idx2 = 0;
        cv::Mat best_rotated;
        double best_avg = std::numeric_limits<double>::max();

        cv::Point2f center(roi_gray.cols / 2.0f, roi_gray.rows / 2.0f);

        for (double angle = -5; angle <= 5.0; angle += 0.25)
        {
            if (angle==0)
            {
                continue;
            }
            // 计算旋转矩阵，绕roi中心旋转
            cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

            // 执行旋转变换
            cv::Mat rotated, rotated_bin;
            cv::warpAffine(roi_gray, rotated, rot_mat, roi_gray.size(),
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

            cv::adaptiveThreshold(rotated, rotated_bin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 7, 2);

            //向y向投影
            cv::Mat reduceMat;
            cv::reduce(rotated_bin, reduceMat, 0, cv::REDUCE_AVG, CV_8UC1);

            // 基于 reduceMat 寻峰: 找两个谷底（液柱间隙）
            cv::Vec<float,5> valleys = findTwoValleys(reduceMat, angle);
            valleyList.push_back(valleys);

            // 寻找最佳旋转角度,基于两个谷底的平均值(排除-1)最小, 确定best_angle,best_rotated
            if (valleys[0] >= 0 && valleys[2] >= 0)
            {
                double avgValley = (static_cast<double>(valleys[1]) + valleys[3]) * 0.5;
                if (avgValley < best_avg)
                {
                    best_avg     = avgValley;
                    best_angle   = angle;
                    best_idx1 = valleys[0] + target_rect.x;
                    best_idx2 = valleys[2] + target_rect.x;
                    best_rotated = rotated_bin.clone();
                    target_rect;
                }
            }
            continue;
        }

        diag_log << "valley_cnt  = " << valleyList.size() << "\n";
        diag_log << "best_avg    = " << best_avg << "\n";
        diag_log << "best_angle  = " << best_angle << " deg\n";
        diag_log << "best_idx1   = " << best_idx1 << "\n";
        diag_log << "best_idx2   = " << best_idx2 << "\n";
        if (!best_rotated.empty())
            diag_log << "best_rotated= " << best_rotated.rows << "x" << best_rotated.cols << "\n";
        else
            diag_log << "best_rotated= (empty)\n";

        //可视化,将best_idx1,best_idx2在grayImage上绘制两条竖线,颜色为绿色,线宽为1
        if (best_idx1 > 0 && best_idx2 > 0)
        {
            cv::line(grayImage, cv::Point(static_cast<int>(best_idx1), 0),
                     cv::Point(static_cast<int>(best_idx1), grayImage.rows - 1),
                     cv::Scalar(0, 255, 0), 1);
            cv::line(grayImage, cv::Point(static_cast<int>(best_idx2), 0),
                     cv::Point(static_cast<int>(best_idx2), grayImage.rows - 1),
                     cv::Scalar(0, 255, 0), 1);
        }

        diag_log << "STATUS: SUCCESS\n";
        std::cout << diag_log.str() << std::endl;
    }
    catch (const cv::Exception& e)
    {
        diag_log << "STATUS: FAIL - OpenCV Exception\n";
        diag_log << "  err_code = " << e.code << "\n";
        diag_log << "  func     = " << e.func << "\n";
        diag_log << "  file     = " << e.file << "\n";
        diag_log << "  line     = " << e.line << "\n";
        diag_log << "  msg      = " << e.what() << "\n";
        std::cerr << diag_log.str() << std::endl;
    }
    catch (const std::exception& e)
    {
        diag_log << "STATUS: FAIL - std::exception\n";
        diag_log << "  msg = " << e.what() << "\n";
        std::cerr << diag_log.str() << std::endl;
    }
    catch (...)
    {
        diag_log << "STATUS: FAIL - Unknown exception\n";
        std::cerr << diag_log.str() << std::endl;
    }
}

//S4 - 检测倾斜液流（批量入口）
void processLiquidFlows()
{
    std::string imageDirectory = R"(E:\01wk\vs2026\opencv-pro\image\*.jpg)";
    std::vector<cv::String> imagePaths;
    cv::glob(imageDirectory, imagePaths, false);

    if (imagePaths.empty()) {
        std::cerr << "未找到匹配 image/*.jpg 的图像文件 \n" << std::endl;
        return;
    }

    for (const auto& path : imagePaths)
    {
        cv::Mat image = cv::imread(path);
        processLiquidFlowImage(image);
    }
}
