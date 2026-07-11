#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ContourUtils.hpp"


//S4 - 检测倾斜液流
void processLiquidFlow()
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

        if (image.empty()) {
            std::cerr << "无法加载图像: " << path << std::endl;
            continue;
        }

        //生成灰度图 & 双边滤波
        cv::Mat grayImage,bilateraImage,binaryImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        cv::bilateralFilter(grayImage, bilateraImage, 9, 75, 75);

        //二值化
        cv::threshold(bilateraImage, binaryImage, 210, 255, cv::THRESH_BINARY);

        //基于binaryImage，使用连通域分析方法connectedComponentsWithStats，获取每个连通域的面积和中心坐标，外接矩形
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
            
            //计算当前blob与圆形的相似度,与长方形的相似度.
            double circularity = 0.0;
            double rectangularity = 0.0;
            double curveCircleSim  = 0.0;  // 曲率法: 圆形相似度
            double curveRectSim    = 0.0;  // 曲率法: 矩形相似度
            int shapeType = 0;  // 0=未知, 1=圆形, 2=矩形
            if (!contours.empty()) {
                const auto& ct = contours[0];

                // ---- 方法 A: 全局几何特征 ----
                circularity   = ContourUtils::circleSimilarity(ct);
                rectangularity = ContourUtils::rectangleSimilarity(ct);

                // ---- 方法 B: 轮廓曲率特征 ----
                curveCircleSim = ContourUtils::circleSimilarityByCurvature(ct, /*k=*/5);
                curveRectSim   = ContourUtils::rectangleSimilarityByCurvature(ct, /*k=*/5);

                // ---- 综合判断: 同时参考两种方法 ----
                constexpr double CIRCLE_THRESHOLD = 1.25;
                constexpr double RECT_THRESHOLD   = 0.80;

                if (rectangularity > 1e-9) {
                    double ratio = circularity / rectangularity;

                    bool isCircleByGlobal = (ratio >= CIRCLE_THRESHOLD);
                    bool isRectByGlobal   = (ratio <= RECT_THRESHOLD);

                    // 曲率法确认: 圆形曲率评分应 > 0.75, 矩形曲率评分应 > 0.60
                    bool isCircleByCurve  = (curveCircleSim > 0.75);
                    bool isRectByCurve    = (curveRectSim   > 0.60);

                    if (isCircleByGlobal && isCircleByCurve) {
                        shapeType = 1;  // 圆形 (两种方法一致)
                    } else if (isRectByGlobal && isRectByCurve) {
                        shapeType = 2;  // 矩形 (两种方法一致)
                    } else if (isCircleByGlobal) {
                        shapeType = 1;  // 圆形 (仅全局方法, 曲率不确认但全局信号强)
                    } else if (isRectByGlobal) {
                        shapeType = 2;  // 矩形 (仅全局方法)
                    }
                    // 若 ratio 在中间区间 (0.80~1.25), 保持未知
                }
            }
            
            if (shapeType==1)
            {
                continue;
            }
        
        }

        // TODO: 在此处对 image 进行液流检测处理

        continue;
    }
}
