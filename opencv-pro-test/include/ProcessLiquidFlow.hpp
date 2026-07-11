#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ContourUtils.hpp"
#include "ContourHighOrderFeatures.hpp"


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
                // 打印 D1 特征向量到控制台 (逗号分隔, 保留5位小数)
                //std::cout << "[D1] ";
                //for (size_t i = 0; i < d1Feat.size(); ++i) {
                //    std::cout << std::fixed << std::setprecision(5) << d1Feat[i];
                //    if (i + 1 < d1Feat.size()) std::cout << ", ";
                //}
                //std::cout << "\n";

                //创建D1 特征向量模板(16维)，用于储存模板向量
                std::vector<double> d1Template = { 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.30078, 0.19141, 0.17188, 0.22656, 0.10938 };
                //std::vector<double> d1Template = { 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.06641, 0.14062, 0.08203, 0.07422, 0.06250, 0.06250, 0.05469, 0.09375, 0.36328 };

                //计算d1Feat与d1Template01，d1Template02的余弦相似度
                sim = ContourHighOrder::featureCosineSimilarity(d1Feat, d1Template);
                std::cout << "[CosineSimilarity] Template01=" << std::fixed << std::setprecision(5) << sim << std::endl;

                // d) 可视化: 在 blobMask 拷贝上绘制简化轮廓(蓝色)和凸包轮廓(红色)
                cv::cvtColor(blobMask, visImage, cv::COLOR_GRAY2BGR);
                cv::drawContours(visImage, std::vector<std::vector<cv::Point>>{simplified},
                                 -1, cv::Scalar(255, 0, 0), 2);   // 蓝色: 简化轮廓
                cv::drawContours(visImage, std::vector<std::vector<cv::Point>>{hull},
                                 -1, cv::Scalar(0, 0, 255), 2);   // 红色: 凸包轮廓
            }

            
            if (sim > 0.85)
            {
                target_rect = cv::Rect(x, y, w, binaryImage.rows - y);
            }
            continue;
        }

        //基于target_rect，检测倾斜的液流
        if (target_rect.area() == 0)
        {
            target_rect = cv::Rect(0, 0, binaryImage.cols, binaryImage.rows);
        }
        //可视化
		cv::rectangle(grayImage, target_rect, cv::Scalar(255, 200, 0), 5);

        ///////////////////////////////////////////////////////////////////////////
        //开始计算roi区域中，液流特征
        cv::Mat roi_gray = grayImage(target_rect).clone();

        //对roi_gray进行增强，让亮的地方更亮，暗的地方更暗



        continue;
    }
}