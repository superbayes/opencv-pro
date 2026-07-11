#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


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

            //计算当前blob的外层轮廓,依据labels
            cv::Mat blobMask = (labels == idx);
            blobMask.convertTo(blobMask, CV_8UC1);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(blobMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
        
        }

        // TODO: 在此处对 image 进行液流检测处理

        continue;
    }
}
