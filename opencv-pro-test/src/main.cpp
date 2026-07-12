#include <iostream>

#include <opencv2/core/version.hpp>

#include "ProcessLiquidFlow.hpp"

int main()
{
    std::cout << "opencv-pro-test" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    processLiquidFlows();

    return 0;
}
