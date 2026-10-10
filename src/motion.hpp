#pragma once
#include <array>
#include <opencv2/core.hpp>

// 256-bin histogram of a CV_8UC1 image.
std::array<int, 256> histogram(const cv::Mat& gray);

// Otsu's method: the T in [0, 255] that maximizes between-class variance.
// Foreground is defined as pixel > T.
int otsuThreshold(const cv::Mat& gray);

// CV_8UC1 mask: 255 where src > T, else 0.
cv::Mat thresholdMask(const cv::Mat& src, int T);
