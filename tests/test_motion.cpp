#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <opencv2/imgproc.hpp>
#include "motion.hpp"

void test_histogram_counts() {
    cv::Mat img(10, 10, CV_8UC1, cv::Scalar(7));
    img(cv::Rect(0, 0, 10, 3)).setTo(200);
    auto h = histogram(img);
    assert(h[7] == 70 && h[200] == 30);
}

void test_otsu_two_groups() {
    cv::Mat img(100, 100, CV_8UC1, cv::Scalar(20));
    img(cv::Rect(0, 0, 50, 100)).setTo(200);
    int T = otsuThreshold(img);
    assert(T >= 20 && T < 200);
}

void test_otsu_matches_opencv() {
    cv::Mat img(240, 320, CV_8UC1);
    cv::RNG rng(42);
    rng.fill(img, cv::RNG::NORMAL, 60, 15);
    img(cv::Rect(100, 50, 120, 100)) += 90;  // bright object
    cv::Mat dummy;
    int cvT = (int)cv::threshold(img, dummy, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    assert(std::abs(otsuThreshold(img) - cvT) <= 1);
}

void test_otsu_uniform() {
    cv::Mat zeros(50, 50, CV_8UC1, cv::Scalar(0));
    int T = otsuThreshold(zeros);
    assert(T >= 0 && T <= 255);
    cv::Mat flat(50, 50, CV_8UC1, cv::Scalar(128));
    T = otsuThreshold(flat);
    assert(T >= 0 && T <= 255);
}

void test_threshold_mask() {
    cv::Mat img(1, 3, CV_8UC1);
    img.at<uchar>(0, 0) = 10; img.at<uchar>(0, 1) = 50; img.at<uchar>(0, 2) = 51;
    cv::Mat m = thresholdMask(img, 50);
    assert(m.at<uchar>(0, 0) == 0 && m.at<uchar>(0, 1) == 0 && m.at<uchar>(0, 2) == 255);
}

int main() {
    test_histogram_counts();
    test_otsu_two_groups();
    test_otsu_matches_opencv();
    test_otsu_uniform();
    test_threshold_mask();
    std::puts("all tests passed");
}
