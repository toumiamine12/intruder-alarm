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

void test_bg_learns() {
    cv::Mat f(60, 80, CV_8UC1, cv::Scalar(90));
    BackgroundModel bg;
    for (int i = 0; i < 100; ++i) bg.update(f, cv::Mat());
    double mx; cv::minMaxLoc(bg.diff(f), nullptr, &mx);
    assert(mx <= 1);
}

void test_bg_skip_mask() {
    cv::Mat black(60, 80, CV_8UC1, cv::Scalar(0)), white(60, 80, CV_8UC1, cv::Scalar(255));
    BackgroundModel bg(0.5f);
    bg.update(black, cv::Mat());
    cv::Rect box(10, 10, 20, 20);
    cv::Mat skip(60, 80, CV_8UC1, cv::Scalar(0)); skip(box).setTo(255);
    bg.update(white, skip);
    cv::Mat d = bg.diff(white);
    assert(d.at<uchar>(15, 15) == 255);   // inside box: not learned
    assert(d.at<uchar>(50, 70) < 255);    // outside: learned
}

void test_bg_reinit_on_size_change() {
    BackgroundModel bg;
    bg.update(cv::Mat(60, 80, CV_8UC1, cv::Scalar(10)), cv::Mat());
    cv::Mat big(120, 160, CV_8UC1, cv::Scalar(10));
    bg.update(big, cv::Mat());
    assert(bg.diff(big).size() == big.size());
}

void test_skip_mask_light_change() {
    cv::Size s(80, 60);
    assert(makeSkipMask(s, cv::Rect(0, 0, 80, 60)).empty());
    assert(makeSkipMask(s, cv::Rect()).empty());
    cv::Mat m = makeSkipMask(s, cv::Rect(10, 10, 20, 20));
    assert(m.size() == s && m.at<uchar>(15, 15) == 255 && m.at<uchar>(0, 0) == 0);
}

void test_normalize_zone() {
    cv::Size s(640, 480);
    assert(normalizeZone({300, 200}, {100, 50}, s) == cv::Rect(100, 50, 200, 150));
    assert(normalizeZone({-20, -20}, {700, 900}, s) == cv::Rect(0, 0, 640, 480));
    assert(normalizeZone({5, 5}, {5, 100}, s).empty());
}

int main() {
    test_histogram_counts();
    test_otsu_two_groups();
    test_otsu_matches_opencv();
    test_otsu_uniform();
    test_threshold_mask();
    test_bg_learns();
    test_bg_skip_mask();
    test_bg_reinit_on_size_change();
    test_skip_mask_light_change();
    test_normalize_zone();
    std::puts("all tests passed");
}
