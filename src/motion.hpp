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

// Running-average background: bg = (1 - alpha) * bg + alpha * gray, per pixel.
class BackgroundModel {
public:
    explicit BackgroundModel(float alpha = 0.02f);
    // First call (or a frame size change) sets bg = gray.
    // skipMask: CV_8UC1, non-zero = don't update that pixel. May be empty.
    void update(const cv::Mat& gray, const cv::Mat& skipMask);
    // |gray - bg| as CV_8UC1. Requires update() to have run once.
    cv::Mat diff(const cv::Mat& gray) const;
    bool initialized() const;

private:
    float alpha_;
    cv::Mat bg_;  // CV_32FC1
};

// Mask for BackgroundModel::update. Empty (= learn everywhere) when the box is
// empty or covers more than half the frame, e.g. a light switched on.
cv::Mat makeSkipMask(cv::Size frame, const cv::Rect& box);

// Rectangle spanned by a drag from a to b in any direction, clipped to the
// frame. Empty if narrower or shorter than 2 px.
cv::Rect normalizeZone(cv::Point a, cv::Point b, cv::Size frame);
