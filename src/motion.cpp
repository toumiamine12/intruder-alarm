#include "motion.hpp"

std::array<int, 256> histogram(const cv::Mat& gray) {
    CV_Assert(gray.type() == CV_8UC1);
    std::array<int, 256> h{};
    for (int r = 0; r < gray.rows; ++r) {
        const uchar* row = gray.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
            h[row[c]]++;
    }
    return h;
}

int otsuThreshold(const cv::Mat& gray) {
    const auto h = histogram(gray);
    const double total = static_cast<double>(gray.total());

    double sumAll = 0;
    for (int g = 0; g < 256; ++g) sumAll += g * static_cast<double>(h[g]);

    double w0 = 0, sum0 = 0, bestVar = -1;
    int bestT = -1;
    for (int t = 0; t < 256; ++t) {
        w0 += h[t];                 // pixels <= t (background)
        sum0 += t * static_cast<double>(h[t]);
        const double w1 = total - w0;
        if (w0 == 0 || w1 == 0) continue;
        const double mu0 = sum0 / w0;
        const double mu1 = (sumAll - sum0) / w1;
        const double betweenVar = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
        if (betweenVar > bestVar) {
            bestVar = betweenVar;
            bestT = t;
        }
    }
    if (bestT < 0)                  // uniform image: no split exists
        bestT = static_cast<int>(sumAll / std::max(total, 1.0));
    return bestT;
}

cv::Mat thresholdMask(const cv::Mat& src, int T) {
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat dst(src.size(), CV_8UC1);
    for (int r = 0; r < src.rows; ++r) {
        const uchar* in = src.ptr<uchar>(r);
        uchar* out = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c)
            out[c] = in[c] > T ? 255 : 0;
    }
    return dst;
}
