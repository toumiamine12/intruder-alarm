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

BackgroundModel::BackgroundModel(float alpha) : alpha_(alpha) {}

bool BackgroundModel::initialized() const { return !bg_.empty(); }

void BackgroundModel::update(const cv::Mat& gray, const cv::Mat& slowMask) {
    CV_Assert(gray.type() == CV_8UC1);
    if (bg_.size() != gray.size()) {
        gray.convertTo(bg_, CV_32FC1);
        return;
    }
    const bool useMask = !slowMask.empty();
    const float slowAlpha = alpha_ * 0.1f;
    if (useMask) CV_Assert(slowMask.type() == CV_8UC1 && slowMask.size() == gray.size());

    for (int r = 0; r < gray.rows; ++r) {
        const uchar* in = gray.ptr<uchar>(r);
        const uchar* slow = useMask ? slowMask.ptr<uchar>(r) : nullptr;
        float* b = bg_.ptr<float>(r);
        for (int c = 0; c < gray.cols; ++c) {
            const float a = (slow && slow[c]) ? slowAlpha : alpha_;
            b[c] = (1.0f - a) * b[c] + a * in[c];
        }
    }
}

cv::Mat BackgroundModel::diff(const cv::Mat& gray) const {
    CV_Assert(initialized() && gray.type() == CV_8UC1 && gray.size() == bg_.size());
    cv::Mat d(gray.size(), CV_8UC1);
    for (int r = 0; r < gray.rows; ++r) {
        const uchar* in = gray.ptr<uchar>(r);
        const float* b = bg_.ptr<float>(r);
        uchar* out = d.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
            out[c] = cv::saturate_cast<uchar>(std::abs(in[c] - b[c]));
    }
    return d;
}

cv::Mat makeSlowMask(cv::Size frame, const cv::Rect& box) {
    if (box.empty() || box.area() > 0.5 * frame.area()) return {};
    cv::Mat m(frame, CV_8UC1, cv::Scalar(0));
    m(box & cv::Rect(cv::Point(), frame)).setTo(255);
    return m;
}

cv::Rect normalizeZone(cv::Point a, cv::Point b, cv::Size frame) {
    cv::Rect r(cv::Point(std::min(a.x, b.x), std::min(a.y, b.y)),
               cv::Point(std::max(a.x, b.x), std::max(a.y, b.y)));
    r &= cv::Rect(cv::Point(), frame);
    if (r.width < 2 || r.height < 2) return {};
    return r;
}

MotionDebouncer::MotionDebouncer(int onFrames, int offFrames)
    : onFrames_(onFrames), offFrames_(offFrames) {}

bool MotionDebouncer::push(bool motion) {
    if (motion) {
        ++motionRun_;
        quietRun_ = 0;
        if (motionRun_ >= onFrames_) on_ = true;
    } else {
        ++quietRun_;
        motionRun_ = 0;
        if (quietRun_ >= offFrames_) on_ = false;
    }
    return on_;
}

bool MotionDebouncer::on() const { return on_; }
