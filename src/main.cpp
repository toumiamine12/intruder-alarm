#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "motion.hpp"

namespace {

// Tuning knobs
constexpr float kAlpha = 0.02f;       // background learning rate
constexpr int kMinThreshold = 25;     // floor for Otsu so pure noise isn't "motion"
constexpr double kMinArea = 1500;     // px; smaller blobs are ignored
constexpr int kOnFrames = 5;          // motion frames in a row before alarm
constexpr int kOffFrames = 15;        // quiet frames in a row before alarm clears
constexpr int kWarmupFrames = 30;     // let the background settle at startup
constexpr int kMaxEmptyFrames = 30;   // give up if the camera stops delivering
constexpr double kSnapshotCooldownSec = 3.0;

const char* const kMainWindow = "intruder-alarm";
const char* const kDebugWindows[] = {"difference", "mask", "histogram"};

// Mouse drag state for drawing the detection zone.
struct ZoneDrag {
    bool dragging = false;
    cv::Point start, current;
    bool finished = false;  // a drag just ended; main loop applies it
};

void onMouse(int event, int x, int y, int /*flags*/, void* userdata) {
    auto* drag = static_cast<ZoneDrag*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN) {
        drag->dragging = true;
        drag->start = drag->current = {x, y};
    } else if (event == cv::EVENT_MOUSEMOVE && drag->dragging) {
        drag->current = {x, y};
    } else if (event == cv::EVENT_LBUTTONUP && drag->dragging) {
        drag->dragging = false;
        drag->current = {x, y};
        drag->finished = true;
    }
}

// Plays the desktop bell without blocking; terminal bell if paplay fails.
void beep() {
    std::system("(paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null || printf '\\a') &");
}

std::string timestampName() {
    const std::time_t t = std::time(nullptr);
    std::ostringstream name;
    name << "snapshots/" << std::put_time(std::localtime(&t), "%Y-%m-%d_%H-%M-%S") << ".png";
    return name.str();
}

// Histogram of the difference image, log-scaled (bin 0 dwarfs everything else),
// with the threshold T as a vertical line.
cv::Mat drawHistogram(const cv::Mat& diff, int T) {
    const int width = 512, height = 400;
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto h = histogram(diff);
    const double maxLog = std::log1p(*std::max_element(h.begin(), h.end()));
    for (int g = 0; g < 256; ++g) {
        const int barH = cvRound(std::log1p(h[g]) / std::max(maxLog, 1.0) * (height - 20));
        cv::rectangle(img, cv::Point(g * 2, height - 1), cv::Point(g * 2 + 1, height - 1 - barH),
                      cv::Scalar(0, 255, 255), cv::FILLED);
    }
    cv::line(img, cv::Point(T * 2, 0), cv::Point(T * 2, height - 1), cv::Scalar(255, 0, 0), 2);
    cv::putText(img, "T = " + std::to_string(T), cv::Point(T * 2 + 6, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    return img;
}

}  // namespace

int main() {
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam (device 0)\n";
        return 1;
    }

    BackgroundModel background(kAlpha);
    MotionDebouncer debouncer(kOnFrames, kOffFrames);
    cv::Size bgSize;
    cv::Rect zone;
    bool debug = false;
    int frameNo = 0, emptyRun = 0;
    double fps = 0;
    auto lastTick = std::chrono::steady_clock::now();

    std::error_code ec;
    std::filesystem::create_directories("snapshots", ec);
    if (ec) std::cerr << "Warning: cannot create snapshots/: " << ec.message() << "\n";

    ZoneDrag drag;
    bool wasAlarmOn = false;
    auto lastSnapshot = std::chrono::steady_clock::time_point{};

    // Resizable window, scaled up from the 640x480 camera image; processing size is unchanged.
    cv::namedWindow(kMainWindow, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(kMainWindow, 1280, 960);
    cv::setMouseCallback(kMainWindow, onMouse, &drag);

    while (true) {
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty()) {
            if (++emptyRun >= kMaxEmptyFrames) {
                std::cerr << "Webcam stopped sending frames\n";
                return 1;
            }
            continue;
        }
        emptyRun = 0;

        const cv::Rect frameRect(cv::Point(), frame.size());
        zone &= frameRect;
        if (drag.finished) {
            drag.finished = false;
            const cv::Rect newZone = normalizeZone(drag.start, drag.current, frame.size());
            if (!newZone.empty()) zone = newZone;  // a click without a drag keeps the old zone
        }
        if (zone.empty()) zone = frameRect;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        // First frame, or the camera changed resolution: (re)learn the background.
        if (gray.size() != bgSize) {
            background.update(gray, cv::Mat());
            bgSize = gray.size();
            continue;
        }

        // Segmentation: difference -> automatic threshold -> cleanup
        const cv::Mat diff = background.diff(gray);
        const int T = std::max(otsuThreshold(diff), kMinThreshold);
        cv::Mat mask = thresholdMask(diff, T);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));

        cv::Mat inZone = cv::Mat::zeros(mask.size(), CV_8UC1);
        mask(zone).copyTo(inZone(zone));
        mask = inZone;

        // Region analysis: keep big blobs, box them all together
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Rect box;
        for (const auto& c : contours)
            if (cv::contourArea(c) >= kMinArea)
                box = box.empty() ? cv::boundingRect(c) : (box | cv::boundingRect(c));

        const bool warmingUp = ++frameNo <= kWarmupFrames;
        const bool alarmOn = debouncer.push(!warmingUp && !box.empty());

        // Don't let an intruder who stands still fade into the background
        background.update(gray, alarmOn ? makeSlowMask(gray.size(), box) : cv::Mat());

        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - lastTick).count();
        lastTick = now;
        if (dt > 0) fps = fps == 0 ? 1.0 / dt : 0.9 * fps + 0.1 / dt;

        // Overlay
        cv::rectangle(frame, zone, cv::Scalar(0, 255, 255), 2);
        if (drag.dragging)
            cv::rectangle(frame, drag.start, drag.current, cv::Scalar(0, 255, 255), 1);

        if (alarmOn) {
            cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 3);
            cv::rectangle(frame, cv::Rect(0, 0, frame.cols, 50), cv::Scalar(0, 0, 255), cv::FILLED);
            const std::string text = "INTRUDER";
            int baseline = 0;
            const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 1.4, 3, &baseline);
            cv::putText(frame, text, cv::Point((frame.cols - ts.width) / 2, 25 + ts.height / 2),
                        cv::FONT_HERSHEY_DUPLEX, 1.4, cv::Scalar(255, 255, 255), 3);
        }
        char status[64];
        std::snprintf(status, sizeof status, "%.0f fps  T=%d  %s", fps, T,
                      warmingUp ? "WARMING UP" : "ARMED");
        cv::putText(frame, status, cv::Point(10, frame.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);

        if (alarmOn && !wasAlarmOn) beep();
        wasAlarmOn = alarmOn;

        if (alarmOn && now - lastSnapshot >= std::chrono::duration<double>(kSnapshotCooldownSec)) {
            lastSnapshot = now;
            const std::string path = timestampName();
            if (!cv::imwrite(path, frame)) std::cerr << "Warning: could not save " << path << "\n";
        }

        cv::imshow(kMainWindow, frame);
        if (debug) {
            cv::imshow("difference", diff);
            cv::imshow("mask", mask);
            cv::imshow("histogram", drawHistogram(diff, T));
        }

        const int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') break;
        // Window closed with the X button (imshow would otherwise reopen it).
        // The Qt backend throws instead of returning 0 once the window is gone.
        try {
            if (cv::getWindowProperty(kMainWindow, cv::WND_PROP_VISIBLE) < 1) break;
        } catch (const cv::Exception&) {
            break;
        }
        if (key == 'r' || key == 'R') zone = frameRect;
        if (key == 'd' || key == 'D') {
            debug = !debug;
            if (!debug)
                for (const char* w : kDebugWindows) cv::destroyWindow(w);
        }
    }
    return 0;
}
