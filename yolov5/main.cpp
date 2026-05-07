/**
 * @file main.cpp
 * @brief Multi-card YOLOv5 object detection demo entry point. See ../README.md for workflow details.
 */
#include "yolo.h"

#include "argparse/argparse.hpp"
#include "logger.h"
#include "rpp_drv_api.h"
#include "rpp_runtime.h"
#include "utils.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// Default paths keep the demo runnable from the build directory without extra image arguments.
const char* DEFAULT_IMAGE_DIR = "image";
const char* DEFAULT_OUTPUT_DIR = "final_demo_result";

// Default YOLOv5 post-processing thresholds match the single-card sample behavior.
const float DEFAULT_SCORE_THRESHOLD = 0.2F;
const float DEFAULT_NMS_THRESHOLD = 0.4F;
const float DEFAULT_CONFIDENCE_THRESHOLD = 0.4F;

// Stable overlay colors make adjacent boxes easier to distinguish in saved preview images.
const std::vector<cv::Scalar> BOX_COLORS = {
    cv::Scalar(255, 255, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 0),
    cv::Scalar(255, 0, 255),
    cv::Scalar(0, 128, 255),
};

/**
 * @brief Convert uncaught RPP runtime failures into a readable process-level error.
 *
 * RPP can throw non-standard exceptions during card-memory allocation; this handler avoids
 * leaving users with only a core-dump message.
 */
void handle_unhandled_rpp_exception()
{
    std::cerr << "\nFatal RPP runtime exception during YOLOv5 execution. "
              << "This usually means the selected model/context could not be allocated on the card. "
              << "Try a smaller YOLOv5 export, a quantized model, fewer active devices, or freeing RPP card memory."
              << std::endl;
    std::_Exit(EXIT_FAILURE);
}

struct ImageInput {
    // Path shown in reports, kept relative to the selected input root when possible.
    std::string relative_path;
    // Absolute path used by OpenCV image loading inside worker threads.
    std::filesystem::path absolute_path;
};

struct Detection {
    // COCO/custom class index selected during YOLOv5 output decoding.
    int class_id{-1};
    // Objectness score kept for display and result files.
    float confidence{0.0F};
    // Bounding box clipped to the original image coordinate system.
    cv::Rect box;
};

struct DecodeConfig {
    // Number of classes after the [x,y,w,h,objectness] fields in each YOLO row.
    int class_count{0};
    // Maximum number of boxes retained after NMS; 0 means no explicit cap.
    int max_detections{300};
    float score_threshold{DEFAULT_SCORE_THRESHOLD};
    float confidence_threshold{DEFAULT_CONFIDENCE_THRESHOLD};
    float nms_threshold{DEFAULT_NMS_THRESHOLD};
};

struct RequestResult {
    // Set when a worker has filled this ordered request slot.
    bool processed{false};
    size_t request_index{0};
    int worker_index{-1};
    int device_id{-1};
    std::string image_relative_path;
    size_t detection_count{0};
    std::string detections;
    std::string overlay_path;
    double model_inference_ms{0.0};
    double wall_latency_ms{0.0};
};

struct DevicePerformance {
    // Logical worker index and physical RPP card id used by one worker thread.
    int worker_index{-1};
    int device_id{-1};
    size_t requests{0};
    size_t detections{0};
    double elapsed_sec{0.0};
    double throughput_ips{0.0};
    double average_wall_latency_ms{0.0};
    double average_model_ms{0.0};
};

struct ModelDimensionsSummary {
    // Filled once by the first initialized worker so the main thread can print model I/O shapes.
    bool available{false};
    std::string input;
    std::string output;
};

/**
 * @brief Return the embedded COCO-80 class labels used by standard YOLOv5 exports.
 * @return Reference to the static class-name list used for reports and overlays.
 */
const std::vector<std::string>& coco80_class_labels()
{
    static const std::vector<std::string> kLabels = {
        "person",        "bicycle",       "car",           "motorcycle",    "airplane",      "bus",           "train",         "truck",
        "boat",          "traffic light", "fire hydrant",  "stop sign",     "parking meter", "bench",         "bird",          "cat",
        "dog",           "horse",         "sheep",         "cow",           "elephant",      "bear",          "zebra",         "giraffe",
        "backpack",      "umbrella",      "handbag",       "tie",           "suitcase",      "frisbee",       "skis",          "snowboard",
        "sports ball",   "kite",          "baseball bat",  "baseball glove", "skateboard",   "surfboard",     "tennis racket", "bottle",
        "wine glass",    "cup",           "fork",          "knife",         "spoon",         "bowl",          "banana",        "apple",
        "sandwich",      "orange",        "broccoli",      "carrot",        "hot dog",       "pizza",         "donut",         "cake",
        "chair",         "couch",         "potted plant",  "bed",           "dining table",  "toilet",        "tv",            "laptop",
        "mouse",         "remote",        "keyboard",      "cell phone",    "microwave",     "oven",          "toaster",       "sink",
        "refrigerator",  "book",          "clock",         "vase",          "scissors",      "teddy bear",    "hair drier",    "toothbrush",
    };
    return kLabels;
}

/**
 * @brief Convert a string to lowercase for case-insensitive extension checks.
 * @param str Source string to normalize.
 * @return Lowercase copy of the input string.
 */
std::string to_lower(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

/**
 * @brief Trim leading and trailing ASCII whitespace from a string copy.
 * @param s Source string that may contain surrounding whitespace.
 * @return Trimmed string copy.
 */
std::string trim_copy(const std::string& s)
{
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

/**
 * @brief Format an RPP tensor dimension object for run-configuration output.
 * @param dims Binding dimensions reported by the RPP engine.
 * @return Human-readable shape such as "[3, 640, 640]".
 */
std::string dims_to_string(const infer1::Dims& dims)
{
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

/**
 * @brief Check whether a file extension is supported by the demo image loader.
 * @param path Filesystem path whose extension is inspected.
 * @return True when the file looks like a supported still image.
 */
bool is_supported_image_extension(const std::filesystem::path& path)
{
    const std::string ext = to_lower(path.extension().string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

/**
 * @brief Resolve a user path from cwd, build directory, or nearby source directories.
 * @param requested_path Path from the CLI or default configuration.
 * @return First existing candidate path, or the expanded input path when unresolved.
 */
std::filesystem::path resolve_existing_path(const std::string& requested_path)
{
    if (requested_path.empty()) {
        return {};
    }

    std::filesystem::path path(expand_user_path(requested_path));
    if (std::filesystem::exists(path)) {
        return path;
    }
    if (path.is_absolute()) {
        return path;
    }

    // Demo binaries usually run from build/, so search cwd and parent source folders.
    const auto current = std::filesystem::current_path();
    const std::vector<std::filesystem::path> candidates = {
        current / path,
        current.parent_path() / path,
        current.parent_path().parent_path() / path,
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return path;
}

/**
 * @brief Convert an output directory argument into an absolute result directory.
 * @param requested_path User-provided output directory.
 * @return Absolute output directory path.
 */
std::filesystem::path resolve_output_path(const std::string& requested_path)
{
    std::filesystem::path path(expand_user_path(requested_path));
    if (path.is_absolute()) {
        return path;
    }
    return std::filesystem::current_path() / path;
}

/**
 * @brief Recursively discover supported input images under one directory.
 * @param image_dir Root directory selected by --image-dir.
 * @return Sorted image inputs with report-friendly relative paths.
 */
std::vector<ImageInput> collect_image_files(const std::filesystem::path& image_dir)
{
    std::vector<ImageInput> files;
    if (!std::filesystem::exists(image_dir) || !std::filesystem::is_directory(image_dir)) {
        return files;
    }

    // Keep discovery deterministic so request order and result files are stable.
    for (const auto& entry : std::filesystem::recursive_directory_iterator(image_dir)) {
        if (!entry.is_regular_file() || !is_supported_image_extension(entry.path())) {
            continue;
        }

        std::error_code ec;
        const auto relative_path = std::filesystem::relative(entry.path(), image_dir, ec);
        files.push_back({
            (ec ? entry.path().filename() : relative_path).generic_string(),
            std::filesystem::absolute(entry.path()),
        });
    }

    std::sort(files.begin(), files.end(), [](const ImageInput& a, const ImageInput& b) {
        return a.relative_path < b.relative_path;
    });
    return files;
}

/**
 * @brief Load custom class labels from a one-label-per-line text file.
 * @param labels_path Path passed through --labels.
 * @return Label list used to decode class ids in reports and overlays.
 */
std::vector<std::string> load_label_file(const std::filesystem::path& labels_path)
{
    std::vector<std::string> labels;
    std::ifstream labels_file(labels_path);
    if (!labels_file.is_open()) {
        return labels;
    }

    std::string line;
    while (std::getline(labels_file, line)) {
        line = trim_copy(line);
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    return labels;
}

/**
 * @brief Resolve one class id into a readable label for reports and overlays.
 * @param labels Loaded label table, usually COCO-80.
 * @param class_id Model class index selected during decoding.
 * @return Label text, or class_<id> when the id is outside the table.
 */
std::string label_for_class(const std::vector<std::string>& labels, int class_id)
{
    if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
        return labels[static_cast<size_t>(class_id)];
    }
    return "class_" + std::to_string(class_id);
}

/**
 * @brief Parse a comma-separated list of requested RPP device ids.
 * @param s Raw --device-list value, for example "0,2,3".
 * @return Device ids in the same order requested by the user.
 */
std::vector<int> parse_device_list(const std::string& s)
{
    std::vector<int> out;
    if (trim_copy(s).empty()) {
        return out;
    }

    // Parse individual ids manually so invalid tokens can produce actionable messages.
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (item.empty()) {
            continue;
        }
        try {
            size_t pos = 0;
            const long long value = std::stoll(item, &pos, 10);
            if (pos != item.size() ||
                value < std::numeric_limits<int>::min() ||
                value > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("bad device id");
            }
            out.push_back(static_cast<int>(value));
        } catch (const std::exception&) {
            throw std::runtime_error("invalid integer in --device-list: " + item);
        }
    }
    return out;
}

/**
 * @brief Query how many RPP cards are visible to this process.
 * @return Available device count, or 0 when the runtime query fails.
 */
int get_available_device_count()
{
    int device_count = 0;
    // rtGetDeviceCount: query how many accelerator cards are available.
    const rtError_t status = rtGetDeviceCount(&device_count);
    if (status != rtSuccess) {
        return 0;
    }
    return device_count;
}

/**
 * @brief Join integer values into one delimited string for terminal output.
 * @param values Integer list to print.
 * @param delimiter Separator inserted between adjacent values.
 * @return Joined integer list.
 */
std::string join_ints(const std::vector<int>& values, const std::string& delimiter = ",")
{
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << delimiter;
        }
        oss << values[i];
    }
    return oss.str();
}

/**
 * @brief Build the standard worker/device prefix used in logs.
 * @param worker_index Logical worker thread index.
 * @param rt_device Physical RPP device id bound to the worker.
 * @return Prefix such as "[worker 00 | device 00]".
 */
std::string worker_prefix(int worker_index, int rt_device)
{
    std::ostringstream oss;
    oss << "[worker " << std::setw(2) << std::setfill('0') << worker_index
        << " | device " << std::setw(2) << std::setfill('0') << rt_device << "]";
    return oss.str();
}

/**
 * @brief Assign request indices to workers with balanced round-robin scheduling.
 * @param total_requests Number of image requests to process.
 * @param worker_count Number of active worker threads.
 * @return Per-worker lists of request indices.
 */
std::vector<std::vector<size_t>> build_balanced_dispatch_plan(size_t total_requests, size_t worker_count)
{
    std::vector<std::vector<size_t>> assignments(worker_count);
    if (worker_count == 0) {
        return assignments;
    }

    for (size_t request_index = 0; request_index < total_requests; ++request_index) {
        assignments[request_index % worker_count].push_back(request_index);
    }
    return assignments;
}

/**
 * @brief Format a floating-point value with fixed decimal precision.
 * @param value Numeric value to format.
 * @param precision Number of digits after the decimal point.
 * @return Fixed-point decimal string.
 */
std::string format_decimal(double value, int precision = 3)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

/**
 * @brief Format a request index for stable result filenames and logs.
 * @param request_index Zero-based request index.
 * @return Six-digit zero-padded request id.
 */
std::string format_request_index(size_t request_index)
{
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << request_index;
    return oss.str();
}

/**
 * @brief Build a compact ASCII utilization/progress bar.
 * @param ratio Filled ratio in the range [0, 1].
 * @param width Number of characters in the bar body.
 * @return ASCII bar string.
 */
std::string make_bar(double ratio, size_t width = 24)
{
    ratio = std::clamp(ratio, 0.0, 1.0);
    const size_t filled = static_cast<size_t>(ratio * static_cast<double>(width) + 0.5);
    return "[" + std::string(filled, '=') + std::string(width - filled, '.') + "]";
}

/**
 * @brief Clear the current console line before drawing a live progress update.
 * @param width Number of spaces used to overwrite the line.
 */
void clear_console_line(size_t width = 140)
{
    std::cout << '\r' << std::string(width, ' ') << '\r';
}

/**
 * @brief Print a horizontal rule for terminal dashboard sections.
 * @param ch Character used for the rule.
 * @param width Number of characters to print.
 */
void print_rule(char ch = '=', size_t width = 78)
{
    std::cout << std::string(width, ch) << std::endl;
}

/**
 * @brief Print a dashboard section title and optional subtitle.
 * @param title Primary section heading.
 * @param subtitle Optional explanatory subtitle.
 */
void print_section_title(const std::string& title, const std::string& subtitle = std::string())
{
    print_rule('=');
    std::cout << title << std::endl;
    if (!subtitle.empty()) {
        std::cout << subtitle << std::endl;
    }
    print_rule('-');
}

/**
 * @brief Print one aligned key/value row in the run-configuration panel.
 * @param label Left-hand label text.
 * @param value Right-hand value text.
 * @param label_width Width used to align the label column.
 */
void print_key_value_row(const std::string& label, const std::string& value, size_t label_width = 18)
{
    const auto old_flags = std::cout.flags();
    std::cout << "  " << std::left << std::setw(static_cast<int>(label_width)) << label
              << " : " << value << std::endl;
    std::cout.flags(old_flags);
}

/**
 * @brief Print one highlighted metric row in the performance dashboard.
 * @param label Metric name printed in brackets.
 * @param value Primary metric value.
 * @param detail Optional explanatory detail.
 */
void print_metric_row(const std::string& label, const std::string& value, const std::string& detail = std::string())
{
    const auto old_flags = std::cout.flags();
    std::cout << "  " << std::left << std::setw(16) << ("[" + label + "]")
              << " " << value;
    if (!detail.empty()) {
        std::cout << "   " << detail;
    }
    std::cout << std::endl;
    std::cout.flags(old_flags);
}

/**
 * @brief Render the startup initialization progress line while workers build engines.
 * @param ready Number of workers that have finished engine creation and warmup.
 * @param total Number of active workers expected for this run.
 * @param start_time Time point when startup monitoring began.
 * @param final True when the line should be terminated with a newline.
 */
void print_initialization_line(size_t ready,
                               size_t total,
                               const std::chrono::steady_clock::time_point& start_time,
                               bool final)
{
    ready = std::min(ready, total);
    const double ratio = (total > 0) ? (static_cast<double>(ready) / static_cast<double>(total)) : 1.0;
    const double elapsed_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

    clear_console_line();
    std::cout << "  init     " << make_bar(ratio, 16)
              << " " << ready << "/" << total << " workers"
              << " | loading YOLOv5 engines + warmup"
              << " | " << format_decimal(elapsed_sec, 1) << "s elapsed";
    if (final) {
        std::cout << (ready == total ? " | ready" : " | stopped") << std::endl;
    } else {
        std::cout << std::flush;
    }
}

/**
 * @brief Build the live inference progress text shown while workers process images.
 * @param completed Number of completed image requests.
 * @param total Total scheduled image requests.
 * @param inference_count Timed execute loop count per image request.
 * @param start_time Time point when inference work began.
 * @param final True when building the final progress line.
 * @return Fully formatted progress line.
 */
std::string make_progress_line(size_t completed,
                               size_t total,
                               int inference_count,
                               const std::chrono::steady_clock::time_point& start_time,
                               bool final)
{
    completed = std::min(completed, total);
    const double ratio = (total > 0) ? (static_cast<double>(completed) / static_cast<double>(total)) : 1.0;
    const double percent = ratio * 100.0;
    const double elapsed_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    const double throughput_ips =
        (elapsed_sec > 0.0) ? (static_cast<double>(completed) / elapsed_sec) : 0.0;
    const double eta_sec =
        (throughput_ips > 0.0 && completed < total)
            ? (static_cast<double>(total - completed) / throughput_ips)
            : 0.0;

    std::ostringstream oss;
    oss << "  progress " << make_bar(ratio, 16)
        << " " << completed << "/" << total << " img"
        << " | " << format_decimal(percent, 1) << "%"
        << " | " << format_decimal(throughput_ips) << " img/s"
        << " | " << format_decimal(elapsed_sec, 1) << "s elapsed";
    if (completed < total) {
        oss << " | " << format_decimal(eta_sec, 1) << "s eta";
    } else {
        oss << " | done";
    }
    if (final && completed < total) {
        oss << " | stopped";
    }
    oss << " | loop=" << inference_count;
    return oss.str();
}

/**
 * @brief Print or refresh the live inference progress line.
 * @param completed Number of completed image requests.
 * @param total Total scheduled image requests.
 * @param inference_count Timed execute loop count per image request.
 * @param start_time Time point when inference work began.
 * @param final True when the line should be terminated with a newline.
 */
void print_progress_line(size_t completed,
                         size_t total,
                         int inference_count,
                         const std::chrono::steady_clock::time_point& start_time,
                         bool final)
{
    clear_console_line();
    std::cout << make_progress_line(completed, total, inference_count, start_time, final);
    if (final) {
        std::cout << std::endl;
    } else {
        std::cout << std::flush;
    }
}

/**
 * @brief Print resolved model, workload, threshold, and device settings before inference starts.
 * @param model_path Resolved ONNX model path.
 * @param image_source Input image directory or single image path shown to the user.
 * @param output_dir Result directory for reports and overlay images.
 * @param discovered_image_count Number of unique input images discovered.
 * @param total_requests Number of scheduled image inference requests.
 * @param inference_count Timed execute loop count per request.
 * @param workspace_mib RPP builder workspace size in MiB.
 * @param decode_config YOLOv5 threshold and class-count settings.
 * @param save_limit Maximum number of overlay images to save.
 * @param device_ids Physical RPP card ids used by this run.
 * @param dims Model input/output dimensions reported by the first worker.
 */
void print_run_configuration(const std::filesystem::path& model_path,
                             const std::string& image_source,
                             const std::filesystem::path& output_dir,
                             size_t discovered_image_count,
                             size_t total_requests,
                             int inference_count,
                             size_t workspace_mib,
                             const DecodeConfig& decode_config,
                             int save_limit,
                             const std::vector<int>& device_ids,
                             const ModelDimensionsSummary& dims)
{
    const std::string tensor_layout =
        dims.available ? (dims.input + " -> " + dims.output) : "unknown";

    print_section_title(
        "XDL MULTI-CARD YOLOV5 // RUN CONFIG",
        "YOLOv5 object detection across multiple devices with structured result capture");
    print_key_value_row("Model", model_path.string());
    print_key_value_row("Image Source", image_source);
    print_key_value_row("Tensor Layout", tensor_layout);
    print_key_value_row(
        "Workload",
        std::to_string(discovered_image_count) + " discovered / " +
            std::to_string(total_requests) + " inference requests");
    print_key_value_row("Loop Count", std::to_string(inference_count) + " timed executes per image");
    print_key_value_row("Workspace", std::to_string(workspace_mib) + " MiB per worker");
    print_key_value_row(
        "Thresholds",
        "obj=" + format_decimal(decode_config.confidence_threshold, 2) +
            " cls=" + format_decimal(decode_config.score_threshold, 2) +
            " nms=" + format_decimal(decode_config.nms_threshold, 2));
    print_key_value_row("Classes", std::to_string(decode_config.class_count));
    print_key_value_row("Save Limit", save_limit == 0 ? "disabled" : std::to_string(save_limit) + " overlays");
    print_key_value_row(
        "Devices",
        join_ints(device_ids) + " (" + std::to_string(device_ids.size()) + " active)");
    print_key_value_row("Dispatch", "balanced round-robin");
    print_key_value_row("Result Directory", output_dir.string());
    print_rule('=');
}

/**
 * @brief Pad an image to the square canvas expected by YOLOv5 preprocessing.
 * @param source Original BGR image loaded by OpenCV.
 * @return Square image with source pixels copied to the top-left corner.
 */
cv::Mat format_yolov5(const cv::Mat& source)
{
    const int col = source.cols;
    const int row = source.rows;
    const int padded_size = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(padded_size, padded_size, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

/**
 * @brief Convert one OpenCV image into the normalized CHW tensor used by YOLOv5.
 * @param source Original BGR image loaded from disk.
 * @param input_width Model input width.
 * @param input_height Model input height.
 * @param tensor Output flattened CHW float tensor.
 * @param padded_width Output square canvas width before resize.
 * @param padded_height Output square canvas height before resize.
 * @return True when preprocessing succeeds and tensor size matches the model input.
 */
bool build_yolov5_tensor(const cv::Mat& source,
                         int input_width,
                         int input_height,
                         std::vector<float>& tensor,
                         int& padded_width,
                         int& padded_height)
{
    if (source.empty() || input_width <= 0 || input_height <= 0) {
        return false;
    }

    // Preserve the original aspect ratio by padding to a square before resizing.
    const cv::Mat padded = format_yolov5(source);
    padded_width = padded.cols;
    padded_height = padded.rows;

    cv::Mat resized;
    // OpenCV image data is BGR; YOLOv5 ONNX exports expect normalized RGB input.
    cv::resize(padded, resized, cv::Size(input_width, input_height));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    cv::Mat float_image;
    resized.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(float_image, channels);
    if (channels.size() != 3) {
        return false;
    }

    // Pack channels into the contiguous CHW layout consumed by the RPP input binding.
    const size_t plane_size = static_cast<size_t>(input_width) * static_cast<size_t>(input_height);
    tensor.assign(3 * plane_size, 0.0F);
    for (size_t channel = 0; channel < channels.size(); ++channel) {
        const cv::Mat& plane = channels[channel];
        if (!plane.isContinuous()) {
            return false;
        }
        std::copy(plane.ptr<float>(),
                  plane.ptr<float>() + plane_size,
                  tensor.begin() + static_cast<std::ptrdiff_t>(channel * plane_size));
    }
    return true;
}

/**
 * @brief Apply non-maximum suppression to candidate bounding boxes.
 * @param bboxes Candidate boxes in original image coordinates.
 * @param scores Confidence scores aligned with bboxes.
 * @param score_threshold Minimum score required before NMS.
 * @param nms_threshold IoU threshold used to suppress overlapping boxes.
 * @param indices Output indices of boxes kept after suppression.
 * @param top_k Optional maximum number of candidates considered before NMS.
 */
void nms_boxes(const std::vector<cv::Rect>& bboxes,
               const std::vector<float>& scores,
               float score_threshold,
               float nms_threshold,
               std::vector<int>& indices,
               int top_k = 0)
{
    indices.clear();
    if (bboxes.empty() || scores.empty() || bboxes.size() != scores.size()) {
        return;
    }

    // Keep only candidates that are strong enough to participate in suppression.
    std::vector<int> candidate_indices;
    for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
        if (scores[static_cast<size_t>(i)] >= score_threshold) {
            candidate_indices.push_back(i);
        }
    }
    if (candidate_indices.empty()) {
        return;
    }

    std::sort(candidate_indices.begin(), candidate_indices.end(), [&scores](int a, int b) {
        return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
    });

    if (top_k > 0 && top_k < static_cast<int>(candidate_indices.size())) {
        candidate_indices.resize(static_cast<size_t>(top_k));
    }

    // Cache box areas so repeated IoU comparisons stay inexpensive.
    std::vector<float> areas;
    areas.reserve(bboxes.size());
    for (const auto& rect : bboxes) {
        areas.push_back(static_cast<float>(rect.width * rect.height));
    }

    // Repeatedly keep the best remaining box and remove high-overlap candidates.
    while (!candidate_indices.empty()) {
        const int best_idx = candidate_indices[0];
        indices.push_back(best_idx);
        if (candidate_indices.size() == 1) {
            break;
        }

        std::vector<int> remaining_indices;
        remaining_indices.reserve(candidate_indices.size() - 1);
        const cv::Rect& best_rect = bboxes[static_cast<size_t>(best_idx)];

        for (size_t i = 1; i < candidate_indices.size(); ++i) {
            const int idx = candidate_indices[i];
            const cv::Rect& current_rect = bboxes[static_cast<size_t>(idx)];

            const int x1 = std::max(best_rect.x, current_rect.x);
            const int y1 = std::max(best_rect.y, current_rect.y);
            const int x2 = std::min(best_rect.x + best_rect.width, current_rect.x + current_rect.width);
            const int y2 = std::min(best_rect.y + best_rect.height, current_rect.y + current_rect.height);

            float intersection = 0.0F;
            if (x2 > x1 && y2 > y1) {
                intersection = static_cast<float>((x2 - x1) * (y2 - y1));
            }
            const float union_area =
                areas[static_cast<size_t>(best_idx)] + areas[static_cast<size_t>(idx)] - intersection;
            const float iou = (union_area > 0.0F) ? (intersection / union_area) : 0.0F;
            if (iou <= nms_threshold) {
                remaining_indices.push_back(idx);
            }
        }

        candidate_indices = std::move(remaining_indices);
    }
}

/**
 * @brief Decode raw YOLOv5 output rows into final detections.
 * @param output_data Flattened RPP output tensor containing YOLOv5 rows.
 * @param input_width Model input width used during preprocessing.
 * @param input_height Model input height used during preprocessing.
 * @param padded_width Width of the square padded image before resize.
 * @param padded_height Height of the square padded image before resize.
 * @param original_width Original image width used for clipping.
 * @param original_height Original image height used for clipping.
 * @param config Thresholds, class count, and max-detection settings.
 * @return Final detections after candidate filtering and NMS.
 */
std::vector<Detection> decode_yolov5_output(const std::vector<float>& output_data,
                                            int input_width,
                                            int input_height,
                                            int padded_width,
                                            int padded_height,
                                            int original_width,
                                            int original_height,
                                            const DecodeConfig& config)
{
    if (config.class_count <= 0) {
        throw std::runtime_error("--class-count must be greater than 0.");
    }

    // Each row stores center box, objectness, then class scores.
    const size_t row_stride = static_cast<size_t>(5 + config.class_count);
    if (row_stride <= 5 || output_data.empty() || output_data.size() % row_stride != 0) {
        std::ostringstream oss;
        oss << "YOLOv5 output size " << output_data.size()
            << " is not divisible by row stride " << row_stride
            << " (5 + class_count). Pass --class-count for custom models.";
        throw std::runtime_error(oss.str());
    }

    // Scale boxes from model input space back to the padded square image.
    const size_t rows = output_data.size() / row_stride;
    const float x_factor = static_cast<float>(padded_width) / static_cast<float>(input_width);
    const float y_factor = static_cast<float>(padded_height) / static_cast<float>(input_height);
    const cv::Rect image_bounds(0, 0, original_width, original_height);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    class_ids.reserve(rows);
    confidences.reserve(rows);
    boxes.reserve(rows);

    // Decode every output row, keeping only confident class/object pairs.
    const float* data = output_data.data();
    for (size_t row = 0; row < rows; ++row) {
        const float objectness = data[4];
        if (objectness >= config.confidence_threshold) {
            const float* class_scores = data + 5;
            int best_class = 0;
            float best_score = class_scores[0];
            for (int class_id = 1; class_id < config.class_count; ++class_id) {
                const float score = class_scores[class_id];
                if (score > best_score) {
                    best_score = score;
                    best_class = class_id;
                }
            }

            if (best_score > config.score_threshold) {
                const float x = data[0];
                const float y = data[1];
                const float w = data[2];
                const float h = data[3];
                const int left = static_cast<int>((x - 0.5F * w) * x_factor);
                const int top = static_cast<int>((y - 0.5F * h) * y_factor);
                const int width = static_cast<int>(w * x_factor);
                const int height = static_cast<int>(h * y_factor);

                const cv::Rect clipped = (cv::Rect(left, top, width, height) & image_bounds);
                if (clipped.area() > 0) {
                    class_ids.push_back(best_class);
                    confidences.push_back(objectness);
                    boxes.push_back(clipped);
                }
            }
        }

        data += row_stride;
    }

    // Remove duplicate overlapping boxes before returning final detections.
    std::vector<int> nms_result;
    nms_boxes(boxes,
              confidences,
              config.score_threshold,
              config.nms_threshold,
              nms_result,
              config.max_detections);

    std::vector<Detection> detections;
    detections.reserve(nms_result.size());
    for (int idx : nms_result) {
        detections.push_back({
            class_ids[static_cast<size_t>(idx)],
            confidences[static_cast<size_t>(idx)],
            boxes[static_cast<size_t>(idx)],
        });
        if (config.max_detections > 0 &&
            static_cast<int>(detections.size()) >= config.max_detections) {
            break;
        }
    }

    return detections;
}

/**
 * @brief Convert an arbitrary image stem into a filename-safe token.
 * @param value Raw filename stem.
 * @return Token containing only alphanumeric, dash, and underscore characters.
 */
std::string sanitize_filename_token(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch) != 0 || ch == '-' || ch == '_') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) {
        return "image";
    }
    return out;
}

/**
 * @brief Draw detections on one image and save the overlay into the result directory.
 * @param image_input Source image metadata and absolute path.
 * @param request_index Request id used in the output filename.
 * @param output_dir Root result directory.
 * @param detections Final detections to render.
 * @param labels Class-name table used for overlay text.
 * @param saved_relative_path Output relative image path written into result files.
 * @return True when the overlay image is written successfully.
 */
bool save_detection_overlay(const ImageInput& image_input,
                            size_t request_index,
                            const std::filesystem::path& output_dir,
                            const std::vector<Detection>& detections,
                            const std::vector<std::string>& labels,
                            std::string& saved_relative_path)
{
    // Reload the original image so overlay drawing never mutates preprocessing buffers.
    cv::Mat frame = cv::imread(image_input.absolute_path.string(), cv::IMREAD_COLOR);
    if (frame.empty()) {
        return false;
    }

    // Draw each retained detection with a colored rectangle and compact label background.
    for (const auto& detection : detections) {
        const cv::Scalar color = BOX_COLORS[static_cast<size_t>(std::max(0, detection.class_id)) % BOX_COLORS.size()];
        cv::rectangle(frame, detection.box, color, 2);

        const std::string label =
            label_for_class(labels, detection.class_id) + " " + format_decimal(detection.confidence, 2);
        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        const int label_x = std::max(0, detection.box.x);
        const int label_y = std::max(text_size.height + 4, detection.box.y);
        const cv::Rect label_bg(
            label_x,
            std::max(0, label_y - text_size.height - 6),
            std::min(text_size.width + 8, std::max(1, frame.cols - label_x)),
            text_size.height + baseline + 8);

        cv::rectangle(frame, label_bg, color, cv::FILLED);
        cv::putText(frame,
                    label,
                    cv::Point(label_x + 4, label_y - 4),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    cv::Scalar(0, 0, 0),
                    1,
                    cv::LINE_AA);
    }

    // Save rendered overlays under images/ so text reports and images stay grouped.
    const auto image_output_dir = output_dir / "images";
    std::filesystem::create_directories(image_output_dir);

    const std::string stem =
        sanitize_filename_token(std::filesystem::path(image_input.relative_path).stem().string());
    const std::string filename =
        "request_" + format_request_index(request_index) + "_" + stem + ".jpg";
    const auto output_path = image_output_dir / filename;
    if (!cv::imwrite(output_path.string(), frame)) {
        return false;
    }

    saved_relative_path = (std::filesystem::path("images") / filename).generic_string();
    return true;
}

/**
 * @brief Format detections into the compact text representation used by detections_result.txt.
 * @param detections Final detections for one request.
 * @param labels Class-name table used to resolve class ids.
 * @return Semicolon-separated list of label, score, and box tuples.
 */
std::string format_detection_list(const std::vector<Detection>& detections,
                                  const std::vector<std::string>& labels)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (i > 0) {
            oss << ";";
        }
        const auto& detection = detections[i];
        oss << "(" << label_for_class(labels, detection.class_id)
            << "," << detection.confidence
            << "," << detection.box.x
            << "," << detection.box.y
            << "," << detection.box.width
            << "," << detection.box.height
            << ")";
    }
    return oss.str();
}

/**
 * @brief Write ordered detection and performance reports after all workers finish.
 * @param output_dir Result directory.
 * @param results Ordered per-request results.
 * @param device_performance Per-worker performance summary.
 * @param wall_clock_ms Total run wall-clock time in milliseconds.
 * @param wall_clock_sec Total run wall-clock time in seconds.
 * @param aggregate_ips Aggregate image throughput across active devices.
 * @return True when both report files are written successfully.
 */
bool write_result_files(const std::filesystem::path& output_dir,
                        const std::vector<RequestResult>& results,
                        const std::vector<DevicePerformance>& device_performance,
                        long long wall_clock_ms,
                        double wall_clock_sec,
                        double aggregate_ips)
{
    std::filesystem::create_directories(output_dir);

    std::ofstream detections_file(output_dir / "detections_result.txt");
    if (!detections_file.is_open()) {
        std::cerr << "Failed to open detections_result.txt in output directory." << std::endl;
        return false;
    }

    std::ofstream performance_file(output_dir / "performance_result.txt");
    if (!performance_file.is_open()) {
        std::cerr << "Failed to open performance_result.txt in output directory." << std::endl;
        return false;
    }

    // Emit request-level reports in deterministic request order.
    for (const auto& result : results) {
        detections_file << "Request " << format_request_index(result.request_index)
                        << " | worker=" << result.worker_index
                        << " | device=" << result.device_id
                        << " | image=" << result.image_relative_path
                        << " | detections=" << result.detection_count
                        << " | overlay=" << (result.overlay_path.empty() ? "none" : result.overlay_path)
                        << " | boxes=" << (result.detections.empty() ? "none" : result.detections)
                        << "\n";

        performance_file << "Request " << format_request_index(result.request_index)
                         << " | worker=" << result.worker_index
                         << " | device=" << result.device_id
                         << " | image=" << result.image_relative_path
                         << " | detection_count=" << result.detection_count
                         << " | yolo_inference_ms=" << std::fixed << std::setprecision(3)
                         << result.model_inference_ms
                         << " | wall_latency_ms=" << std::fixed << std::setprecision(3)
                         << result.wall_latency_ms
                         << "\n";
    }

    performance_file << "Summary"
                     << " | wall_clock_ms=" << wall_clock_ms
                     << " | wall_clock_sec=" << std::fixed << std::setprecision(3) << wall_clock_sec
                     << " | aggregate_images_per_sec=" << std::fixed << std::setprecision(3) << aggregate_ips
                     << "\n";
    performance_file << "Columns"
                     << " | device=physical RPP device id"
                     << " | worker=logical host thread index"
                     << " | requests=image inference requests processed by this worker"
                     << " | detections=retained boxes after thresholding and NMS"
                     << " | elapsed_sec=worker wall-clock runtime"
                     << " | images_per_sec=per-worker throughput"
                     << " | avg_wall_latency_ms=average end-to-end request latency"
                     << " | avg_yolo_ms=average YOLOv5 model execute time"
                     << "\n";

    // Append per-device summaries after the aggregate run metrics.
    for (const auto& device : device_performance) {
        performance_file << "Device " << device.device_id
                         << " | worker=" << device.worker_index
                         << " | requests=" << device.requests
                         << " | detections=" << device.detections
                         << " | elapsed_sec=" << std::fixed << std::setprecision(3) << device.elapsed_sec
                         << " | images_per_sec=" << std::fixed << std::setprecision(3) << device.throughput_ips
                         << " | avg_wall_latency_ms=" << std::fixed << std::setprecision(3)
                         << device.average_wall_latency_ms
                         << " | avg_yolo_ms=" << std::fixed << std::setprecision(3)
                         << device.average_model_ms
                         << "\n";
    }

    return true;
}

/**
 * @brief Print aggregate throughput, latency, model timing, and per-device summary rows.
 * @param processed Number of successfully completed image requests.
 * @param wall_clock_ms Total timed inference phase wall-clock duration in milliseconds.
 * @param wall_clock_sec Total timed inference phase wall-clock duration in seconds.
 * @param aggregate_ips Aggregate image requests per second across all workers.
 * @param inference_count Timed execute loop count per image request.
 * @param total_detections Total boxes retained after thresholding and NMS.
 * @param average_wall_latency_ms Average end-to-end request latency in milliseconds.
 * @param average_model_ms Average YOLOv5 execute time in milliseconds.
 * @param device_performance Per-worker/device performance rows.
 */
void print_performance_summary(size_t processed,
                               long long wall_clock_ms,
                               double wall_clock_sec,
                               double aggregate_ips,
                               int inference_count,
                               size_t total_detections,
                               double average_wall_latency_ms,
                               double average_model_ms,
                               const std::vector<DevicePerformance>& device_performance)
{
    print_section_title(
        "PERFORMANCE DASHBOARD",
        "High-signal run summary with aggregate throughput, latency, and device-level distribution");

    print_metric_row(
        "THROUGHPUT",
        format_decimal(aggregate_ips) + " img/s",
        std::to_string(processed) + " completed requests across " +
            std::to_string(device_performance.size()) + " devices");
    print_metric_row(
        "LOOP COUNT",
        std::to_string(inference_count),
        "timed executes per YOLOv5 image request");
    print_metric_row(
        "DETECTIONS",
        std::to_string(total_detections),
        "retained boxes after thresholding and NMS");
    print_metric_row(
        "LATENCY",
        format_decimal(average_wall_latency_ms) + " ms",
        "average per image request");
    print_metric_row(
        "YOLOV5",
        format_decimal(average_model_ms) + " ms",
        "average model execute time, measured with loop=" + std::to_string(inference_count));
    print_metric_row(
        "WALL CLOCK",
        std::to_string(wall_clock_ms) + " ms",
        format_decimal(wall_clock_sec, 3) + " s end-to-end");

    std::cout << std::endl;
    std::cout << "DEVICE BREAKDOWN" << std::endl;
    std::cout << "  columns: card=physical RPP device id, worker=logical host thread,"
              << " util=relative per-worker throughput bar, imgs=image requests,"
              << " boxes=retained detections, img/s=per-worker throughput,"
              << " lat_ms=average request wall latency, yolo_ms=average model execute time,"
              << " elapsed_s=worker wall-clock runtime" << std::endl;
    std::cout << "  "
              << std::setw(4) << "card"
              << " " << std::setw(6) << "worker"
              << " " << std::setw(14) << "util"
              << " " << std::setw(5) << "imgs"
              << " " << std::setw(5) << "boxes"
              << " " << std::setw(8) << "img/s"
              << " " << std::setw(8) << "lat_ms"
              << " " << std::setw(8) << "yolo_ms"
              << " " << std::setw(9) << "elapsed_s"
              << std::endl;

    double max_throughput = 0.0;
    for (const auto& device : device_performance) {
        max_throughput = std::max(max_throughput, device.throughput_ips);
    }

    for (const auto& device : device_performance) {
        const double ratio = (max_throughput > 0.0) ? (device.throughput_ips / max_throughput) : 0.0;
        std::cout << "  "
                  << std::setw(4) << device.device_id
                  << " " << std::setw(6) << device.worker_index
                  << " " << std::setw(14) << make_bar(ratio, 12)
                  << " " << std::setw(5) << device.requests
                  << " " << std::setw(5) << device.detections
                  << " " << std::setw(8) << format_decimal(device.throughput_ips)
                  << " " << std::setw(8) << format_decimal(device.average_wall_latency_ms)
                  << " " << std::setw(8) << format_decimal(device.average_model_ms)
                  << " " << std::setw(9) << format_decimal(device.elapsed_sec)
                  << std::endl;
    }
    print_rule('=');
}

} // namespace

/**
 * @brief Run the multi-card YOLOv5 demo from CLI parsing through reports and overlays.
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument values.
 * @return EXIT_SUCCESS when all scheduled image requests complete, otherwise EXIT_FAILURE.
 */
int main(int argc, char** argv)
{
    // Convert uncaught RPP allocation failures into a readable terminal message.
    std::set_terminate(handle_unhandled_rpp_exception);

    // CLI state collected before validation and path resolution.
    std::string model_path_arg;
    std::string image_dir_arg = DEFAULT_IMAGE_DIR;
    std::string image_path_arg;
    std::string output_dir_arg = DEFAULT_OUTPUT_DIR;
    std::string labels_path_arg;
    std::string device_list_arg;
    long long requested_image_count = 0;
    int requested_device_count = 0;
    int inference_count = 1;
    int save_limit = 100;
    long long workspace_mib = 256;
    DecodeConfig decode_config;
    bool verbose = false;

    // Declare the CLI surface expected by automation and README examples.
    argparse::ArgumentParser program(
        "multi_card_yolov5 demo: multi-device YOLOv5 object detection",
        "0.1",
        argparse::default_arguments::help);
    program.add_argument("-o", "--onnx")
        .required()
        .store_into(model_path_arg)
        .help("path to the YOLOv5 ONNX model file.");
    program.add_argument("-i", "--image-dir")
        .default_value(image_dir_arg)
        .store_into(image_dir_arg)
        .help("directory containing input images.");
    program.add_argument("--image")
        .default_value(image_path_arg)
        .store_into(image_path_arg)
        .help("single input image; overrides --image-dir when set.");
    program.add_argument("-j", "--output-json", "--output")
        .default_value(output_dir_arg)
        .store_into(output_dir_arg)
        .help("output result directory.");
    program.add_argument("-n", "--image-count")
        .default_value(requested_image_count)
        .store_into(requested_image_count)
        .help("number of image inference requests; 0 means each discovered image is processed once.");
    program.add_argument("--loop")
        .default_value(inference_count)
        .store_into(inference_count)
        .help("timed inference loop count per image request.");
    program.add_argument("-d", "--device-count")
        .default_value(requested_device_count)
        .store_into(requested_device_count)
        .help("device count, 0 means use all available devices.");
    program.add_argument("--device-list")
        .default_value(device_list_arg)
        .store_into(device_list_arg)
        .help("comma-separated device indices, for example 0,2,3.");
    program.add_argument("--labels")
        .default_value(labels_path_arg)
        .store_into(labels_path_arg)
        .help("optional class-name file, one label per line; default is embedded COCO-80.");
    program.add_argument("--class-count")
        .default_value(0)
        .store_into(decode_config.class_count)
        .help("class count in the YOLOv5 output; 0 uses the loaded label count.");
    program.add_argument("--score-threshold")
        .default_value(decode_config.score_threshold)
        .store_into(decode_config.score_threshold)
        .help("minimum best-class score before NMS.");
    program.add_argument("--confidence-threshold")
        .default_value(decode_config.confidence_threshold)
        .store_into(decode_config.confidence_threshold)
        .help("minimum objectness score before class-score evaluation.");
    program.add_argument("--nms-threshold")
        .default_value(decode_config.nms_threshold)
        .store_into(decode_config.nms_threshold)
        .help("IoU threshold used by non-maximum suppression.");
    program.add_argument("--max-detections")
        .default_value(decode_config.max_detections)
        .store_into(decode_config.max_detections)
        .help("maximum retained detections per image; 0 means no explicit cap.");
    program.add_argument("--save-limit")
        .default_value(save_limit)
        .store_into(save_limit)
        .help("maximum number of rendered overlay images to save; 0 disables overlay output.");
    program.add_argument("--workspace-mib")
        .default_value(workspace_mib)
        .store_into(workspace_mib)
        .help("RPP builder workspace size per worker in MiB.");
    program.add_argument("-v", "--verbose")
        .default_value(false)
        .implicit_value(true)
        .store_into(verbose)
        .help("show verbose RPP log output.");

    // Normalize arguments before argparse handles aliases and negative-style values.
    auto preprocessed_arguments = preprocess_args(argc, argv);
    std::vector<const char*> fixed_arguments;
    to_char_argument_vector(preprocessed_arguments, argv, fixed_arguments);
    try {
        program.parse_args(static_cast<int>(fixed_arguments.size()), fixed_arguments.data());
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program << std::endl;
        return EXIT_FAILURE;
    }

    // Configure RPP log verbosity after CLI parsing.
    sample::gLogger.setReportableSeverity(
        verbose ? infer1::ILogger::Severity::kVERBOSE : infer1::ILogger::Severity::kERROR);

    // Resolve and validate model/output arguments before worker startup.
    const auto model_path = resolve_existing_path(model_path_arg);
    const auto output_dir = resolve_output_path(output_dir_arg);
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Cannot find YOLOv5 ONNX model file: " << model_path << std::endl;
        return EXIT_FAILURE;
    }
    if (requested_image_count < 0) {
        requested_image_count = 0;
    }
    if (requested_device_count < 0) {
        std::cerr << "--device-count must be >= 0." << std::endl;
        return EXIT_FAILURE;
    }
    if (inference_count < 1) {
        inference_count = 1;
    }
    if (save_limit < 0) {
        save_limit = 0;
    }
    if (workspace_mib < 16) {
        std::cerr << "--workspace-mib must be at least 16." << std::endl;
        return EXIT_FAILURE;
    }
    if (decode_config.max_detections < 0) {
        decode_config.max_detections = 0;
    }
    const size_t workspace_bytes =
        static_cast<size_t>(workspace_mib) * static_cast<size_t>(1024) * static_cast<size_t>(1024);

    // Build the label table used to decode class ids into readable names.
    std::vector<std::string> labels;
    if (!labels_path_arg.empty()) {
        const auto labels_path = resolve_existing_path(labels_path_arg);
        if (!std::filesystem::exists(labels_path)) {
            std::cerr << "Cannot find labels file: " << labels_path << std::endl;
            return EXIT_FAILURE;
        }
        labels = load_label_file(labels_path);
        if (labels.empty()) {
            std::cerr << "Labels file is empty: " << labels_path << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        labels = coco80_class_labels();
    }
    if (decode_config.class_count == 0) {
        decode_config.class_count = static_cast<int>(labels.size());
    }
    if (decode_config.class_count <= 0) {
        std::cerr << "--class-count must be greater than 0." << std::endl;
        return EXIT_FAILURE;
    }

    // Resolve either one explicit image or a deterministic directory scan.
    std::vector<ImageInput> image_inputs;
    std::string image_source_label;
    if (!image_path_arg.empty()) {
        const auto image_path = resolve_existing_path(image_path_arg);
        if (!std::filesystem::exists(image_path) || !std::filesystem::is_regular_file(image_path)) {
            std::cerr << "Cannot find input image: " << image_path << std::endl;
            return EXIT_FAILURE;
        }
        image_inputs.push_back({image_path.filename().generic_string(), std::filesystem::absolute(image_path)});
        image_source_label = image_path.string();
    } else {
        const auto image_dir = resolve_existing_path(image_dir_arg);
        if (!std::filesystem::exists(image_dir) || !std::filesystem::is_directory(image_dir)) {
            std::cerr << "Cannot find image directory: " << image_dir << std::endl;
            return EXIT_FAILURE;
        }
        image_inputs = collect_image_files(image_dir);
        image_source_label = image_dir.string();
    }

    if (image_inputs.empty()) {
        std::cerr << "No input images found." << std::endl;
        return EXIT_FAILURE;
    }

    // Expand --image-count into concrete request slots, reusing discovered images if needed.
    const size_t total_requests =
        requested_image_count > 0
            ? static_cast<size_t>(requested_image_count)
            : image_inputs.size();
    if (total_requests == 0) {
        std::cerr << "No image inference requests scheduled." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<int> requested_device_ids;
    try {
        requested_device_ids = parse_device_list(device_list_arg);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Resolve the physical RPP cards used by worker threads.
    const int available_device_count = get_available_device_count();
    int device_count_to_use = requested_device_count;
    if (!requested_device_ids.empty()) {
        if (requested_device_count > 0 &&
            static_cast<int>(requested_device_ids.size()) != requested_device_count) {
            std::cerr << "--device-count (" << requested_device_count
                      << ") must equal --device-list length (" << requested_device_ids.size()
                      << "), or use --device-count 0." << std::endl;
            return EXIT_FAILURE;
        }
        for (int id : requested_device_ids) {
            if (available_device_count <= 0 || id < 0 || id >= available_device_count) {
                std::cerr << "Invalid device id " << id << " in --device-list (valid: 0.."
                          << std::max(0, available_device_count - 1) << ")." << std::endl;
                return EXIT_FAILURE;
            }
        }
        device_count_to_use = static_cast<int>(requested_device_ids.size());
    } else {
        if (requested_device_count > 0 && requested_device_count > available_device_count) {
            std::cerr << "Requested device count (" << requested_device_count
                      << ") exceeds available device count (" << available_device_count << ")." << std::endl;
            return EXIT_FAILURE;
        }
        if (device_count_to_use <= 0) {
            device_count_to_use = available_device_count;
        }
    }

    if (device_count_to_use <= 0) {
        std::cerr << "No available RPP devices found." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<int> resolved_device_ids;
    if (!requested_device_ids.empty()) {
        resolved_device_ids = requested_device_ids;
    } else {
        resolved_device_ids.reserve(static_cast<size_t>(device_count_to_use));
        for (int i = 0; i < device_count_to_use; ++i) {
            resolved_device_ids.push_back(i);
        }
    }

    if (resolved_device_ids.size() > total_requests) {
        resolved_device_ids.resize(total_requests);
        device_count_to_use = static_cast<int>(resolved_device_ids.size());
        std::cout << "Active device list reduced to " << device_count_to_use
                  << " because only " << total_requests << " request(s) are scheduled." << std::endl;
    }

    // Map each request slot back to an input image index for repeat workloads.
    std::vector<size_t> request_image_indices(total_requests);
    for (size_t request_index = 0; request_index < total_requests; ++request_index) {
        request_image_indices[request_index] = request_index % image_inputs.size();
    }

    std::filesystem::create_directories(output_dir);
    const auto per_worker_request_indices =
        build_balanced_dispatch_plan(total_requests, static_cast<size_t>(device_count_to_use));

    // Shared worker coordination and result state.
    std::mutex init_mtx;
    std::mutex sync_mtx;
    std::mutex stats_mtx;
    std::mutex error_mtx;
    std::mutex results_mtx;
    std::mutex console_mtx;
    std::mutex dims_mtx;
    std::condition_variable sync_cv;
    size_t ready_worker_count = 0;
    bool start_processing = false;
    std::atomic<size_t> completed_requests{0};
    std::atomic<bool> worker_failed{false};
    std::vector<std::string> worker_errors;
    ModelDimensionsSummary dims_summary;

    std::vector<size_t> per_device_requests(static_cast<size_t>(device_count_to_use), 0);
    std::vector<size_t> per_device_detections(static_cast<size_t>(device_count_to_use), 0);
    std::vector<double> per_device_elapsed_sec(static_cast<size_t>(device_count_to_use), 0.0);
    std::vector<double> per_device_wall_latency_sum_ms(static_cast<size_t>(device_count_to_use), 0.0);
    std::vector<double> per_device_model_sum_ms(static_cast<size_t>(device_count_to_use), 0.0);
    std::vector<RequestResult> results(total_requests);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(device_count_to_use));

    // Start one worker per active card; each worker owns its runtime context and request list.
    for (int worker_index = 0; worker_index < device_count_to_use; ++worker_index) {
        const int rt_device = resolved_device_ids[static_cast<size_t>(worker_index)];
        workers.emplace_back([&,
                              worker_index,
                              rt_device]() {
            auto append_error = [&](const std::string& message) {
                worker_failed.store(true, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> error_lock(error_mtx);
                    worker_errors.emplace_back(message);
                }
                sync_cv.notify_all();
            };

            size_t requests_done = 0;
            size_t detections_done = 0;
            double wall_latency_sum_ms = 0.0;
            double model_sum_ms = 0.0;
            std::chrono::steady_clock::time_point work_start {};
            bool has_started_work = false;

            auto record_device_stats = [&]() {
                const auto work_end = std::chrono::steady_clock::now();
                const double elapsed_sec = has_started_work
                    ? std::chrono::duration<double>(work_end - work_start).count()
                    : 0.0;
                std::lock_guard<std::mutex> stats_lock(stats_mtx);
                per_device_requests[static_cast<size_t>(worker_index)] = requests_done;
                per_device_detections[static_cast<size_t>(worker_index)] = detections_done;
                per_device_elapsed_sec[static_cast<size_t>(worker_index)] = elapsed_sec;
                per_device_wall_latency_sum_ms[static_cast<size_t>(worker_index)] = wall_latency_sum_ms;
                per_device_model_sum_ms[static_cast<size_t>(worker_index)] = model_sum_ms;
            };

            // rtSetDevice: bind this host thread to one physical RPP card.
            if (rtSetDevice(rt_device) != rtSuccess) {
                append_error(worker_prefix(worker_index, rt_device) + " failed to set RPP device.");
                return;
            }

            std::unique_ptr<Yolo> yolo;
            {
                std::lock_guard<std::mutex> init_lock(init_mtx);
                if (worker_failed.load(std::memory_order_acquire)) {
                    return;
                }

                try {
                    // Build the worker-local YOLOv5 engine/context on the selected RPP card.
                    yolo = std::make_unique<Yolo>(model_path.string(), workspace_bytes);
                    if (!yolo->init_engine()) {
                        append_error(worker_prefix(worker_index, rt_device) +
                                     " failed to initialize YOLOv5 engine/context. "
                                     "For yolov5m-sized models this usually means the RPP context cannot be created "
                                     "with the currently available card memory.");
                        return;
                    }
                    if (yolo->getInputChannels() != 3) {
                        append_error(worker_prefix(worker_index, rt_device) +
                                     " YOLOv5 model input channel count is " +
                                     std::to_string(yolo->getInputChannels()) + ", expected 3.");
                        return;
                    }

                    // Run one all-zero warmup request so measured requests use a ready context.
                    std::vector<float> warmup_input(yolo->getInputSize(), 0.0F);
                    std::vector<float> warmup_output;
                    float warmup_ms = 0.0F;
                    if (!yolo->infer(warmup_input, 1, warmup_output, warmup_ms)) {
                        append_error(worker_prefix(worker_index, rt_device) +
                                     " YOLOv5 warmup failed. "
                                     "If RPP reported memory allocation error 2, try a smaller YOLOv5 model, "
                                     "free card memory, or select fewer devices.");
                        return;
                    }

                    {
                        // The first ready worker provides model I/O dimensions for the run panel.
                        std::lock_guard<std::mutex> dims_lock(dims_mtx);
                        if (!dims_summary.available) {
                            dims_summary.available = true;
                            dims_summary.input = dims_to_string(yolo->getInputDimensions());
                            dims_summary.output = dims_to_string(yolo->getOutputDimensions());
                        }
                    }
                } catch (const std::exception& e) {
                    append_error(worker_prefix(worker_index, rt_device) +
                                 " YOLOv5 initialization threw exception: " + e.what());
                    return;
                } catch (int error_code) {
                    append_error(worker_prefix(worker_index, rt_device) +
                                 " YOLOv5 initialization threw RPP error code " +
                                 std::to_string(error_code) + ".");
                    return;
                } catch (...) {
                    append_error(worker_prefix(worker_index, rt_device) +
                                 " YOLOv5 initialization threw an unknown exception.");
                    return;
                }
            }

            {
                // Signal that this worker is initialized and ready for the synchronized start.
                std::lock_guard<std::mutex> sync_lock(sync_mtx);
                ++ready_worker_count;
            }
            sync_cv.notify_all();

            {
                std::unique_lock<std::mutex> sync_lock(sync_mtx);
                sync_cv.wait(sync_lock, [&]() {
                    return start_processing || worker_failed.load(std::memory_order_acquire);
                });
            }
            if (worker_failed.load(std::memory_order_acquire)) {
                record_device_stats();
                return;
            }

            work_start = std::chrono::steady_clock::now();
            has_started_work = true;

            // Process this worker's assigned image requests in deterministic request order.
            for (size_t request_index : per_worker_request_indices[static_cast<size_t>(worker_index)]) {
                if (worker_failed.load(std::memory_order_acquire)) {
                    break;
                }

                try {
                    // Load the source image and convert it into the YOLOv5 input tensor.
                    const ImageInput& image_input =
                        image_inputs[request_image_indices[request_index]];
                    const auto request_start = std::chrono::steady_clock::now();

                    cv::Mat frame = cv::imread(image_input.absolute_path.string(), cv::IMREAD_COLOR);
                    if (frame.empty()) {
                        record_device_stats();
                        append_error(worker_prefix(worker_index, rt_device) +
                                     " failed to load image: " + image_input.absolute_path.string());
                        return;
                    }

                    std::vector<float> input_tensor;
                    int padded_width = 0;
                    int padded_height = 0;
                    if (!build_yolov5_tensor(frame,
                                             yolo->getInputWidth(),
                                             yolo->getInputHeight(),
                                             input_tensor,
                                             padded_width,
                                             padded_height)) {
                        record_device_stats();
                        append_error(worker_prefix(worker_index, rt_device) +
                                     " failed to preprocess image: " + image_input.absolute_path.string());
                        return;
                    }

                    // Execute YOLOv5 on RPP and decode the raw output rows on the host.
                    std::vector<float> output_data;
                    float model_inference_ms = 0.0F;
                    if (!yolo->infer(input_tensor, inference_count, output_data, model_inference_ms)) {
                        record_device_stats();
                        append_error(worker_prefix(worker_index, rt_device) +
                                     " YOLOv5 inference failed for request " +
                                     format_request_index(request_index) + ".");
                        return;
                    }

                    const auto detections = decode_yolov5_output(output_data,
                                                                 yolo->getInputWidth(),
                                                                 yolo->getInputHeight(),
                                                                 padded_width,
                                                                 padded_height,
                                                                 frame.cols,
                                                                 frame.rows,
                                                                 decode_config);

                    // Persist the optional visual overlay for the first configured requests.
                    std::string overlay_relative_path;
                    if (save_limit > 0 && request_index < static_cast<size_t>(save_limit)) {
                        if (!save_detection_overlay(image_input,
                                                    request_index,
                                                    output_dir,
                                                    detections,
                                                    labels,
                                                    overlay_relative_path)) {
                            record_device_stats();
                            append_error(worker_prefix(worker_index, rt_device) +
                                         " failed to save overlay for request " +
                                         format_request_index(request_index) + ".");
                            return;
                        }
                    }

                    const auto request_end = std::chrono::steady_clock::now();
                    const double wall_latency_ms =
                        std::chrono::duration<double, std::milli>(request_end - request_start).count();

                    // Store ordered per-request data for final report generation.
                    RequestResult result;
                    result.processed = true;
                    result.request_index = request_index;
                    result.worker_index = worker_index;
                    result.device_id = rt_device;
                    result.image_relative_path = image_input.relative_path;
                    result.detection_count = detections.size();
                    result.detections = format_detection_list(detections, labels);
                    result.overlay_path = overlay_relative_path;
                    result.model_inference_ms = model_inference_ms;
                    result.wall_latency_ms = wall_latency_ms;

                    {
                        std::lock_guard<std::mutex> results_lock(results_mtx);
                        results[request_index] = std::move(result);
                    }

                    // Accumulate worker-local counters used by the device summary table.
                    ++requests_done;
                    detections_done += detections.size();
                    wall_latency_sum_ms += wall_latency_ms;
                    model_sum_ms += static_cast<double>(model_inference_ms);
                    completed_requests.fetch_add(1, std::memory_order_relaxed);
                } catch (const std::exception& e) {
                    record_device_stats();
                    append_error(worker_prefix(worker_index, rt_device) +
                                 " request " + format_request_index(request_index) +
                                 " failed: " + e.what());
                    return;
                } catch (int error_code) {
                    record_device_stats();
                    append_error(worker_prefix(worker_index, rt_device) +
                                 " request " + format_request_index(request_index) +
                                 " failed with RPP error code " + std::to_string(error_code) + ".");
                    return;
                } catch (...) {
                    record_device_stats();
                    append_error(worker_prefix(worker_index, rt_device) +
                                 " request " + format_request_index(request_index) +
                                 " failed with an unknown exception.");
                    return;
                }
            }

            record_device_stats();
            {
                std::lock_guard<std::mutex> console_lock(console_mtx);
                clear_console_line();
                std::cout << worker_prefix(worker_index, rt_device) << " complete" << std::endl;
            }
        });
    }

    std::chrono::steady_clock::time_point start_time;
    const auto init_start_time = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> console_lock(console_mtx);
        std::cout << "STARTUP" << std::endl;
        print_initialization_line(
            0,
            static_cast<size_t>(device_count_to_use),
            init_start_time,
            false);
    }

    {
        // Wait until all workers have built engines and completed warmup.
        std::unique_lock<std::mutex> sync_lock(sync_mtx);
        while (ready_worker_count != static_cast<size_t>(device_count_to_use) &&
               !worker_failed.load(std::memory_order_acquire)) {
            sync_cv.wait_for(sync_lock, std::chrono::milliseconds(500));
            const size_t initialized_workers = ready_worker_count;
            if (initialized_workers == static_cast<size_t>(device_count_to_use) ||
                worker_failed.load(std::memory_order_acquire)) {
                break;
            }
            sync_lock.unlock();
            {
                std::lock_guard<std::mutex> console_lock(console_mtx);
                print_initialization_line(
                    initialized_workers,
                    static_cast<size_t>(device_count_to_use),
                    init_start_time,
                    false);
            }
            sync_lock.lock();
        }

        const size_t initialized_workers = ready_worker_count;
        sync_lock.unlock();
        {
            std::lock_guard<std::mutex> console_lock(console_mtx);
            print_initialization_line(
                initialized_workers,
                static_cast<size_t>(device_count_to_use),
                init_start_time,
                true);
        }
        sync_lock.lock();

        if (!worker_failed.load(std::memory_order_acquire)) {
            {
                // Print run settings only after model dimensions are known.
                std::lock_guard<std::mutex> dims_lock(dims_mtx);
                print_run_configuration(
                    model_path,
                    image_source_label,
                    output_dir,
                    image_inputs.size(),
                    total_requests,
                    inference_count,
                    static_cast<size_t>(workspace_mib),
                    decode_config,
                    save_limit,
                    resolved_device_ids,
                    dims_summary);
            }
            for (int worker_index = 0; worker_index < device_count_to_use; ++worker_index) {
                const int rt_device = resolved_device_ids[static_cast<size_t>(worker_index)];
                const size_t assigned_images =
                    per_worker_request_indices[static_cast<size_t>(worker_index)].size();
                std::cout << worker_prefix(worker_index, rt_device)
                          << " ready | assigned "
                          << assigned_images << " image requests" << std::endl;
            }
            // Show the initial 0/N progress bar before workers begin timed work.
            std::cout << "PROGRESS" << std::endl;
            print_progress_line(
                0,
                total_requests,
                inference_count,
                std::chrono::steady_clock::now(),
                false);
            start_time = std::chrono::steady_clock::now();
        }
        start_processing = true;
    }
    sync_cv.notify_all();

    // Refresh the progress line while workers process requests.
    std::atomic<bool> progress_done{false};
    std::thread progress_thread;
    if (!worker_failed.load(std::memory_order_acquire)) {
        progress_thread = std::thread([&]() {
            while (!progress_done.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (progress_done.load(std::memory_order_acquire)) {
                    break;
                }
                {
                    std::lock_guard<std::mutex> console_lock(console_mtx);
                    print_progress_line(
                        completed_requests.load(std::memory_order_acquire),
                        total_requests,
                        inference_count,
                        start_time,
                        false);
                }
            }

            {
                std::lock_guard<std::mutex> console_lock(console_mtx);
                print_progress_line(
                    completed_requests.load(std::memory_order_acquire),
                    total_requests,
                    inference_count,
                    start_time,
                    true);
            }
        });
    }

    // Join workers before printing final status and writing reports.
    for (auto& worker : workers) {
        worker.join();
    }
    progress_done.store(true, std::memory_order_release);
    if (progress_thread.joinable()) {
        progress_thread.join();
    }

    // Fail the run if any worker reported an error or did not fill every request slot.
    if (worker_failed.load(std::memory_order_acquire) ||
        completed_requests.load() != total_requests) {
        for (const auto& error : worker_errors) {
            std::cerr << error << std::endl;
        }
        std::cerr << "YOLOv5 multi-card inference did not finish successfully." << std::endl;
        return EXIT_FAILURE;
    }

    const auto end_time = std::chrono::steady_clock::now();
    const double wall_clock_sec = std::chrono::duration<double>(end_time - start_time).count();
    const auto wall_clock_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Convert per-worker accumulators into printable per-device performance rows.
    std::vector<DevicePerformance> device_performance;
    device_performance.reserve(static_cast<size_t>(device_count_to_use));
    for (int worker_index = 0; worker_index < device_count_to_use; ++worker_index) {
        const size_t requests_done = per_device_requests[static_cast<size_t>(worker_index)];
        const size_t detections_done = per_device_detections[static_cast<size_t>(worker_index)];
        const double elapsed_sec = per_device_elapsed_sec[static_cast<size_t>(worker_index)];
        const double latency_sum_ms = per_device_wall_latency_sum_ms[static_cast<size_t>(worker_index)];
        const double model_sum_ms = per_device_model_sum_ms[static_cast<size_t>(worker_index)];

        DevicePerformance stats;
        stats.worker_index = worker_index;
        stats.device_id = resolved_device_ids[static_cast<size_t>(worker_index)];
        stats.requests = requests_done;
        stats.detections = detections_done;
        stats.elapsed_sec = elapsed_sec;
        stats.throughput_ips =
            (elapsed_sec > 0.0) ? (static_cast<double>(requests_done) / elapsed_sec) : 0.0;
        stats.average_wall_latency_ms =
            (requests_done > 0) ? (latency_sum_ms / static_cast<double>(requests_done)) : 0.0;
        stats.average_model_ms =
            (requests_done > 0) ? (model_sum_ms / static_cast<double>(requests_done)) : 0.0;
        device_performance.emplace_back(stats);
    }

    // Aggregate throughput is measured from synchronized worker start to final join.
    const double aggregate_ips =
        (wall_clock_sec > 0.0)
            ? (static_cast<double>(completed_requests.load()) / wall_clock_sec)
            : 0.0;

    // Weighted averages use each worker's completed request count.
    double total_wall_latency_ms = 0.0;
    double total_model_ms = 0.0;
    size_t total_detections = 0;
    for (const auto& device : device_performance) {
        total_wall_latency_ms += device.average_wall_latency_ms * static_cast<double>(device.requests);
        total_model_ms += device.average_model_ms * static_cast<double>(device.requests);
        total_detections += device.detections;
    }
    const double average_wall_latency_ms =
        (completed_requests.load() > 0)
            ? (total_wall_latency_ms / static_cast<double>(completed_requests.load()))
            : 0.0;
    const double average_model_ms =
        (completed_requests.load() > 0)
            ? (total_model_ms / static_cast<double>(completed_requests.load()))
            : 0.0;

    // Ensure every ordered result slot was filled before writing text reports.
    for (size_t request_index = 0; request_index < results.size(); ++request_index) {
        if (!results[request_index].processed) {
            std::cerr << "Missing result for request index " << request_index << "." << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Save machine-readable text summaries, then print the terminal dashboard.
    if (!write_result_files(output_dir,
                            results,
                            device_performance,
                            wall_clock_ms,
                            wall_clock_sec,
                            aggregate_ips)) {
        return EXIT_FAILURE;
    }

    print_performance_summary(
        completed_requests.load(),
        wall_clock_ms,
        wall_clock_sec,
        aggregate_ips,
        inference_count,
        total_detections,
        average_wall_latency_ms,
        average_model_ms,
        device_performance);
    std::cout << "Result directory saved -> " << output_dir.string() << std::endl;

    return completed_requests.load() > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
