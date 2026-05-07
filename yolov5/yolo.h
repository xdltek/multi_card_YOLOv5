/**
 * @file yolo.h
 * @brief YOLOv5 inference wrapper declaration. See ../README.md for runtime workflow details.
 */
#ifndef XDLTEK_SAMPLES_YOLO_H
#define XDLTEK_SAMPLES_YOLO_H

#include "rpp_buffer_manager.h"

#include <memory>
#include <string>
#include <vector>


class Yolo {
public:
    /**
     * @brief Store model path and RPP workspace limit for one worker-owned YOLOv5 runtime.
     * @param onnx_path YOLOv5 ONNX model path resolved by the CLI.
     * @param max_workspace_size RPP builder workspace size in bytes for this worker.
     */
    explicit Yolo(const std::string& onnx_path,
                  size_t max_workspace_size = 256ULL * 1024ULL * 1024ULL)
        : onnx_model_path_(onnx_path)
        , max_workspace_size_(max_workspace_size) {}

    /**
     * @brief Release the execution context and any explicitly allocated RPP context memory.
     */
    ~Yolo();

    /**
     * @brief Build and initialize the inference engine from the ONNX model.
     * @return True when engine and buffers are initialized successfully.
     */
    bool init_engine();

    /**
     * @brief Run inference on a preprocessed input tensor and collect output data.
     * @param input_tensor Flattened input tensor matching getInputSize().
     * @param inference_count Number of timed inference iterations after warmup.
     * @param output_data Output vector filled with raw model results.
     * @param average_ms Output average model execution time in milliseconds.
     * @return True when execution and output copy succeed.
     */
    bool infer(const std::vector<float>& input_tensor,
               int inference_count,
               std::vector<float>& output_data,
               float& average_ms);

    /** @brief Return the model input channel count used by preprocessing. */
    int getInputChannels() const { return input_channels_; }

    /** @brief Return the model input width used by preprocessing. */
    int getInputWidth() const { return input_width_; }

    /** @brief Return the model input height used by preprocessing. */
    int getInputHeight() const { return input_height_; }

    /** @brief Return the flattened input tensor element count. */
    size_t getInputSize() const { return input_tensor_size_; }

    /** @brief Return the flattened output tensor element count. */
    size_t getOutputSize() const { return output_tensor_size_; }

    /** @brief Return RPP execution-context device memory size reported by the engine. */
    size_t getContextDeviceMemorySize() const { return context_device_memory_size_; }

    /** @brief Return raw RPP input binding dimensions for run-configuration output. */
    infer1::Dims getInputDimensions() const { return input_dimensions_; }

    /** @brief Return raw RPP output binding dimensions for run-configuration output. */
    infer1::Dims getOutputDimensions() const { return output_dimensions_; }

private:
    /**
     * @brief Interpret RPP binding dimensions as channel/height/width metadata.
     * @param dims Binding dimensions reported by the RPP engine.
     * @param channels Output channel count.
     * @param height Output height.
     * @param width Output width.
     * @return True when this demo can map the rank to C/H/W values.
     */
    static bool parse_chw_dims(const infer1::Dims& dims, int& channels, int& height, int& width);

    std::shared_ptr<infer1::IEngine> engine_ptr_ {nullptr};
    std::shared_ptr<samplesCommon::RppBufferManager> buffer_ptr_ {nullptr};
    std::unique_ptr<infer1::IExecutionContext> context_ptr_ {nullptr};

    int input_channels_ = 0;
    int input_width_ = 0;
    int input_height_ = 0;
    size_t input_tensor_size_ = 0;
    size_t output_tensor_size_ = 0;
    size_t max_workspace_size_ = 256ULL * 1024ULL * 1024ULL;
    size_t context_device_memory_size_ = 0;
    void* context_device_memory_ = nullptr;
    std::string onnx_model_path_;

    std::string input_name_;
    std::string output_name_;
    infer1::Dims input_dimensions_;
    infer1::Dims output_dimensions_;
    infer1::DataType input_data_type_;
    infer1::DataType output_data_type_;
    bool warmed_up_ = false;

};


#endif //XDLTEK_SAMPLES_YOLO_H
