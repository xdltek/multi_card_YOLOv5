/**
 * @file yolo.cpp
 * @brief YOLOv5 runtime wrapper implementation. See ../README.md for runtime workflow details.
 */
#include "yolo.h"
#include "parser_api.h"
#include "logger.h"
#include "sampleCommon.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

/**
 * @brief Release the worker-owned execution context and optional RPP context memory.
 */
Yolo::~Yolo()
{
    // Drop the execution context before freeing any context-owned device memory.
    context_ptr_.reset();
    if (context_device_memory_ != nullptr) {
        // rtFree: release device memory allocated for an explicit execution context.
        const rtError_t status = rtFree(context_device_memory_);
        if (status != rtSuccess) {
            std::cerr << "Warning: failed to free YOLOv5 context device memory." << std::endl;
        }
        context_device_memory_ = nullptr;
        context_device_memory_size_ = 0;
    }
}

/**
 * @brief Interpret common RPP binding ranks as channel/height/width metadata.
 * @param dims Binding dimensions reported by the RPP engine.
 * @param channels Output channel count.
 * @param height Output height.
 * @param width Output width.
 * @return True when dims can be interpreted by this demo.
 */
bool Yolo::parse_chw_dims(const infer1::Dims& dims, int& channels, int& height, int& width)
{
    channels = 0;
    height = 0;
    width = 0;

    if (dims.nbDims == 4) {
        channels = dims.d[1];
        height = dims.d[2];
        width = dims.d[3];
        return true;
    }
    if (dims.nbDims == 3) {
        channels = dims.d[0];
        height = dims.d[1];
        width = dims.d[2];
        return true;
    }
    if (dims.nbDims == 2) {
        channels = dims.d[0];
        height = dims.d[1];
        width = 1;
        return true;
    }
    if (dims.nbDims == 1) {
        channels = dims.d[0];
        height = 1;
        width = 1;
        return true;
    }

    return false;
}

/**
 * @brief Parse ONNX graph, configure builder options, and create runtime buffers.
 * @return True when engine creation and binding metadata initialization succeed.
 */
bool Yolo::init_engine() {
    try {

    // createInferBuilder: create the RPP builder used to parse and compile YOLOv5.
    std::unique_ptr<infer1::IBuilder> builder {infer1::createInferBuilder(sample::gLogger.getLogger())};
    if (builder == nullptr)
    {
        std::cerr << "Unable to create builder object." << std::endl;
        return false;
    }

    // createNetwork: allocate the network definition that receives parsed ONNX operators.
    std::unique_ptr<infer1::INetworkDefinition> network { builder->createNetwork() };
    if (network == nullptr)
    {
        std::cerr << "Unable to create network object." << std::endl;
        return false;
    }

    // createBuilderConfig: prepare workspace and precision options for engine compilation.
    std::unique_ptr<infer1::IBuilderConfig> config {builder->createBuilderConfig()};
    if (config == nullptr)
    {
        std::cerr << "Unable to create config object." << std::endl;
        return false;
    }

    // createParser: create the ONNX parser that imports the YOLOv5 graph.
    std::unique_ptr<onnxparser::IParser> parser { onnxparser::createParser(*network, sample::gLogger.getLogger()) };
    if (parser == nullptr)
    {
        std::cerr << "Unable to parse ONNX model file: " << onnx_model_path_ << std::endl;
        return false;
    }
    // onnx_parser: import the ONNX graph into the RPP network definition.
    if (onnx_parser(onnx_model_path_, builder.get(), network.get(), parser.get()) != 0) {
        return false;
    }

    // setMaxBatchSize: fix runtime execution to one image per request.
    builder->setMaxBatchSize(1);
    // setMaxWorkspaceSize: limit temporary card/runtime workspace used by this worker.
    config->setMaxWorkspaceSize(max_workspace_size_);
    // setFlag(kBF16): prefer BF16 execution when platform and model support it.
    config->setFlag(BuilderFlag::kBF16);

    // buildEngineWithConfig: compile the parsed YOLOv5 network into an executable engine.
    engine_ptr_ = std::shared_ptr<infer1::IEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine_ptr_.get()) {
        return false;
    }

    auto has_flag = config->getFlag(infer1::BuilderFlag::kINT8);
    if (has_flag) {
        // Inform users when quantized execution path is selected by model/config.
        sample::user_visible_log("Since the model includes quantization layers, inference is performed with int8 precision.");
    }

    // Query binding metadata used later by host/device buffer manager.
    for (int i =0; i < engine_ptr_->getNbBindings(); i++) {
        if (engine_ptr_->bindingIsInput(i)) {
            // Cache input binding metadata used for host buffer sizing and shape mapping.
            input_name_ = engine_ptr_->getBindingName(i);
            input_dimensions_ = engine_ptr_->getBindingDimensions(i);
            input_data_type_ = engine_ptr_->getBindingDataType(i);

            input_tensor_size_ = samplesCommon::volume(input_dimensions_);
            if (!parse_chw_dims(input_dimensions_, input_channels_, input_height_, input_width_)) {
                std::cerr << "Unsupported YOLOv5 input dimensions." << std::endl;
                return false;
            }
        }
        else {
            // Cache output binding metadata used for post-processing buffer reads.
            int output_channels = 0;
            int output_height = 0;
            int output_width = 0;
            output_name_ = engine_ptr_->getBindingName(i);
            output_dimensions_ = engine_ptr_->getBindingDimensions(i);
            output_data_type_ = engine_ptr_->getBindingDataType(i);

            output_tensor_size_ = samplesCommon::volume(output_dimensions_);
            if (!parse_chw_dims(output_dimensions_, output_channels, output_height, output_width)) {
                std::cerr << "Unsupported YOLOv5 output dimensions." << std::endl;
                return false;
            }
        }
    }
    if (input_name_.empty() || output_name_.empty()) {
        std::cerr << "YOLOv5 model must expose at least one input and one output binding." << std::endl;
        return false;
    }
    if (input_data_type_ != infer1::DataType::kFLOAT || output_data_type_ != infer1::DataType::kFLOAT) {
        std::cerr << "This demo expects float input and output tensors." << std::endl;
        return false;
    }
    // Allocate host/device buffer manager based on final engine bindings.
    buffer_ptr_ = std::make_shared<samplesCommon::RppBufferManager>(engine_ptr_);
    if (buffer_ptr_ == nullptr) {
        return false;
    }

    // getDeviceMemorySize: report context memory demand for diagnostics and future tuning.
    context_device_memory_size_ = engine_ptr_->getDeviceMemorySize();
    // createExecutionContext: create reusable execution state for this worker/card pair.
    context_ptr_.reset(engine_ptr_->createExecutionContext());
    if (context_ptr_ == nullptr) {
        std::cerr << "Unable to create YOLOv5 execution context." << std::endl;
        return false;
    }
    return true;
    } catch (const std::exception& e) {
        std::cerr << "YOLOv5 engine initialization failed: " << e.what() << std::endl;
        return false;
    } catch (int error_code) {
        std::cerr << "YOLOv5 engine initialization failed with RPP error code: "
                  << error_code << std::endl;
        return false;
    } catch (...) {
        std::cerr << "YOLOv5 engine initialization failed with an unknown RPP error." << std::endl;
        return false;
    }
}

/**
 * @brief Copy input tensor to device, execute inference, and read back output tensor.
 * @param input_tensor Input tensor produced by preprocessing.
 * @param inference_count Number of execution loops used for profiling.
 * @param output_data Output vector that receives flattened model output.
 * @param average_ms Output average model execution time in milliseconds.
 * @return True when inference and host output collection complete.
 */
bool Yolo::infer(const std::vector<float>& input_tensor,
                 int inference_count,
                 std::vector<float>& output_data,
                 float& average_ms) {
    if (input_tensor.size() != input_tensor_size_) {
        std::cerr << "Input tensor size mismatch. Expected " << input_tensor_size_
                  << " floats, got " << input_tensor.size() << "." << std::endl;
        return false;
    }
    if (inference_count < 1) {
        inference_count = 1;
    }

    // Copy host input tensor into the runtime-managed host buffer.
    try {
        size_t input_data_size = input_tensor_size_ * sizeof(float);
        std::memcpy(buffer_ptr_->getHostBuffer(input_name_), input_tensor.data(), input_data_size);

        // copyInputToDevice: copy prepared host input into RPP device memory.
        buffer_ptr_->copyInputToDevice();

        bool ok = true;
        if (!warmed_up_) {
            // execute: run one warmup inference before timed measurements.
            ok = context_ptr_->execute(1, buffer_ptr_->getDeviceBindings().data());
            if (!ok)
            {
                sample::LOG_ERROR() << "YOLOv5 warmup inference failed." << std::endl;
                return false;
            }
            warmed_up_ = true;
        }

        samplesCommon::PreciseCpuTimer infer_timer;
        infer_timer.start();

        // Execute multiple rounds for stable latency/FPS measurement.
        for (int i = 0; i < inference_count; i++)
        {
            // execute: run one timed YOLOv5 inference on the selected RPP card.
            ok = context_ptr_->execute(1, buffer_ptr_->getDeviceBindings().data());
            if (!ok)
            {
                sample::LOG_ERROR() << "YOLOv5 execute returned false." << std::endl;
                return false;
            }
        }

        infer_timer.stop();
        average_ms = infer_timer.milliseconds() / static_cast<float>(inference_count);

        // copyOutputToHost: copy raw YOLOv5 output rows back for CPU post-processing.
        buffer_ptr_->copyOutputToHost();

        // Flatten output tensor from host buffer into std::vector for downstream decode logic.
        const float* output_data_ptr = static_cast<const float*>(buffer_ptr_->getHostBuffer(output_name_));
        output_data.assign(output_data_ptr, output_data_ptr + output_tensor_size_);
    } catch (const std::exception& e) {
        std::cerr << "YOLOv5 RPP inference failed: " << e.what() << std::endl;
        return false;
    } catch (int error_code) {
        std::cerr << "YOLOv5 RPP inference failed with error code: " << error_code << std::endl;
        return false;
    } catch (...) {
        std::cerr << "YOLOv5 RPP inference failed with an unknown error." << std::endl;
        return false;
    }

    return true;
}
