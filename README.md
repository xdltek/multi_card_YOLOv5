![XDL Logo](doc/logo/logo_color_horizontal.png)

# Multi Card YOLOv5 C++ Inference Demo

`multi_card_yolov5_demo` is a standalone C++ RPP runtime demo for running YOLOv5 object detection across multiple RPP cards. The host process creates one worker thread per active device, binds each worker with `rtSetDevice`, builds a YOLOv5 runtime context on that card, then distributes image requests with balanced round-robin scheduling.

The main outputs are written under `final_demo_result` by default:

- `detections_result.txt`
- `performance_result.txt`
- `images/request_XXXXXX_<name>.jpg` for the first `--save-limit` requests

## Runtime Context

This demo uses RPP as the inference runtime. In this codebase, RPP is a proprietary compute card / accelerator runtime. The CPU host process loads images, builds NCHW float tensors, copies tensors to RPP card memory through `RppBufferManager`, executes the YOLOv5 ONNX engine, copies output tensors back to host memory, decodes boxes, runs NMS, and optionally saves overlay images.

Runtime concepts used by this demo:

| Term | Meaning in this demo |
| --- | --- |
| Host | CPU memory owned by the C++ process. OpenCV images, preprocessed tensors, decoded boxes, and text reports live here. |
| Device | RPP card memory managed through `RppBufferManager` and the RPP runtime. |
| Engine | Compiled YOLOv5 model created by `buildEngineWithConfig`. |
| Context | Reusable execution state created by `createExecutionContext` inside each worker. |
| Worker | One host thread bound to one physical RPP card with `rtSetDevice`. |

Important runtime APIs:

| API | Purpose |
| --- | --- |
| `rtGetDeviceCount` | Query visible RPP cards. |
| `rtSetDevice` | Bind a worker thread to one physical card. |
| `createInferBuilder` / `createNetwork` | Build the RPP network definition. |
| `createParser` / `onnx_parser` | Parse YOLOv5 ONNX into the RPP network. |
| `buildEngineWithConfig` | Compile the executable RPP engine. |
| `RppBufferManager` | Allocate host/device buffers for model bindings. |
| `copyInputToDevice` / `copyOutputToHost` | Move tensors between host and card memory. |
| `execute` | Run inference on the selected RPP card. |

## Model I/O

Standard YOLOv5 ONNX exports use one image input and one detection output:

| Tensor | Typical Shape | Meaning |
| --- | --- | --- |
| Input | `[3, 640, 640]` | RGB float tensor in CHW layout, normalized by `1/255`. |
| Output | `[25200, 85]` | Detection rows containing `[x,y,w,h,obj,80 class scores]` for COCO models. |

For custom class counts, pass `--labels` and `--class-count` so the decoder uses the correct row stride (`5 + class_count`).

## Project Structure

```text
multi_card_YOLOv5/
|-- 3rd_party/argparse/
|-- common/
|-- doc/logo/
|-- image/
|   `-- test_1.png
|-- yolov5/
|   |-- main.cpp
|   |-- yolo.cpp
|   `-- yolo.h
|-- CMakeLists.txt
`-- README.md
```

## Workflow

High-level phases:

1. Parse CLI options and resolve model, image, output, and device settings.
2. Discover input images from `--image-dir` or use one file from `--image`.
3. Resolve `--device-list` or `--device-count` against available RPP cards.
4. Build a balanced round-robin dispatch plan where each image request is one unit of work.
5. Start one worker thread per active device and bind it with `rtSetDevice`.
6. Each worker builds and warms its own YOLOv5 engine/context on that device.
7. For each assigned request, load an image, pad to square, resize to model input, convert BGR to RGB, normalize by `1/255`, and pack CHW tensor data.
8. Run YOLOv5 inference on the worker's RPP card.
9. Decode `[x,y,w,h,obj,class_scores...]` rows, filter candidates, apply NMS, and clip boxes to the original image.
10. Save ordered detection and performance reports, plus optional overlay images.

Diagram workflow:

```text
+--------------------------+
| multi_card_yolov5 CLI    |
+------------+-------------+
             |
             v
+--------------------------+
| Resolve model/images     |
| Resolve RPP devices      |
+------------+-------------+
             |
             v
+--------------------------+
| Build round-robin        |
| image request plan       |
+------------+-------------+
             |
             v
  +------------------------+
  | worker per device      |
  | - rtSetDevice          |
  | - build/warm YOLOv5    |
  +-----------+------------+
              |
              v
+--------------------------+
| Load image with OpenCV   |
| pad, resize, RGB, CHW    |
+------------+-------------+
             |
             v
+--------------------------+
| YOLOv5 inference         |
| copyInputToDevice        |
| execute                  |
| copyOutputToHost         |
+------------+-------------+
             |
             v
+--------------------------+
| Decode boxes             |
| threshold + NMS          |
+------------+-------------+
             |
             v
+--------------------------+
| Save detections,         |
| overlays, performance    |
+--------------------------+
```

## Function Summary

| Phase | Function | Purpose |
| --- | --- | --- |
| Path setup | `resolve_existing_path` / `resolve_output_path` | Find model, image, label, and result paths from build or source directories. |
| Input discovery | `collect_image_files` | Recursively collect supported still images in deterministic order. |
| Device discovery | `get_available_device_count` / `parse_device_list` | Resolve the active RPP cards requested by CLI options. |
| Scheduling | `build_balanced_dispatch_plan` | Assign image requests across worker threads. |
| Runtime wrapper | `Yolo::init_engine` | Parse ONNX, build the RPP engine, allocate buffers, and create an execution context. |
| Runtime wrapper | `Yolo::infer` | Copy input, execute YOLOv5, time model execution, and copy output back. |
| Preprocess | `format_yolov5` / `build_yolov5_tensor` | Pad images, resize to model input, convert BGR to RGB, normalize, and pack CHW data. |
| Postprocess | `decode_yolov5_output` / `nms_boxes` | Decode YOLO rows, filter by thresholds, apply NMS, and clip boxes. |
| Visualization | `save_detection_overlay` | Save the single per-request overlay image under `final_demo_result/images`. |
| Reports | `write_result_files` | Write ordered detection and performance text reports. |
| Multi-card worker | worker lambda in `main` | Bind one thread to one device, build/warm its model, process assigned requests, and collect stats. |

## Build

Required:

- CMake 3.10+
- C++17 compiler
- OpenCV with `core`, `imgproc`, and `imgcodecs`
- RppRT installed under `/usr/local/rpp`

Build commands:

```bash
cd multi_card_YOLOv5
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

The executable is:

```bash
./multi_card_yolov5_demo
```

## Run

The YOLOv5 ONNX model is not bundled. Pass your exported model with `--onnx`:

```bash
cd multi_card_YOLOv5/build
./multi_card_yolov5_demo \
  --onnx /path/to/yolov5.onnx \
  --image ../image/test_1.png
```

Use all available RPP cards by default, or control placement explicitly:

```bash
./multi_card_yolov5_demo --onnx /path/to/yolov5.onnx --image-dir image -n 100
./multi_card_yolov5_demo --onnx /path/to/yolov5.onnx --image-dir image -n 100 -d 4
./multi_card_yolov5_demo --onnx /path/to/yolov5.onnx --image-dir image -n 100 --device-list 0,2,3
```

Useful options:

| Option | Description |
| --- | --- |
| `-o`, `--onnx` | Path to YOLOv5 ONNX model. Required. |
| `-i`, `--image-dir` | Directory of input images. Default: `image`. |
| `--image` | Single input image; overrides `--image-dir`. |
| `-j`, `--output`, `--output-json` | Output result directory. Default: `final_demo_result`. |
| `-n`, `--image-count` | Number of image requests; `0` processes each discovered image once. |
| `--loop` | Timed execute loop count for each image inference. |
| `-d`, `--device-count` | RPP device count to use; `0` means all available devices. |
| `--device-list` | Comma-separated device ids, for example `0,1,3`. |
| `--labels` | Optional class-name file, one label per line. Default is embedded COCO-80. |
| `--class-count` | Number of classes in the YOLOv5 output. `0` uses the loaded label count. |
| `--confidence-threshold` | Objectness threshold before class-score filtering. Default: `0.4`. |
| `--score-threshold` | Best-class score threshold. Default: `0.2`. |
| `--nms-threshold` | IoU threshold for non-maximum suppression. Default: `0.4`. |
| `--max-detections` | Maximum retained detections per image; `0` disables the cap. |
| `--save-limit` | Maximum overlay images to save; `0` disables overlay output. Default: `100`. |
| `--workspace-mib` | RPP builder workspace size per worker. Default: `256`. Reduce this if context creation reports card-memory allocation error 2. |
| `-v`, `--verbose` | Enable verbose RPP logging. |

If a larger model such as YOLOv5m reports RPP memory allocation error 2 during startup or warmup, the RPP runtime/card could not allocate the execution context for that engine. Use a smaller export such as YOLOv5s, a quantized model, fewer active devices, or free card memory before retrying.

## Output

`detections_result.txt` contains one ordered line per request:

```text
Request 000000 | worker=0 | device=0 | image=test_1.png | detections=3 | overlay=images/request_000000_test_1.jpg | boxes=(person,0.812,42,31,120,240);...
```

`performance_result.txt` reports per-request model timing, worker/device placement, wall latency, and a final per-device summary:

```text
Request 000000 | worker=0 | device=0 | image=test_1.png | detection_count=3 | yolo_inference_ms=1.234 | wall_latency_ms=5.678
Summary | wall_clock_ms=1834 | wall_clock_sec=1.834 | aggregate_images_per_sec=54.526
Columns | device=physical RPP device id | worker=logical host thread index | requests=image inference requests processed by this worker | detections=retained boxes after thresholding and NMS | elapsed_sec=worker wall-clock runtime | images_per_sec=per-worker throughput | avg_wall_latency_ms=average end-to-end request latency | avg_yolo_ms=average YOLOv5 model execute time
Device 0 | worker=0 | requests=25 | detections=73 | elapsed_sec=1.821 | images_per_sec=13.729 | avg_wall_latency_ms=72.840 | avg_yolo_ms=1.234
```

`yolo_inference_ms` is model execution time only. Wall latency includes image loading, preprocessing, output decoding, NMS, optional overlay writing, and the requested execute loop.

The terminal output includes startup initialization progress, a run-configuration panel, worker ready/complete lines, a live inference progress bar, a performance dashboard, and a final saved-path line. Per-request timing lines are kept in `performance_result.txt`.

```text
STARTUP
  init     [================] 1/1 workers | loading YOLOv5 engines + warmup | 2.7s elapsed | ready
==============================================================================
XDL MULTI-CARD YOLOV5 // RUN CONFIG
YOLOv5 object detection across multiple devices with structured result capture
------------------------------------------------------------------------------
  Model              : /path/to/yolov5s.onnx
  Image Source       : image
  Tensor Layout      : [3, 640, 640] -> [25200, 85]
  Workload           : 1 discovered / 1 inference requests
  Loop Count         : 1 timed executes per image
  Workspace          : 256 MiB per worker
  Thresholds         : obj=0.40 cls=0.20 nms=0.40
  Classes            : 80
  Save Limit         : 100 overlays
  Devices            : 0 (1 active)
  Dispatch           : balanced round-robin
  Result Directory   : final_demo_result
==============================================================================
[worker 00 | device 00] ready | assigned 1 image requests
PROGRESS
  progress [================] 1/1 img | 100.0% | 1.997 img/s | 0.5s elapsed | done | loop=1
[worker 00 | device 00] complete
==============================================================================
PERFORMANCE DASHBOARD
High-signal run summary with aggregate throughput, latency, and device-level distribution
------------------------------------------------------------------------------
  [THROUGHPUT]     1.997 img/s   1 completed requests across 1 devices
  [LOOP COUNT]     1   timed executes per YOLOv5 image request
  [DETECTIONS]     13   retained boxes after thresholding and NMS
  [LATENCY]        67.140 ms   average per image request
  [YOLOV5]         5.506 ms   average model execute time, measured with loop=1
  [WALL CLOCK]     500 ms   0.501 s end-to-end

DEVICE BREAKDOWN
  columns: card=physical RPP device id, worker=logical host thread, util=relative per-worker throughput bar, imgs=image requests, boxes=retained detections, img/s=per-worker throughput, lat_ms=average request wall latency, yolo_ms=average model execute time, elapsed_s=worker wall-clock runtime
  card worker           util  imgs boxes    img/s   lat_ms  yolo_ms elapsed_s
     0      0 [============]     1    13   14.891   67.140    5.506     0.067
==============================================================================
Result directory saved -> final_demo_result
```

## Notes

The embedded label list matches standard YOLOv5 COCO-80 ordering. For custom models, pass `--labels` and `--class-count` so the output row stride matches `5 + class_count`.
