cmake_minimum_required(VERSION 3.10)

project(multi_card_yolov5_demo_workspace VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wno-write-strings -Wno-error=deprecated-declarations -Wno-deprecated-declarations -Werror")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(RppRT_DIR /usr/local/rpp)

link_directories(${RppRT_DIR}/lib)

add_executable(multi_card_yolov5_demo
    yolov5/main.cpp
    yolov5/yolo.cpp
    common/logger.cpp
    common/logging.cpp
)

target_include_directories(multi_card_yolov5_demo PRIVATE
    ${ROOT_DIRECTORY}/3rd_party
    ${ROOT_DIRECTORY}/common
    ${ROOT_DIRECTORY}/yolov5
    ${RppRT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(multi_card_yolov5_demo PRIVATE
    ${OpenCV_LIBS}
    RppRT nvonnxparser caffeparser urpp
    pthread
)

file(COPY "${ROOT_DIRECTORY}/image" DESTINATION "${CMAKE_BINARY_DIR}")
