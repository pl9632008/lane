cmake_minimum_required(VERSION 3.0)
project(testlane)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(SRC src/lane.cpp
        src/segmentation.cpp
        src/customthread.cpp
        )
add_library(detectlane SHARED ${SRC})


find_package (OpenCV REQUIRED)
include_directories(include /usr/local/cuda-11.4/include ${OpenCV_INCLUDE_DIRS})
link_directories(lib /usr/local/cuda-11.4/lib64)

add_executable(a src/main.cpp ${SRC})

target_link_libraries(a nvinfer cudart ${OpenCV_LIBS} pthread serial )
