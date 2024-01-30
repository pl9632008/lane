
#ifndef LANE_INCLUDE_SEGMENTATION_H_
#define LANE_INCLUDE_SEGMENTATION_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <fstream>
#include <algorithm>
//#define _CRT_SECURE_NO_WARNINGS  //local_time

#include "cuda_runtime_api.h"
#include "NvInfer.h"

using namespace nvinfer1;


struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float>maskdata;
    cv::Mat mask;
};


class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class Segmentation {
public:
    void loadEngine(const std::string& path);
    cv::Mat preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh);
    cv::Mat doInference(cv::Mat& org_img);

    void loadEngine_det(const std::string& path);
    cv::Mat doInference_det(cv::Mat& org_img);


    void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<Object>& faceobjects);
    float intersection_area(Object& a, Object& b);
    void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
    float CONF_THRESHOLD = 0.52;

private:
    Logger logger_;
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;


    const int BATCH_SIZE = 1;
    const int CHANNELS = 3;
    const int INPUT_H = 640;
    const int INPUT_W = 640;

    const int OUTPUT0_BOXES = 8400;
    const int OUTPUT0_ELEMENT = 37;
    const int CLASSES = 1;

    const int OUTPUT1_CHANNELS = 32;
    const int OUTPUT1_H = 160;
    const int OUTPUT1_W = 160;

    const char* images_ = "images";
    const char* output0_ = "output0";
    const char* output1_ = "output1";

   

    IRuntime* runtime_det = nullptr;
    ICudaEngine* engine_det = nullptr;
    IExecutionContext* context_det = nullptr;
    const int CLASSES_DET = 80;
    const int OUTPUT0_ELEMENT_DET = 84;

    const char * class_names[80]={
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
      "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
      "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
       "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
       "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
       "hair drier", "toothbrush"
    };

};


#endif //LANE_INCLUDE_SEGMENTATION_H_