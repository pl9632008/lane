#include "customthread.h"

int main() {

    FindLane findlane;
    std::string path = "/home/nvidia/customlane/models/lanetrans.engine";
    std::string path_det = "/home/nvidia/customlane/models/yolov8s.engine";

    findlane.loadEngine(path);
    findlane.loadEngine_det(path_det);
    auto mp = findlane.readConfig();

    cv::VideoCapture cap_11(mp["rtsp_1"]);
    cv::VideoCapture cap_22(mp["rtsp_2"]);
    cv::VideoCapture cap_33(mp["rtsp_3"]);
    cv::VideoCapture cap_44(mp["rtsp_4"]);

    //int WIDTH_1 = cap_11.get(cv::CAP_PROP_FRAME_WIDTH);
    //int HEIGHT_1 = cap_11.get(cv::CAP_PROP_FRAME_HEIGHT);
    //double fps_1 = cap_11.get(cv::CAP_PROP_FPS);
    //int WIDTH_2 = cap_22.get(cv::CAP_PROP_FRAME_WIDTH);
    //int HEIGHT_2 = cap_22.get(cv::CAP_PROP_FRAME_HEIGHT);
    //double fps_2 = cap_22.get(cv::CAP_PROP_FPS);
    // 
    //time_t now;
    //time(&now);
    //tm* ltm = localtime(&now);

    //std::string current_time = std::to_string(ltm->tm_hour) + "_" + std::to_string(ltm->tm_min) + "_" + std::to_string(ltm->tm_sec);
    //std::string str_1 = std::string("F:/test_line/Project1/outvideo/") + "out1_" + current_time + ".avi";
    //std::string str_2 = std::string("F:/test_line/Project1/outvideo/") + "out2_" + current_time + ".avi";

    // cv::VideoWriter writer_1(str_1, cv::VideoWriter::fourcc('X', 'V','I','D'), fps_1, cv::Size(WIDTH_1, HEIGHT_1));
    // cv::VideoWriter writer_2(str_2, cv::VideoWriter::fourcc('X', 'V','I','D'), fps_2, cv::Size(WIDTH_2, HEIGHT_2));

    CustomThread customthread;

    std::thread serial_thread(&CustomThread::SerialThread,& customthread);
    std::thread frame1_thread(&CustomThread::capToFrame1, &customthread, std::ref(cap_11));
    std::thread frame2_thread(&CustomThread::capToFrame2, &customthread, std::ref(cap_22));
    std::thread frame3_thread(&CustomThread::capToFrame3, &customthread, std::ref(cap_33));
    std::thread frame4_thread(&CustomThread::capToFrame4, &customthread, std::ref(cap_44));
    std::thread run_thread(&CustomThread::run, &customthread, std::ref(findlane));

    serial_thread.join();
    frame1_thread.join();
    frame2_thread.join();
    frame3_thread.join();
    frame4_thread.join();
    run_thread.join();

   
}

