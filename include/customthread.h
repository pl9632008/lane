#ifndef LANE_CUSTOMTHREAD_H_
#define LANE_CUSTOMTHREAD_H_
#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include "serial/serial.h"
#include "lane.h"

struct GetFrame {
    cv::Mat frame;
    bool have_frame = false;

};

class CustomThread {
public:
    CustomThread() {
    }

    int get_num();
    void set_num(int stm);
    void SerialThread(serial::Serial &ser);
   
    void putFrame1Thread(cv::Mat & frame_1);
    void putFrame2Thread(cv::Mat & frame_2);
    void putFrame3Thread(cv::Mat & frame_3);
    void putFrame4Thread(cv::Mat & frame_4);


    void capToFrame1();
    void capToFrame2();
    void capToFrame3();
    void capToFrame4();
    void run(FindLane &findlane);

    GetFrame getFrame1();
    GetFrame getFrame2();
    GetFrame getFrame3();
    GetFrame getFrame4();

    std::map<std::string, std::string> config_mp;
private:
    std::shared_mutex mtx;
    int currentStream = 1;

    std::mutex mtx1;
    std::mutex mtx2;
    std::mutex mtx3;
    std::mutex mtx4;

    std::deque<cv::Mat>deque_frame1;
    std::deque<cv::Mat>deque_frame2;
    std::deque<cv::Mat>deque_frame3;
    std::deque<cv::Mat>deque_frame4;

    std::atomic<bool> run_flag = true;
    std::atomic<bool> saveimg_flag = false;

    int window_width = 1280;
    int window_height = 720;

    cv::VideoCapture cap_11;
    cv::VideoCapture cap_22;
    cv::VideoCapture cap_33;
    cv::VideoCapture cap_44;

   

};

#endif //LANE_CUSTOMTHREAD_H_
