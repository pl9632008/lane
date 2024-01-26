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
        //ser.setPort("COM4");//win
        ser.setPort("/dev/ttyUSB0");//linux
        ser.setBaudrate(9600);
        serial::Timeout timeout = serial::Timeout::simpleTimeout(4000);
        ser.setTimeout(timeout);
        ser.open();
    }

    int get_num();
    void set_num(int stm);
    void SerialThread();
   
    void putFrame1Thread(cv::Mat & frame_1);
    void putFrame2Thread(cv::Mat & frame_2);
    void putFrame3Thread(cv::Mat & frame_3);
    void putFrame4Thread(cv::Mat & frame_4);


    void capToFrame1(cv::VideoCapture& cap_11);
    void capToFrame2(cv::VideoCapture& cap_22);
    void capToFrame3(cv::VideoCapture& cap_44);
    void capToFrame4(cv::VideoCapture& cap_44);
    void run(FindLane &findlane);

    GetFrame getFrame1();
    GetFrame getFrame2();
    GetFrame getFrame3();
    GetFrame getFrame4();

    
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
    serial::Serial ser;

    std::atomic<bool> run_flag = true;

    int window_width = 1280;
    int window_height = 720;
   

};

#endif //LANE_CUSTOMTHREAD_H_
