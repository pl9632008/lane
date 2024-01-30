#include "customthread.h"

int main() {

    FindLane findlane;
    std::string path = "/home/nvidia/customlane/models/lane2.engine";
    std::string path_det = "/home/nvidia/customlane/models/yolov8s.engine";

    findlane.loadEngine(path);
    findlane.loadEngine_det(path_det);
    findlane.readConfig();

    CustomThread customthread;
    customthread.config_mp = findlane.config_mp;
    serial::Serial ser;
    try{
        
        //ser.setPort("COM4");//win
        ser.setPort("/dev/ttyUSB0");//linux
        ser.setBaudrate(9600);
        serial::Timeout timeout = serial::Timeout::simpleTimeout(4000);
        ser.setTimeout(timeout);
        ser.open();
        
        
    } catch(std::exception & e){
            std::cerr << "Unhandled Exception: " << e.what() << std::endl; 
          
    }
    std::thread serial_thread(&CustomThread::SerialThread,& customthread, std::ref(ser));
    std::thread frame1_thread(&CustomThread::capToFrame1, &customthread);
    std::thread frame2_thread(&CustomThread::capToFrame2, &customthread);
    std::thread frame3_thread(&CustomThread::capToFrame3, &customthread);
    std::thread frame4_thread(&CustomThread::capToFrame4, &customthread);
    std::thread run_thread(&CustomThread::run, &customthread, std::ref(findlane));

    serial_thread.join();
    frame1_thread.join();
    frame2_thread.join();
    frame3_thread.join();
    frame4_thread.join();
    run_thread.join();

   
}

