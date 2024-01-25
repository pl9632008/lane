#include "customthread.h"

int CustomThread::get_num() {
    std::lock_guard<std::shared_mutex> lock(mtx);
    return currentStream;
}
    
void CustomThread::set_num(int stm){
    std::lock_guard<std::shared_mutex> lock(mtx);
    currentStream = stm;
}

void CustomThread::SerialThread() {
    while (run_flag ) {
 
        uint8_t *serialData = new uint8_t[16];
        
       
        try {

            ser.read(serialData, 16);
        }
        catch(std::exception & e){
            std::cerr << "Unhandled Exception: " << e.what() << std::endl; 
            break;
        }
  

        if ((serialData[0] == 0x01 && serialData[1] == 0x02 && serialData[2] == 0x03 && serialData[3] == 0x01 && serialData[4] == 0x00 && serialData[5] == 0x00 && serialData[6] == 0x29 && serialData[7] == 0x8E &&
            serialData[8] == 0x01 && serialData[9] == 0x02 && serialData[10] == 0x03 && serialData[11] == 0x00 && serialData[12] == 0x00 && serialData[13] == 0x00 && serialData[14] == 0x78 && serialData[15] == 0x4E
            ) ||
            (serialData[0] == 0x01 && serialData[1] == 0x02 && serialData[2] == 0x03 && serialData[3] == 0x00 && serialData[4] == 0x00 && serialData[5] == 0x00 && serialData[6] == 0x78 && serialData[7] == 0x4E &&
                serialData[8] == 0x01 && serialData[9] == 0x02 && serialData[10] == 0x03 && serialData[11] == 0x01 && serialData[12] == 0x00 && serialData[13] == 0x00 && serialData[14] == 0x29 && serialData[15] == 0x8E
                )) {
            int serial_currentStream = get_num();
            int temp = (serial_currentStream % 3) + 1;
            set_num(temp);
            std::cout << "Press button, currentStream = " << currentStream << std::endl;
        }
     /*   for (int i = 0; i < 16; i++) {
            printf("%#X ", serialData[i]);
            if (i == 15) {
                printf("\n");
            }
        }*/
        delete[] serialData;
    }
}

void CustomThread::capToFrame1(cv::VideoCapture & cap_11) {

    while (run_flag){
        if (get_num() == 1) {
            cv::Mat img_1;
            cap_11 >> img_1;
            putFrame1Thread(img_1);
        }
    }
}

void CustomThread::capToFrame2(cv::VideoCapture & cap_22) {

    while (run_flag) {
        if (get_num() == 2) {
            cv::Mat img_2;
            cap_22 >> img_2;
            putFrame2Thread(img_2);
        
        }

    }
}

void CustomThread::capToFrame3(cv::VideoCapture & cap_33) {
    while (run_flag){
        if (get_num() == 3) {
            cv::Mat img_3;
            cap_33 >> img_3;
            putFrame3Thread(img_3);
        }
  
    }
}

void CustomThread::capToFrame4(cv::VideoCapture & cap_44) {

    while (run_flag){
        if (get_num() == 3) {
            cv::Mat img_4;
            cap_44 >> img_4;
            putFrame4Thread(img_4);
        
        }

    }

}


void CustomThread::putFrame1Thread(cv::Mat & frame_1) {
    std::lock_guard<std::mutex> lock(mtx1);
    if (frame_1.empty()) {
        while (!deque_frame1.empty()){

            deque_frame1.pop_front();
        }
        return;
    }
    if (deque_frame1.size() > 0) {
        deque_frame1.pop_front();
    }
    deque_frame1.push_back(frame_1);
    return;
}

void CustomThread::putFrame2Thread(cv::Mat & frame_2) {
    std::lock_guard<std::mutex> lock(mtx2);
    if (frame_2.empty()) {
        while (!deque_frame2.empty()) {

            deque_frame2.pop_front();
        }
        return;
    }

    if (deque_frame2.size() > 0) {
        deque_frame2.pop_front();
    }
    deque_frame2.push_back(frame_2);
    return;
}

void CustomThread::putFrame3Thread(cv::Mat & frame_3) {
    std::lock_guard<std::mutex> lock(mtx3);
    if (frame_3.empty()) {
        while (!deque_frame3.empty()) {
            deque_frame3.pop_front();
        }
        return;
    }

    if (deque_frame3.size() > 0) {
        deque_frame3.pop_front();
    }
    deque_frame3.push_back(frame_3);
    return;
}
void CustomThread::putFrame4Thread(cv::Mat & frame_4) {
    std::lock_guard<std::mutex> lock(mtx4);
    if (frame_4.empty()) {
        while (!deque_frame4.empty()) {
            deque_frame4.pop_front();
        }
        return;
    }

    if (deque_frame4.size() > 0) {
        deque_frame4.pop_front();
    }
    deque_frame4.push_back(frame_4);
    return;
}




GetFrame CustomThread::getFrame1() {
    std::lock_guard<std::mutex> lock(mtx1);
    GetFrame getframe1;
    if (!deque_frame1.empty()) {
        getframe1.frame = deque_frame1.back();
        getframe1.have_frame = true;
    }
    return getframe1;
}

GetFrame CustomThread::getFrame2() {
    std::lock_guard<std::mutex> lock(mtx2);
    GetFrame getframe2;
    if (!deque_frame2.empty()) {
        getframe2.frame = deque_frame2.back();
        getframe2.have_frame = true;
    }
    return getframe2;
}

GetFrame CustomThread::getFrame3() {
    std::lock_guard<std::mutex> lock(mtx3);
    GetFrame getframe3;
    if (!deque_frame3.empty()) {
        getframe3.frame = deque_frame3.back();
        getframe3.have_frame = true;
    }
    return getframe3;
}

GetFrame CustomThread::getFrame4() {
    std::lock_guard<std::mutex> lock(mtx4);
    GetFrame getframe4;
    if (!deque_frame4.empty()) {
        getframe4.frame = deque_frame4.back();
        getframe4.have_frame = true;
    }
    return getframe4;
}


void CustomThread::run(FindLane &findlane) {

   cv::namedWindow("Video Player", cv::WINDOW_NORMAL);
   cv::setWindowProperty("Video Player", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
   cv::resizeWindow("Video Player", cv::Size(window_width, window_height));

    while (true) {
    
        int temp = get_num();
        //std::cout << "currentStream = " << temp << std::endl;
        int key = cv::waitKey(5);
        if (key == 27) {
            break;
        }
        cv::Mat ans;
        if (temp == 1) {
            GetFrame getframe1 = getFrame1();

            if (getframe1.have_frame == false) {
                cv::Mat m = cv::Mat::zeros(window_height, window_width, CV_8UC3);
                std::string text = "Please check camera 1!";
                cv::putText(m, text, cv::Point(120, window_height/2), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));
                cv::imshow("Video Player", m);
                continue;
            }
            ans = findlane.doInference_det(getframe1.frame);
    
        }

        else if (temp == 2) {
            GetFrame getframe2 = getFrame2();

            if (getframe2.have_frame == false) {
                cv::Mat m = cv::Mat::zeros(window_height, window_width, CV_8UC3);
                std::string text = "Please check camera 2!";
                cv::putText(m, text, cv::Point(120, window_height / 2), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));
                cv::imshow("Video Player", m);
                continue;
            }
            ans = getframe2.frame;
            

        }
        else if (temp == 3) {
            GetFrame getframe3 = getFrame3();
            GetFrame getframe4 = getFrame4();
            
            if (getframe3.have_frame == false || getframe4.have_frame == false) {
                cv::Mat m = cv::Mat::zeros(window_height, window_width, CV_8UC3);
                std::string text = "Please check camera 3 and 4!";
                cv::putText(m, text, cv::Point(120, window_height / 2), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));
                cv::imshow("Video Player", m);
                continue;
            }
            ans = findlane.get_img_mask(getframe3.frame, getframe4.frame);

        }
        cv::resize(ans, ans, cv::Size(window_width, window_height));
        cv::imshow("Video Player", ans);
    

    }
    run_flag = false;

}
