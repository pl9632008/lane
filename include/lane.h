#ifndef LANE_LANE_H_
#define LANE_LANE_H_

#include "segmentation.h"


struct ExtractOut {
    std::vector<cv::Point2f> pts;

};

class FindLane : public Segmentation {
public:

    ExtractOut extract_color_lanes(cv::Mat& img);
    cv::Mat get_img_mask(cv::Mat& img_1, cv::Mat& img_2);
    void readConfig();

    std::map<std::string, std::string> config_mp;


    std::string config_path = "/home/nvidia/customlane/config.txt";
    
    //const cv::Size cam_size = cv::Size(1920, 1080);
    cv::Size cam_size = cv::Size(1280, 720) ;
    double sca_1 = 640.0 / cam_size.width;
    double sca_2 = sca_1;
    double mm_pre_x = 0.206896; //每个像素对应实际的距离cm
    double mm_pre_y = 15.0 / 23.0 / sca_1;

    int stand_bar_x = 450;
    //int stand_bar_x = 470;
    int stand_bar_y = 360;


    cv::Point stand_bar_p = cv::Size(stand_bar_x, stand_bar_y);

};



#endif //LANE_LANE_H_