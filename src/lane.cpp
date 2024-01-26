#include "lane.h"


ExtractOut FindLane::extract_color_lanes(cv::Mat& img) {

    ExtractOut extractout;
    auto res = doInference(img);

    int differ = 30;
    for (int row = 0; row < res.rows; row++) {
        for (int col = 0; col < res.cols; col++) {
            if (col >= stand_bar_x + differ) {
                res.at<uint8_t>(row, col) = 0;
            }

        }
    }

  /*  cv::imshow("res", res);*/

    cv::Mat mask = res;

    // 获取连通区域信息
    cv::Mat labels, stats, centroids;
    int n_components = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

    // 定义面积阈值
    int area_threshold = 100;

    // 遍历每个连通区域的面积
    for (int i = 1; i < n_components; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < area_threshold) {
            // 将该区域所对应的像素值设为背景（黑色）
            labels.setTo(0, labels == i);
        }
    }
    // 将标记后的掩码转换为二值掩码，白色表示前景区域，黑色表示背景区域
    cv::Mat result_mask = cv::Mat::zeros(mask.size(), CV_8U);
    result_mask.setTo(255, labels > 0);


    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat dilation, erosion;
    cv::dilate(result_mask, dilation, kernel);
    cv::erode(dilation, erosion, kernel);


    // 对区域取最大连通域
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(erosion, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) > cv::contourArea(b);
        });
    cv::Mat max_erosion = cv::Mat::zeros(erosion.size(), CV_8U);
    cv::drawContours(max_erosion, contours, 0, 255, cv::FILLED);

    //cv::imshow("max_erosion", max_erosion);


    std::vector<cv::Point> right_points;
    std::vector<cv::Point> left_points;
    for (int row = 2; row < img.rows - 2; row++) {
        for (int col = stand_bar_x + differ; col >= 0; col--) {
            if (max_erosion.at<uint8_t>(row, stand_bar_x - 1 + differ) == 255)
            {
                break;
            }

            if (max_erosion.at<uint8_t>(row, col) == 255) {
                cv::Point pt(col, row);
                right_points.push_back(pt);
                break;

            }

        }

    }

    for (int row = 2; row < img.rows - 2; row++) {
        for (int col = 0; col < img.cols; col++) {
            if (max_erosion.at<uint8_t>(row, 0) == 255) {
                break;
            }
            if (max_erosion.at<uint8_t>(row, col) == 255) {
                cv::Point pt(col, row);
                left_points.push_back(pt);
                break;

            }
        }

    }


    if (!right_points.empty() && right_points.size() > 7) {
        cv::Vec4f line11;
        cv::fitLine(right_points, line11, cv::DIST_L2, 0, 0.01, 0.01);

        int x11 = line11[2];
        int y11 = line11[3];
        cv::Point2f p1_top;
        cv::Point2f p1_bottom;
        cv::Point2f xy(line11[0], line11[1]);


        if (line11[1] == 0) {

            p1_bottom.x = 0;
            p1_bottom.y = y11;

            p1_top.x = img.cols - 1;
            p1_top.y = y11;

        }

        if (line11[1] != 0) {
            p1_top.x = x11 + 1.0 * (0 - y11) * line11[0] / line11[1];
            p1_top.y = 0;

            p1_bottom.x = x11 + 1.0 * (img.rows - 1 - y11) * line11[0] / line11[1];
            p1_bottom.y = img.rows - 1;

        }

        std::vector<cv::Point2f>pts = { p1_top, p1_bottom ,xy };

        extractout.pts = pts;

    }


    if (!left_points.empty()) {

        cv::Vec4f line22;

        cv::fitLine(left_points, line22, cv::DIST_L2, 0, 0.01, 0.01);

        float k22 = line22[1] / line22[0];
        int x22 = line22[2];
        int y22 = line22[3];
    }


    /* for (auto i : left_points) {
          cv::circle(img, i, 2, cv::Scalar(255, 0, 0));
      }
      for (auto i : right_points) {
          cv::circle(img, i, 2, cv::Scalar(0, 0, 255));
      }
    */

    return extractout;

}


cv::Mat  FindLane::get_img_mask(cv::Mat& img_1, cv::Mat& img_2) {



    cv::resize(img_1, img_1, cv::Size(), sca_1, sca_1, cv::INTER_LINEAR);
    cv::resize(img_2, img_2, cv::Size(), sca_2, sca_2, cv::INTER_LINEAR);
    cv::resize(img_2, img_2, img_1.size());



    cv::Mat ans = cv::Mat(img_1.rows + img_2.rows, img_1.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    ExtractOut extract_1 = extract_color_lanes(img_1);
    ExtractOut extract_2 = extract_color_lanes(img_2);

    auto pianyi = static_cast<int>(20 / mm_pre_x);

    bool flag = false;
    bool rotated_flag = false;
    int img_1_col = 0;
    int img_1_top = 0;
    int img_2_col = 0;
    int img_2_bottom = 0;

    if (extract_1.pts.size() != 0 && extract_2.pts.size() != 0) {

        auto p1_top = extract_1.pts[0];
        auto p1_bottom = extract_1.pts[1];
        auto p1_xy = extract_1.pts[2];

        auto p2_top = extract_2.pts[0];
        auto p2_bottom = extract_2.pts[1];
        auto p2_xy = extract_2.pts[2];


        float angle_1 = atan2(-p1_xy.y, p1_xy.x) * 180 / CV_PI;
        float angle_2 = atan2(-p2_xy.y, p2_xy.x) * 180 / CV_PI;


        if (-180 < angle_1 && angle_1 < 0) {
            angle_1 += 180;
        }
        if (-180 < angle_2 && angle_2 < 0) {
            angle_2 += 180;
        }



        double diff_angle = angle_1 - angle_2;

        img_2_col = p2_top.x;
        img_2_bottom = p2_bottom.x;


        int X = 0;
        auto p1_1 = extract_1.pts[0];
        auto p2_1 = extract_1.pts[1];
        auto p3_1 = extract_1.pts[2];
        float xx = p3_1.x;
        float yy = p3_1.y;
        auto p1 = extract_2.pts[0];
        auto p2 = extract_2.pts[1];
        if (p3_1.y == 0) {


        }

        if (p3_1.y != 0) {
            X = 1.0 * xx / yy * (p2.y - p1.y) + p2_1.x;

        }


        auto max_value = std::max({ img_2_bottom,img_2_col,X });

        cv::Point rotated_point = cv::Point(img_2_col, 0);
        cv::Mat M1 = cv::getRotationMatrix2D(rotated_point, diff_angle, 1);

        cv::Mat img_2_clone;
        cv::warpAffine(img_2, img_2_clone, M1, cv::Size(img_2.cols, img_2.rows));

        int y_dis = p2_top.x - p1_bottom.x;



        int len = abs(X - p2_bottom.x) + 2;

        //std::cout << "y_dis = " << y_dis << std::endl;
        //std::cout << "len = "<<len << std::endl;
        //std::cout << "diff_angle = " << diff_angle << std::endl;



        if ((0 < p1_bottom.x && p1_bottom.x < img_1.rows) && (p1_bottom.y == img_1.rows - 1) && (0 < p2_top.x && p2_top.x < img_2.rows) && (p2_top.y == 0)) {

            flag = true;

        }

        if (flag == true) {

 /*           std::cout << "y_dis =" << y_dis << std::endl;*/
            if (y_dis > 0) {
                if (y_dis < max_value && max_value < img_2.cols) {
                    cv::Mat roi = img_2_clone.clone()(cv::Rect(y_dis, 0, max_value - y_dis, img_2.rows));

                    /*roi.copyTo(img_2(cv::Rect(0, 0, max_value - y_dis, img_2.rows)));*/

                    if (max_value + len < img_2.cols) {

                        cv::Mat roi11 = img_2.clone()(cv::Rect(0, 0, y_dis + len, img_2.rows));

                        //printf("img_2.rows = %d, img_2.cols = %d, max_value = %d , y_dis = %d, len = %d\n",
                        //    img_2.rows, img_2.cols, max_value, y_dis, len);

                        roi11.copyTo(img_2(cv::Rect(max_value - y_dis, 0, y_dis + len, img_2.rows)));

                        for (int row = 0; row < roi.rows; row++) {
                            for (int col = 0; col < roi.cols; col++) {
                                if (roi.at<cv::Vec3b>(row, col) != cv::Vec3b(0, 0, 0)) {
                                    img_2.at<cv::Vec3b>(row, col) = roi.at<cv::Vec3b>(row, col);

                                }

                            }

                        }
                        rotated_flag = true;

                    }

                }
            }

            if (y_dis <= 0) {

                int temp_y_dis = abs(y_dis);
                if (temp_y_dis < img_2.cols && 0 < max_value && max_value < img_2.cols) {

                    cv::Mat roi = img_2_clone.clone()(cv::Rect(0, 0, max_value, img_2.rows));

                    for (int row = 0; row < roi.rows; row++) {
                        for (int col = 0; col < roi.cols; col++) {
                            if (roi.at<cv::Vec3b>(row, col) != cv::Vec3b(0, 0, 0)) {
                                if (col + temp_y_dis > img_2.cols) {
                                    continue;
                                }
                                img_2.at<cv::Vec3b>(row, col + temp_y_dis) = roi.at<cv::Vec3b>(row, col);
                            }

                        }
                    }
                    rotated_flag = true;

                }
            }



        }




    }

    cv::line(img_1, cv::Point(stand_bar_x, 0), cv::Point(stand_bar_x, img_1.rows - 1), cv::Scalar(255, 0, 0), 3);
    cv::circle(img_1, stand_bar_p, 2, cv::Scalar(215, 45, 25), 8);

    cv::line(img_2, cv::Point(stand_bar_x, 0), cv::Point(stand_bar_x, img_2.rows - 1), cv::Scalar(255, 0, 0), 3);
    cv::circle(img_2, stand_bar_p, 2, cv::Scalar(215, 45, 25), 8);

    cv::Point p1_xy;

    if (extract_1.pts.size() != 0) {

        auto p1 = extract_1.pts[0];
        auto p2 = extract_1.pts[1];
        auto pxy = extract_1.pts[2];
        p1_xy.x = pxy.x;
        p1_xy.y = pxy.y;


        cv::Point p3;
        if (p1.y != p2.y) {
            p3.x = 1.0 * (p2.x - p1.x) / (p2.y - p1.y) * (stand_bar_y - p1.y) + p1.x;
            p3.y = stand_bar_y;

        }
        if (p1.y == p2.y) {
            p3.x = p1.x;
            p3.y = stand_bar_y;

        }
        double length = (stand_bar_x - p3.x) * mm_pre_x / 10;


        cv::line(img_1, cv::Point(p1.x - pianyi, p1.y), cv::Point(p2.x - pianyi, p2.y), cv::Scalar(0, 255, 0), 3);

        cv::line(img_1, cv::Point(p1.x + pianyi, p1.y), cv::Point(p2.x + pianyi, p2.y), cv::Scalar(0, 255, 0), 3);

        cv::circle(img_1, p3, 2, cv::Scalar(215, 45, 25), 8);
        cv::line(img_1, p3, stand_bar_p, cv::Scalar(0, 0, 255), 3);

        cv::putText(img_1, cv::format("%.2f cm", length), cv::Point(500, stand_bar_y), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2);

    }

    if (rotated_flag == false && extract_2.pts.size() != 0) {
        auto p1 = extract_2.pts[0];
        auto p2 = extract_2.pts[1];

        cv::Point p3;
        if (p1.y != p2.y) {
            p3.x = 1.0 * (p2.x - p1.x) / (p2.y - p1.y) * (stand_bar_y - p1.y) + p1.x;
            p3.y = stand_bar_y;

        }
        if (p1.y == p2.y) {
            p3.x = p1.x;
            p3.y = stand_bar_y;

        }
        double length = (stand_bar_x - p3.x) * mm_pre_x / 10;


        cv::line(img_2, cv::Point(p1.x - pianyi, p1.y), cv::Point(p2.x - pianyi, p2.y), cv::Scalar(0, 255, 0), 3);
        cv::line(img_2, cv::Point(p1.x + pianyi, p1.y), cv::Point(p2.x + pianyi, p2.y), cv::Scalar(0, 255, 0), 3);

        cv::circle(img_2, p3, 2, cv::Scalar(215, 45, 25), 8);
        cv::line(img_2, p3, stand_bar_p, cv::Scalar(0, 0, 255), 3);

        cv::putText(img_2, cv::format("%.2f cm", length), cv::Point(500, stand_bar_y), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2);
    }



    if (rotated_flag == true && extract_2.pts.size() != 0) {

        int X = 0;
        auto p1_1 = extract_1.pts[0];
        auto p2_1 = extract_1.pts[1];
        auto p3_1 = extract_1.pts[2];
        float xx = p3_1.x;
        float yy = p3_1.y;


        auto p1 = extract_2.pts[0];
        auto p2 = extract_2.pts[1];

        if (p3_1.y == 0) {


        }

        if (p3_1.y != 0) {

            X = 1.0 * xx / yy * (p2.y - p1.y) + p2_1.x;

        }

        cv::Point p3;
        if (p1.y != p2.y) {
            p3.x = 1.0 * (p2.x - p1.x) / (p2.y - p1.y) * (stand_bar_y - p1.y) + p1.x;
            p3.y = stand_bar_y;

        }
        if (p1.y == p2.y) {
            p3.x = p1.x;
            p3.y = stand_bar_y;

        }


        cv::Point p4;
        if (p1_1.y != p2_1.y) {
            p4.x = 1.0 * xx / yy * (stand_bar_y - p1.y) + p2_1.x;
            p4.y = stand_bar_y;

        }
        if (p1_1.y == p2_1.y) {
            p4.x = p1.x;
            p4.y = stand_bar_y;

        }


        double length = (stand_bar_x - p3.x) * mm_pre_x / 10;

        cv::line(img_2, cv::Point(p2_1.x - pianyi, p1.y), cv::Point(X - pianyi, p2.y), cv::Scalar(0, 255, 0), 3);
        cv::line(img_2, cv::Point(p2_1.x + pianyi, p1.y), cv::Point(X + pianyi, p2.y), cv::Scalar(0, 255, 0), 3);

        cv::circle(img_2, p4, 2, cv::Scalar(215, 45, 25), 8);
        cv::line(img_2, p4, stand_bar_p, cv::Scalar(0, 0, 255), 3);

        cv::putText(img_2, cv::format("%.2f cm", length), cv::Point(500, stand_bar_y), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2);
    }


    img_1.copyTo(ans(cv::Rect(0, 0, img_1.cols, img_1.rows)));
    img_2.copyTo(ans(cv::Rect(0, img_1.rows, img_2.cols, img_2.rows)));


    return ans;


};


std::map<std::string, std::string> FindLane::readConfig(){

    std::ifstream configFile(config_path); 
    std::map<std::string, std::string> config_mp;
    if (!configFile.is_open()) {
        std::cerr << "Error opening the config file." << std::endl;
        return config_mp;
    }

    
    std::string line;
    while (std::getline(configFile, line)) {
        if (line.empty() || line[0] == '#' || line.find('=') == std::string::npos) {
 
            continue;
        }

        size_t delimiterPos = line.find('=');
        std::string key = line.substr(0, delimiterPos);
        std::string value = line.substr(delimiterPos + 1);
        

        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t\r\n") + 1);

        config_mp[key] = value;
    }

    for (const auto& entry : config_mp) {
        std::cout << entry.first << "=" << entry.second << std::endl;
    }

    configFile.close(); // 关闭文件
    stand_bar_x = std::stoi(config_mp["x"]);
    stand_bar_y = std::stoi(config_mp["y"]);
    stand_bar_p.x = stand_bar_x;
    stand_bar_p.y = stand_bar_y;


    return config_mp;

}