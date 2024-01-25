#include "segmentation.h"

static float in_arr[1 * 3 * 640 * 640];
static float out0_arr[1 * 8400 * 37];
static float out1_arr[1 * 32 * 160 * 160];

static float out0_arr_det[1 * 8400 * 84];

void Segmentation::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    while (i <= j) {
        while (faceobjects[i].prob > p) {
            i++;
        }
        while (faceobjects[j].prob < p) {
            j--;
        }
        if (i <= j) {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }

    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void  Segmentation::qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty()) {
        return;
    }
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

float Segmentation::intersection_area(Object& a, Object& b) {
    cv::Rect2f inter = a.rect & b.rect;
    return inter.area();

}


void Segmentation::nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}



void Segmentation::loadEngine(const std::string& path) {
    size_t size{ 0 };
    char* trtModelStream{ nullptr };
    std::ifstream file(path, std::ios::binary);

    if (file.good()) {
        file.seekg(0, std::ios::end);
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }

    runtime_ = createInferRuntime(logger_);
    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size);
    context_ = engine_->createExecutionContext();

    delete[] trtModelStream;
}


void Segmentation::loadEngine_det(const std::string& path) {
    size_t size{ 0 };
    char* trtModelStream{ nullptr };
    std::ifstream file(path, std::ios::binary);

    if (file.good()) {
        file.seekg(0, std::ios::end);
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }

    runtime_det = createInferRuntime(logger_);
    engine_det = runtime_det->deserializeCudaEngine(trtModelStream, size);
    context_det = engine_det->createExecutionContext();

    delete[] trtModelStream;
}




cv::Mat Segmentation::preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    padw = (input_w - w) / 2;
    padh = (input_h - h) / 2;
    return out;
}


cv::Mat Segmentation::doInference_det(cv::Mat& org_img) {

  
    int32_t input_index = engine_det->getBindingIndex(images_);
    int32_t output0_index = engine_det->getBindingIndex(output0_);

    void* buffers[2];
    cudaMalloc(&buffers[input_index], BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float));
    cudaMalloc(&buffers[output0_index], BATCH_SIZE * OUTPUT0_BOXES * OUTPUT0_ELEMENT_DET * sizeof(float));

    int padw = 0;
    int padh = 0;
    cv::Mat pr_img = preprocessImg(org_img, INPUT_W, INPUT_H, padw, padh);

    for (int i = 0; i < INPUT_W * INPUT_H; i++) {
        in_arr[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
        in_arr[i + INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
        in_arr[i + 2 * INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[input_index], in_arr, BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_det->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(out0_arr_det, buffers[output0_index], BATCH_SIZE * OUTPUT0_BOXES * OUTPUT0_ELEMENT_DET * sizeof(float), cudaMemcpyDeviceToHost, stream);


    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output0_index]);


    float r_w = INPUT_W / (org_img.cols * 1.0);
    float r_h = INPUT_H / (org_img.rows * 1.0);

    std::vector<Object> objects;
    int net_width = OUTPUT0_ELEMENT_DET;
    float* pdata = out0_arr_det;
    for (int i = 0; i < OUTPUT0_BOXES; ++i) {

        float* score_ptr = std::max_element(pdata + 4, pdata + 4 + CLASSES_DET);
        float box_score = *score_ptr;
        int label_index = score_ptr - (pdata + 4);
        if (label_index == 0 || label_index == 1 || label_index == 2 || label_index == 3 || label_index == 5 || label_index == 7) {
        
        
            if (box_score >= CONF_THRESHOLD) {
                int l, r, t, b;
                float x = pdata[0];
                float y = pdata[1];
                float w = pdata[2];
                float h = pdata[3];

                if (r_h > r_w) {
                    l = x - w / 2.0;
                    r = x + w / 2.0;
                    t = y - h / 2.0 - (INPUT_H - r_w * org_img.rows) / 2;
                    b = y + h / 2.0 - (INPUT_H - r_w * org_img.rows) / 2;
                    l = l / r_w;
                    r = r / r_w;
                    t = t / r_w;
                    b = b / r_w;
                }
                else {
                    l = x - w / 2.0 - (INPUT_W - r_h * org_img.cols) / 2;
                    r = x + w / 2.0 - (INPUT_W - r_h * org_img.cols) / 2;
                    t = y - h / 2.0;
                    b = y + h / 2.0;
                    l = l / r_h;
                    r = r / r_h;
                    t = t / r_h;
                    b = b / r_h;
                }

                Object obj;
                obj.rect.x = std::max(l, 0);
                obj.rect.y = std::max(t, 0);
                obj.rect.width = r - l;
                obj.rect.height = b - t;
                obj.label = label_index;
                obj.prob = box_score;
                objects.push_back(obj);
            }
        
        }
      
        pdata += net_width; // 下一行
    }

    qsort_descent_inplace(objects);
    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, 0.7);
    int count = picked.size();

    std::vector<Object>obj_out(count);
    for (int i = 0; i < count; ++i) {
        obj_out[i] = objects[picked[i]];
    }

    for (int i = 0; i < count; i++) {
        auto obj = obj_out[i];

 /*       fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);*/

        cv::rectangle(org_img, obj.rect, cv::Scalar(255, 0, 0));

      /*  char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > org_img.cols)
            x = org_img.cols - label_size.width;

        cv::rectangle(org_img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255, 255, 255), -1);

        cv::putText(org_img, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));*/

    }



    return org_img;
}

cv::Mat Segmentation::doInference(cv::Mat& org_img) {

    int32_t input_index = engine_->getBindingIndex(images_);
    int32_t output0_index = engine_->getBindingIndex(output0_);
    int32_t output1_index = engine_->getBindingIndex(output1_);

    void* buffers[3];
    cudaMalloc(&buffers[input_index], BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float));
    cudaMalloc(&buffers[output0_index], BATCH_SIZE * OUTPUT0_BOXES * OUTPUT0_ELEMENT * sizeof(float));
    cudaMalloc(&buffers[output1_index], BATCH_SIZE * OUTPUT1_CHANNELS * OUTPUT1_H * OUTPUT1_W * sizeof(float));

    int padw = 0;
    int padh = 0;
    cv::Mat pr_img = preprocessImg(org_img, INPUT_W, INPUT_H, padw, padh);

    for (int i = 0; i < INPUT_W * INPUT_H; i++) {
        in_arr[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
        in_arr[i + INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
        in_arr[i + 2 * INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[input_index], in_arr, BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(out0_arr, buffers[output0_index], BATCH_SIZE * OUTPUT0_BOXES * OUTPUT0_ELEMENT * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(out1_arr, buffers[output1_index], BATCH_SIZE * OUTPUT1_CHANNELS * OUTPUT1_H * OUTPUT1_W * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output0_index]);
    cudaFree(buffers[output1_index]);


    float r_w = INPUT_W / (org_img.cols * 1.0);
    float r_h = INPUT_H / (org_img.rows * 1.0);

    std::vector<Object> objects;
    int net_width = OUTPUT0_ELEMENT;
    float* pdata = out0_arr;
    for (int i = 0; i < OUTPUT0_BOXES; ++i) {

        float* score_ptr = std::max_element(pdata + 4, pdata + 4 + CLASSES);
        float box_score = *score_ptr;
        int label_index = score_ptr - (pdata + 4);

        if (box_score >= CONF_THRESHOLD) {
            int l, r, t, b;
            float x = pdata[0];
            float y = pdata[1];
            float w = pdata[2];
            float h = pdata[3];
            std::vector<float> maskdata(pdata + 4 + CLASSES, pdata + net_width);

            if (r_h > r_w) {
                l = x - w / 2.0;
                r = x + w / 2.0;
                t = y - h / 2.0 - (INPUT_H - r_w * org_img.rows) / 2;
                b = y + h / 2.0 - (INPUT_H - r_w * org_img.rows) / 2;
                l = l / r_w;
                r = r / r_w;
                t = t / r_w;
                b = b / r_w;
            }
            else {
                l = x - w / 2.0 - (INPUT_W - r_h * org_img.cols) / 2;
                r = x + w / 2.0 - (INPUT_W - r_h * org_img.cols) / 2;
                t = y - h / 2.0;
                b = y + h / 2.0;
                l = l / r_h;
                r = r / r_h;
                t = t / r_h;
                b = b / r_h;
            }

            Object obj;
            obj.rect.x = std::max(l, 0);
            obj.rect.y = std::max(t, 0);
            obj.rect.width = r - l;
            obj.rect.height = b - t;
            obj.label = label_index;
            obj.prob = box_score;
            obj.maskdata = maskdata;
            objects.push_back(obj);
        }


        pdata += net_width; // 下一行
    }

    qsort_descent_inplace(objects);
    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, 0.7);
    int count = picked.size();

    std::vector<Object>obj_out(count);
    for (int i = 0; i < count; ++i) {
        obj_out[i] = objects[picked[i]];
    }

    for (int i = 0; i < count; i++) {
        Object& obj = obj_out[i];
        cv::Mat mask(OUTPUT1_W, OUTPUT1_H, CV_32FC1);
        mask = cv::Scalar(0.f);
        for (int p = 0; p < OUTPUT1_CHANNELS; p++) {
            std::vector<float>temp(out1_arr + OUTPUT1_W * OUTPUT1_H * p, out1_arr + OUTPUT1_W * OUTPUT1_H * (p + 1));
            float coeff = obj.maskdata[p];
            float* mp = (float*)mask.data;
            for (int j = 0; j < OUTPUT1_W * OUTPUT1_H; j++) {
                mp[j] += temp.data()[j] * coeff;
            }
        }

        //原始图到特征图的缩放比例,padding也要进行缩放
        float ratio_w = 1.0 * INPUT_W / OUTPUT1_W;
        float ratio_h = 1.0 * INPUT_H / OUTPUT1_H;
        cv::Rect roi(int(padw / ratio_w), int(padh / ratio_h), int((INPUT_W - padw * 2) / ratio_w), int((INPUT_H - padh * 2) / ratio_h));
        cv::Mat dest;
        cv::exp(-mask, dest);
        dest = 1. / (1. + dest);
        dest = dest(roi);
        cv::Mat mask2;
        cv::resize(dest, mask2, org_img.size());
        obj.mask = cv::Mat(org_img.rows, org_img.cols, CV_8UC1);
        obj.mask = cv::Scalar(0);
        for (int y = 0; y < org_img.rows; y++) {
            if (y<obj.rect.y || y>obj.rect.y + obj.rect.height) {
                continue;
            }
            float* mp2 = mask2.ptr<float>(y);
            uchar* bmp = obj.mask.ptr<uchar>(y);
            for (int x = 0; x < org_img.cols; x++) {
                if (x < obj.rect.x || x>obj.rect.x + obj.rect.width) {
                    continue;
                }
                bmp[x] = mp2[x] > 0.5f ? 255 : 0;

            }
        }
    }

    cv::Mat result = cv::Mat::zeros(org_img.size(), CV_8UC1);
    for (auto i : obj_out) {
        result |= i.mask;
    }
    return result;
}