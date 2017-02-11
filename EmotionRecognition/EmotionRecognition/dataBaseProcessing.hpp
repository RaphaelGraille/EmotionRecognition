#ifndef DBProcess_H
#define DBProcess_H

#include <opencv2/opencv.hpp>

std::vector<cv::Mat> createDBAndLabels();
std::vector<cv::Mat> createDBAndLabelsFaceRec();
std::string fixedLength(int value, int digits = 3);

#endif