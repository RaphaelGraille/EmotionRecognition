#ifndef ED_HPP_
#define ED_HPP_

#include <opencv2/opencv.hpp>
#include "libflandmark/flandmark_detector.h"

cv::Rect detectFace( cv::Mat frame );
cv::Rect detectROIS(cv::Mat frame, FLANDMARK_Model * model);

#endif