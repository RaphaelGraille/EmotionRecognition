#ifndef TRAINSVM_H
#define TRAINSVM_H

#include "opencv2/opencv.hpp"

void trainSVM(Mat descriptors, Mat labels,bool save, bool autoTrain, string svmFileName);

#endif