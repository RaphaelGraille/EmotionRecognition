#include "opencv2/opencv.hpp"

using namespace cv;


// Function used to train the SVM
void trainSVM(Mat descriptors, Mat labels,bool save, bool autoTrain, string svmFileName){

    // Set the SVM parameters manually (C, gamma ...)
    if (!autoTrain) {
        
        CvSVMParams params;
        
        params.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
        params.degree = 0; // for poly
        params.gamma = 1e-05; // for poly/rbf/sigmoid
        params.coef0 = 0;
        params.C = 2.5;
        params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,100000,0.000001);
        
        CvSVM svm;
        svm.train(descriptors, labels, Mat(), Mat(), params);
        
        if (save) {
            svm.save(svmFileName.c_str());
        }
        
    // Use openCV function "train_auto" to use get the best SVM parameter with Cross Validation
    } else {
        
        CvSVM svm;
    
        CvSVMParams paramz;
        paramz.kernel_type = CvSVM::RBF;
        paramz.svm_type = CvSVM::C_SVC;
        paramz.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,100000,0.000001);
        svm.train_auto(descriptors, labels, Mat(), Mat(), paramz,10, CvSVM::get_default_grid(CvSVM::C), CvSVM::get_default_grid(CvSVM::GAMMA), CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
        paramz = svm.get_params();
        std::cout<<"gamma:"<<paramz.gamma<<std::endl;
        std::cout<<"C:"<<paramz.C<<std::endl;
        
        if (save) {
            svm.save(svmFileName.c_str());
        }
        
    }
    
}