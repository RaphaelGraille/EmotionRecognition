

#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "lbp.hpp"
#include "faceDetection.hpp"
#include "dataBaseProcessing.hpp"
#include "trainSVM.hpp"
#include "var.hpp"
#include "testSVM.hpp"
#include "libflandmark/flandmark_detector.h"

std::map<float,std::string> labelMap{{1, "Angry"},{2,"Neutral"},{3,"Disgust"},{4,"Fear"},{5,"Happy"},{6,"Sadness"},{7,"surpise"}};


// Function used to test the detector on one single image
void testOneFrame(std::string imagePath){

    
    FileStorage fs(var::projectDir + "DescriptorsAndLabels_7emotions.yml", FileStorage::READ);
    
    Mat descriptors;
    Mat labels;
    
    fs["descriptors"] >> descriptors;
    fs["labels"] >> labels;
    
    
    fs.release();
    
    CvSVM m_svm;
    //m_svm.load("/Users/Raphael/Documents/Poly/Maitrise_S2/Applications_multimedias/Projet/svm_save");
    
    CvSVMParams params;
    
    params.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
    params.degree = 0; // for poly
    params.gamma = 1e-06; // for poly/rbf/sigmoid
    params.coef0 = 0;
    params.C = 2.5;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_EPS,1000000,0.000001);
    
    m_svm.train(descriptors, labels, Mat(), Mat(), params);
     
    
    Mat img = imread(imagePath);
    cvtColor( img, img, CV_BGR2GRAY );

    
    Mat lbp;
    
    Rect crop = detectFace( img );
    Mat cropImg = img(crop);
    resize(cropImg, cropImg, Size(110,150));
    
    imshow("face 1", cropImg);
    lbp::ELBP(cropImg,lbp,var::radius, var::neighbors);
    imshow("lbp", lbp);
    Mat lbpConcHist = lbp::lbpConcHist(lbp, var::n, var::m, var::neighbors);
    
    float prediction = m_svm.predict(lbpConcHist);
    
    std::cout << "Predicted Emotion : " << labelMap[prediction] << std::endl;
    
    cv::waitKey(0);
    
}


// function to test the detector on the webcam
int webcamTest(){

    FileStorage fs(var::projectDir + "DescriptorsAndLabels_5emotions.yml", FileStorage::READ);
    
    Mat descriptors;
    Mat labels;
    
    fs["descriptors"] >> descriptors;
    fs["labels"] >> labels;
    
    
    fs.release();
    
    CvSVM m_svm;
    //m_svm.load("/Users/Raphael/Documents/Poly/Maitrise_S2/Applications_multimedias/Projet/svm_save");
    
    CvSVMParams params;
    
    params.kernel_type = CvSVM::LINEAR; //CvSVM::RBF, CvSVM::LINEAR ...
    params.degree = 0; // for poly
    params.gamma = 1e-06; // for poly/rbf/sigmoid
    params.coef0 = 0;
    params.C = 3;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_EPS,1000000,0.000001);
    
    m_svm.train(descriptors, labels, Mat(), Mat(), params);
    
    VideoCapture cap;
    
    if(!cap.open(0))
        return 0;
    for(;;)
    {
        Mat frame;
        Mat gray_frame;
        cap >> frame;
        if( frame.empty() ) break;
        
        Mat lbp;
        
        cvtColor( frame, gray_frame, CV_BGR2GRAY );
        
        Rect cropImg = detectFace( gray_frame );
        
        Mat croppedFace = gray_frame(cropImg);
        resize(croppedFace, croppedFace, Size(110,150));
        
        rectangle(frame, cvPoint(cropImg.x,cropImg.y), cvPoint(cropImg.x + cropImg.width,cropImg.y+cropImg.height) , Scalar(0,0,200));
        
        lbp::ELBP(croppedFace,lbp,var::radius, var::neighbors);
        Mat lbpConcHist = lbp::lbpConcHist(lbp, var::n, var::m, var::neighbors);
        
        float prediction = m_svm.predict(lbpConcHist);
        String emotion = labelMap[prediction];
        putText(frame, emotion, cvPoint(cropImg.x-20,cropImg.y-20),
                FONT_HERSHEY_COMPLEX, 2, cvScalar(200,0,0), 3, CV_AA);
        
        imshow("Emotion Recognition", frame);
        if( waitKey(1) == 27 ) break; // stop capturing by pressing ESC
    }
    return 0;
}

using namespace cv;
int main(int argc, char** argv)
{
    // To use the main, uncomment one of the lines below.
    
    // The first one create a file containing the descriptors and labels with the last configuration of emotions
    // (7 emotions)
    
    // The second one laucnh the test with teh webcam
    
    // The third one makes a test on a single Frame

    // The last one proceed to cross validation
    
    
 
    // 1st Test
    //std::vector<Mat> DBAndLabels = createDBAndLabels();
    
    // 2nd test
    //webcamTest();
    
    // 3rd test
    //testOneFrame(var::imagesDir + "S010/001/S010_001_00000001.png");
    
    // 4th test
    crossValidation(10,"DescriptorsAndLabels_7emotions.yml", var::projectDir + "kfold-CV-Emostions_TEST.txt");

        
    return 0;
}
