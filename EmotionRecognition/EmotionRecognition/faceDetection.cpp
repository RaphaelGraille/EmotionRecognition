#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "faceDetection.hpp"
#include "libflandmark/flandmark_detector.h"

using namespace std;
using namespace cv;


Rect detectFace( Mat frame )
{
    
    // The crop rectangle for the face
    Rect crop;
    
    
    // Initialisation of the Opencv Face detector
    String face_cascade_name = "/Users/Raphael/opencv-2.4.11/data/haarcascades/haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = "/Users/Raphael/opencv-2.4.11/data/haarcascades/haarcascade_mcs_eyepair_big.xml";
    CascadeClassifier face_cascade;
    //CascadeClassifier eyes_cascade;
    //string window_name = "Capture - Face detection";
    RNG rng(1234);
    
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n");};
    //if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n");};
    
    // Vector of recangles containing the detected faces
    std::vector<Rect> faces;
    
    // the grayscale frame
    Mat frame_gray = frame;
    
    // Histogram equalisation (for face detection only)
    equalizeHist( frame_gray, frame_gray );
    
    // Face detection
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    // If no face has been detected
    if (faces.size() == 0) {
        
        crop = Rect(0,0,10,10);
        
    } else {
        
        int shift = cvRound(faces[0].width/10);
        
        // We take the first face detected and shrink the size of the recangle to fit the face better.
        crop =  Rect(faces[0].x + shift,faces[0].y,faces[0].width - 2*shift, faces[0].height);
        
    }
    
    // imshow( window_name, croppedFace );
    
    return crop;
}


// this function was used to test FLANDMARK a face landmark detector in order to get a precise location of the eyes, mouth etc ... This was not used in the final project
Rect detectROIS(Mat frame, FLANDMARK_Model * model){
    
    Rect crop;
    
    String face_cascade_name = "/Users/Raphael/opencv-2.4.11/data/haarcascades/haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = "/Users/Raphael/opencv-2.4.11/data/haarcascades/haarcascade_mcs_eyepair_big.xml";
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    string window_name = "Capture - Face detection";
    RNG rng(1234);
    
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n");};
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n");};
    
    std::vector<Rect> faces;
    Mat frame_gray = frame;
    
    //cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    
    std::vector<Rect> croppedFaces;
    Mat croppedFace;
    
    for( size_t i = 0; i < faces.size(); i++ )
    {
        
        int bbox[] = {faces[0].x -30, faces[0].y-30, faces[0].x + faces[0].width +30, faces[0].y + faces[0].height+30};

        IplImage* iplFrame = cvCreateImage(cvSize(frame.cols, frame.rows), IPL_DEPTH_8U, 1);
        *iplFrame = frame;
        
        double * landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
        flandmark_detect(iplFrame, bbox, model, landmarks);
        
        CvPoint leftEye1 = cvPoint((int)landmarks[10],(int)landmarks[11]);
        CvPoint leftEye2 = cvPoint((int)landmarks[2],(int)landmarks[3]);
        CvPoint rightEye1 = cvPoint((int)landmarks[4],(int)landmarks[5]);
        CvPoint rightEye2 = cvPoint((int)landmarks[12],(int)landmarks[13]);
        
        CvPoint eye1 = cvPoint(int((leftEye1.x + leftEye2.x)/2), int((leftEye1.y + leftEye2.y)/2)); // The eye seen on the left of the picture (right eye of the person)
        CvPoint eye2 = cvPoint(int((rightEye1.x + rightEye2.x)/2), int((rightEye1.y + rightEye2.y)/2)); // The eye seen on the right of the picture (left eye of the person)
        
        int Dist = eye2.x - eye1.x;
        int width = cvRound(2*Dist);
        int height = cvRound(15*width/11);
        
        crop = Rect(max(eye1.x - cvRound(Dist/2),0),max(eye1.y - cvRound(5*height/14),0), width, height);
        
        croppedFaces.push_back(crop);
        
    }
    
    if (faces.size() == 0) {
        crop = Rect(0,0,10,10);
    }
    
    return crop;
    
}
