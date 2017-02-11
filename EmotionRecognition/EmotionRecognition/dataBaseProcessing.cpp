#include <iostream>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "var.hpp"
#include "lbp.hpp"
#include "faceDetection.hpp"
#include "dataBaseProcessing.hpp"
#include "libflandmark/flandmark_detector.h"


using namespace cv;

// this function is used to parse the database and create the descriptors and corresponding labels, and save them in a file.
std::vector<Mat> createDBAndLabels() {
    
    int numberLabel =0;
    int indexImage = 0;
    
    std::ofstream outputFile("/Users/Raphael/Documents/Poly/Maitrise_S2/Applications_multimedias/Projet/table-index-mages.txt");
    
    Mat descriptors;
    Mat labels;
    
    DIR *dirPersons;
    DIR *dirEmotions;
    DIR *dirLabel;
    struct dirent *entPersons;
    
    
    if ((dirPersons = opendir (var::labelsDir.c_str())) != NULL) {
        
        // Iterate through all the persons of the database
        while ((entPersons = readdir (dirPersons)) != NULL) {
            printf ("%s\n", entPersons->d_name);
            
            if (std::strcmp(entPersons->d_name, ".") != 0 && std::strcmp(entPersons->d_name, "..") !=0 ) {
                struct dirent *entEmotions;
                std::string dirEmoName = var::labelsDir + entPersons->d_name;
                if ((dirEmotions = opendir(dirEmoName.c_str())) != NULL) {
                    
                    // Iterate through all emotions for a given person
                    while ((entEmotions = readdir(dirEmotions)) != NULL){
                        
                        if (std::strcmp(entEmotions->d_name, ".") != 0 && std::strcmp(entEmotions->d_name, "..") !=0 ){
                        
                            std::cout << "   | "<< entEmotions->d_name << std::endl;
                            
                            std::string dirLabName = dirEmoName +"/"+ entEmotions->d_name;
                        
                            struct dirent *entLabel;
                            if ((dirLabel = opendir(dirLabName.c_str())) != NULL) {
                                
                                // Iterate through all the label for a given person (their should be only one)
                                while((entLabel = readdir(dirLabel)) != NULL){
                                    
                                    // Avoid the non desired directory/files : ".", ".." and ".DS_Store" (this is)
                                    // an hidden file created by Mac OSX to store informations about the directory
                                    if (std::strcmp(entLabel->d_name, ".") != 0 && std::strcmp(entLabel->d_name, "..")
                                        !=0 && std::strcmp(entLabel->d_name, ".DS_Store") != 0) {
                                        
                                        
                                        std::cout << "      |"<< entLabel->d_name<<std::endl;
                                        std::string labelFileName = dirLabName +"/"+ entLabel->d_name;
                                        std::fstream labelFile;
                                        
                                        // open the file contaoning the label
                                        labelFile.open(labelFileName);
                                        int label;
                                        labelFile >> label;
                                        std::cout << "          |"<<label<<std::endl;
                                        
                                        numberLabel ++;
                                        
                                        // We avoid label 2 wich is "contempt".
                                        if(label != 2){
                                            
                                            string numPictureChar = std::string(entLabel->d_name).substr(9,8);
                                            
                                            int numPicture = std::stoi(numPictureChar);
                                            
                                            // Take the first picture of the sequence for the neutral example
                                            
                                            std::string imagePathNeutral = var::imagesDir+ entPersons->d_name + +"/"+ entEmotions->d_name + "/" + std::string(entLabel->d_name).substr(0,9)+ fixedLength(1,8) + ".png";
                                            
                                            Mat lbpNeutral;
                                            // read the image
                                            Mat imgNeutral = imread(imagePathNeutral,CV_LOAD_IMAGE_GRAYSCALE);
                                            // Detect the face
                                            Rect cropNeutral = detectFace( imgNeutral );
                                            //Rect cropNeutral = detectROIS(imgNeutral, model);
                                            
                                            // crop the image on the face
                                            Mat cropImgNeutal = imgNeutral(cropNeutral);
                                            // resize it to 110*150
                                            resize(cropImgNeutal, cropImgNeutal, Size(110,150));
                                            // Compute descriptor
                                            lbp::ELBP(cropImgNeutal,lbpNeutral,var::radius, var::neighbors);
                                            Mat lbpConcHist = lbp::lbpConcHist(lbpNeutral, var::n, var::m, var::neighbors);
                                            
                                            // Add the label for neutral = 2
                                            labels.push_back(2);
                                            descriptors.push_back(lbpConcHist);
                                            
                                            // Make a Table linking the index of an image, to its path.
                                            // (This was used to know wich image was badly detected since when we
                                            // proceed to detection we only have acces to the index of the image)
                                            outputFile << indexImage << " : " << imagePathNeutral <<std::endl;
                                            indexImage ++;
                                            
                                            std::cout << "          |"<<imagePathNeutral<<"  Neutre"<<std::endl;
                                            
                                            // Then we take 3 pictures in the end of the sequence with one frame between each. For instance, if there are 20 pictures, take the 20th, the 18th and the 16th
                                            for (int ii = 0; ii<3; ii++) {
                                                
                                                std::string imagePath = var::imagesDir+ entPersons->d_name + +"/"+ entEmotions->d_name + "/" + std::string(entLabel->d_name).substr(0,9)+ fixedLength(numPicture-ii*2,8) + ".png";
                                                
                                                std::cout << "          |"<<imagePath<<std::endl;
                                                
                                                Mat lbp;
                                                
                                                // read the image
                                                Mat img = imread(imagePath,CV_LOAD_IMAGE_GRAYSCALE);
                                                // Detect the face
                                                Rect crop = detectFace( img );
                                                //Rect crop = detectROIS(img, model);
                                                
                                                // crop the image on the face
                                                Mat cropImg = img(crop);
                                                // resize it to 110*150
                                                resize(cropImg, cropImg, Size(110,150));
                                                // Compute descriptor
                                                lbp::ELBP(cropImg,lbp,var::radius, var::neighbors);
                                                Mat lbpConcHist = lbp::lbpConcHist(lbp, var::n, var::m, var::neighbors);
                                                
                                                // Add the label
                                                labels.push_back(label);
                                                // add the descriptor
                                                descriptors.push_back(lbpConcHist);
                                                
                                                // Make a Table linking the index of an image, to its path.
                                                // (This was used to know wich image was badly detected since when we
                                                // proceed to detection we only have acces to the index of the image)
                                                outputFile << indexImage << " : " << imagePath <<std::endl;
                                                indexImage ++;
                                                
                                                // This was used to visualize the result of the face detection on all
                                                // the databse
                                                //imwrite(var::projectDir + "EyeCroppedDataBase/" + std::string(entLabel->d_name).substr(0,9)+ fixedLength(numPicture-ii*2,8) + ".png", cropImg);
                                            }
                                        }
                                        
                                    }
                                    
                                }
                                closedir(dirLabel);
                            }
                            
                        }
                    }
                    closedir(dirEmotions);
                }
            }
            
        }
        closedir (dirPersons);
    } else {
        /* could not open directory */
        perror ("");
    }
    
    std::vector<cv::Mat> res;
    res.push_back(descriptors);
    res.push_back(labels);
    
    return res;
}

// This function does the same as the one above but for face recognition
std::vector<Mat> createDBAndLabelsFaceRec() {
    
    int numberLabel =0;
    
    Mat descriptors;
    Mat labels;
    
    DIR *dirPersons;
    DIR *dirEmotions;
    DIR *dirLabel;
    struct dirent *entPersons;
    
    if ((dirPersons = opendir (var::FacesDir.c_str())) != NULL) {
        
        /* print all the files and directories within directory */
        while ((entPersons = readdir (dirPersons)) != NULL) {
            printf ("%s\n", entPersons->d_name);
            
            if (std::strcmp(entPersons->d_name, ".") != 0 && std::strcmp(entPersons->d_name, "..") !=0 ) {
                struct dirent *entEmotions;
                std::string dirEmoName = var::FacesDir + entPersons->d_name;
                if ((dirEmotions = opendir(dirEmoName.c_str())) != NULL) {
                    while ((entEmotions = readdir(dirEmotions)) != NULL){
                        
                        if (std::strcmp(entEmotions->d_name, ".") != 0 && std::strcmp(entEmotions->d_name, "..") !=0 ){
                            
                            std::cout << "   | "<< entEmotions->d_name << std::endl;
                            
                            std::string dirLabName = dirEmoName +"/"+ entEmotions->d_name;
                            
                            struct dirent *entLabel;
                            if ((dirLabel = opendir(dirLabName.c_str())) != NULL) {
                                while((entLabel = readdir(dirLabel)) != NULL){
                                    if (std::strcmp(entLabel->d_name, ".") != 0 && std::strcmp(entLabel->d_name, "..") !=0 && std::strcmp(entLabel->d_name, ".DS_Store") != 0) {
                                        std::cout << "      |"<< entLabel->d_name<<std::endl;
                                        std::string labelFileName = dirLabName +"/"+ entLabel->d_name;
                                        
                                        int label;
                                        std::string labelName = std::string(entPersons->d_name).substr(1,3);
                                        label = std::stoi(labelName);
                                        
                                        std::cout << "          |"<<label<<std::endl;
                                        
                                        numberLabel ++;
                                        
                                        string numPictureChar = std::string(entLabel->d_name).substr(9,8);
                                        
                                        //int numPicture = std::stoi(numPictureChar);
                                        
                                        std::string imagePath = var::imagesDir+ entPersons->d_name + +"/"+ entEmotions->d_name + "/" + std::string(entLabel->d_name);
                                        
                                        std::cout << "          |"<<imagePath<<std::endl;
                                        
                                        Mat lbp;
                                        
                                        Mat img = imread(imagePath,CV_LOAD_IMAGE_GRAYSCALE);
                                        Rect crop = detectFace( img );
                                        Mat cropImg = img(crop);
                                        resize(cropImg, cropImg, Size(110,150));
                                        lbp::ELBP(cropImg,lbp,var::radius, var::neighbors);
                                        Mat lbpConcHist = lbp::lbpConcHist(lbp, var::n, var::m, var::neighbors);
                                        
                                        labels.push_back(label);
                                        descriptors.push_back(lbpConcHist);
                                        
                                    }
                                    
                                }
                                closedir(dirLabel);
                            }
                            
                        }
                    }
                    closedir(dirEmotions);
                }
            }
            
        }
        closedir (dirPersons);
    } else {
        /* could not open directory */
        perror ("");
    }
    
    std::vector<cv::Mat> res;
    res.push_back(descriptors);
    res.push_back(labels);
    
    return res;
}

// This function return a string of a number with a fixed number of digits.
// For example, fixedLength(10,8) = "00000010"
std::string fixedLength(int value, int digits) {
    unsigned int uvalue = value;
    if (value < 0) {
        uvalue = -uvalue;
    }
    std::string result;
    while (digits-- > 0) {
        result += ('0' + uvalue % 10);
        uvalue /= 10;
    }
    if (value < 0) {
        result += '-';
    }
    std::reverse(result.begin(), result.end());
    return result;
}