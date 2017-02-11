#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "lbp.hpp"
#include "faceDetection.hpp"
#include "dataBaseProcessing.hpp"
#include "trainSVM.hpp"
#include "var.hpp"


// Compute K-fold Cross Validation on teh database
void crossValidation(int K, std::string DescAndLabels, std::string resPath){
    
    // number of bad detection
    float nbError =0;
    
    // confusion matrix
    int confMat[7][7];
    
    for (int i = 0; i<7; i++) {
        for (int j = 0; j<7; j++) {
            confMat[i][j] = 0;
        }
    }

    // Get the Descriptors and labels for the database
    FileStorage fs(var::projectDir + DescAndLabels, FileStorage::READ);
    
    Mat descriptors;
    Mat labels;
    
    fs["descriptors"] >> descriptors;
    fs["labels"] >> labels;
    
    fs.release();
    
    // Create a random vetor of indexes for the croos valodation
    std::vector<int> indexes(descriptors.rows, 0);
    
    for (int i=0; i<indexes.size(); i++) {
        indexes[i] = i;
    }
    std::random_shuffle ( indexes.begin(), indexes.end() );
    
    // A vector containing the indexes of the boundaries each of the K group
    std::vector<int> kFoldIndexes(K+1,0);
    
    for (int i = 0; i<K; i++) {
        kFoldIndexes[i] = i*indexes.size()/K;
    }
    
    kFoldIndexes[K] = indexes.size();
    
    // for each of the K group
    for (int k = 0; k<K; k++) {
        
        // Create the indexes for the training set
        std::vector<int> trainIndex = indexes;
        trainIndex.erase(trainIndex.begin() + kFoldIndexes[k], trainIndex.begin() + kFoldIndexes[k+1] -1);
        
        // Create the indexes for the test set
        std::vector<int> testIndex(indexes.begin() + kFoldIndexes[k], indexes.begin() + kFoldIndexes[k+1] -1);
        
        
        // Get the train and test descriptors and labels
        Mat trainDescriptors;
        Mat testDescriptors;
        
        Mat trainLabels;
        Mat testLabels;
        
        // For each element in the training set
        for (std::vector<int>::const_iterator i = trainIndex.begin(); i != trainIndex.end(); i++) {
            
            trainDescriptors.push_back(descriptors.row(*i));
            trainLabels.push_back(labels.row(*i));
            
        }
        
        // For each element in the test set
        for (std::vector<int>::const_iterator i = testIndex.begin(); i != testIndex.end(); i++) {
            
            testDescriptors.push_back(descriptors.row(*i));
            testLabels.push_back(labels.row(*i));
            
        }
        
        // Initialize the SVM
        CvSVMParams params;
        
        params.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
        params.degree = 0; // for poly
        params.gamma = 1e-06; // for poly/rbf/sigmoid
        params.coef0 = 0;
        params.C = 2.5;
        params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,1000000,0.000001);
        
        // train it
        CvSVM svm;
        svm.train(trainDescriptors, trainLabels, Mat(), Mat(), params);
        
        // Or load it
        //svm.load("/Users/Raphael/Documents/Poly/Maitrise_S2/Applications_multimedias/Projet/svm_save");
        
        Mat testPredictions;
        
        // make the redictions
        svm.predict(testDescriptors, testPredictions);
        
        // Get the indexes of the pictures that got wrong detection (this was mainly for the class presentation),
        // to know in what case the detector is wrong
        for (int i=0; i<testIndex.size(); i++) {
            
            confMat[testLabels.at<int>(i)-1][int(testPredictions.at<float>(i))-1]++;
            if (testLabels.at<int>(i) != testPredictions.at<float>(i)) {
                nbError++;
                std::cout<< "Mauvaise détection pour l'image : " << testIndex[i] << ". Reconnu : " <<testPredictions.at<float>(i)<< " au lieu de : " << testLabels.at<int>(i) << std::endl;
            }
        }
        
        
    }
    
    // Write the results (precision and confuion matrix) in the Results file.
    std::ofstream outputFile(resPath);
    
    outputFile << "Taux d'erreur : "<< nbError/indexes.size()<<std::endl<<std::endl;
    
    outputFile <<"Matrice de confusion : "<<std::endl;
    for (int i =0; i<7; i++) {
        for (int j=0; j<7; j++) {
            outputFile<<confMat[i][j]<<"   ";
        }
        outputFile<<std::endl;
    }

}





