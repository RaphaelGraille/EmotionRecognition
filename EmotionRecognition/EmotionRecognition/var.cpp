#include <string>
#include <map>

// This class contains all the global variables of the project
namespace var
{
    
    // The path of the directory of the project
    extern const std::string projectDir = "/Users/Raphael/Documents/Poly/Maitrise_S2/Applications_multimedias/Projet/";
    
    // The path of the directory containing the pictures of the database
    extern const std::string imagesDir = projectDir + "cohn-kanade-images/";
    
    // The path of the directory containing the labels of the database
    extern const std::string labelsDir = projectDir + "Emotion/";
    
    // The path of the directory containing the pictures of the database for face recognition
    extern const std::string FacesDir = projectDir + "cohn-kanade-images-FR/";
    
    // Parameters of the extended LBP
    extern const int neighbors = 8;
    extern const int radius = 2;
    
    // Parameters for the subdivision of each picture
    extern const int n = 7;
    extern const int m = 6;

    
    
    
}