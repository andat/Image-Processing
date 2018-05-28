#include "stdafx.h"
#include "common.h"
#include <stdio.h>
#include <queue>
#include <random>
#include <fstream>

using namespace std;

void loadTrainTestLabel(string &pathName, vector<Mat> &trainCells, vector<Mat> &testCells,vector<int> &trainLabels, vector<int> &testLabels){

    int SZ = 20;
    Mat img = imread(pathName,CV_LOAD_IMAGE_GRAYSCALE);
    int ImgCount = 0;
    for(int i = 0; i < img.rows; i = i + SZ)
    {
        for(int j = 0; j < img.cols; j = j + SZ)
        {
            Mat digitImg = (img.colRange(j,j+SZ).rowRange(i,i+SZ)).clone();
            if(j < int(0.9*img.cols))
            {
                trainCells.push_back(digitImg);
            }
            else
            {
                testCells.push_back(digitImg);
            }
            ImgCount++;
        }
    }

    cout << "Image Count : " << ImgCount << endl;
    float digitClassNumber = 0;

    for(int z=0;z<int(0.9*ImgCount);z++){
        if(z % 450 == 0 && z != 0){
            digitClassNumber = digitClassNumber + 1;
        }
        trainLabels.push_back(digitClassNumber);
    }
    digitClassNumber = 0;
    for(int z=0;z<int(0.1*ImgCount);z++){
        if(z % 50 == 0 && z != 0){
            digitClassNumber = digitClassNumber + 1;
        }
        testLabels.push_back(digitClassNumber);
    }
}


void loadMNIST() {

    vector <Mat> trainCells;
    vector <Mat> testCells;
    vector<int> trainLabels;
    vector<int> testLabels;
    string pathName = "digits.png";
    loadTrainTestLabel(pathName, trainCells, testCells, trainLabels, testLabels);
    
}