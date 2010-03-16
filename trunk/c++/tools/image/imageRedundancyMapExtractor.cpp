/*
 *  main.cpp
 *  RedundancyMapExtractor
 *
 *  Created by turetken on 7/18/09.
 *  (NC) No Copyright! Use it however you like ;-)
 *
 */

#include <stdio.h>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include "RedundancyMapExtractor.h"


int main(int , char* [])
{
	std::string sFileName = "/Users/engin/Documents/phd/cvlab/DIADEM/data/021.tif";
	
	IplImage* pRedundancyMap = 
		RedundancyMapExtractor::EstimRedundancyMap(sFileName.c_str(), 15, 2500, 0);

	IplImage* pOrigImg = cvLoadImage(sFileName.c_str(), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	
//	cvNamedWindow("Original Image", 1);
//	cvShowImage("Original Image", pOrigImg);
	cvSaveImage("../../OriginalImg.png", pOrigImg);
	
//	cvNamedWindow("Redundancy Map", 1);
//	cvShowImage("Redundancy Map", pRedundancyMap);
	cvSaveImage("../../RedundancyMap.png", pRedundancyMap);
	
	cvReleaseImage(&pRedundancyMap);
	cvReleaseImage(&pOrigImg);
	
	return 0;
}