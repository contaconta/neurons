#include "RedundancyMapExtractor.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>

// Estimates the redundant intensity value as either 0 or 255 and extracts 
// the map for this intensity value. 
// The input image must be single channel.
IplImage* RedundancyMapExtractor::EstimInitRedundancyMap(
								IplImage* pImg,
								int& rnNoOfRadundantPels)
{
	IplImage* pRedundacyMap;
	int nNoOfMinPels;
	int nNoOfMaxPels;
	IplImage* pMinImg;
	IplImage* pMaxImg;
	
	pMinImg = cvCreateImage(cvGetSize(pImg), IPL_DEPTH_8U, 1);	
	pMaxImg = cvCreateImage(cvGetSize(pImg), IPL_DEPTH_8U, 1);
	
	cvCmpS( pImg, 0, pMinImg, CV_CMP_EQ );
	cvCmpS( pImg, 255, pMaxImg, CV_CMP_EQ );
	nNoOfMinPels = cvCountNonZero( pMinImg );
	nNoOfMaxPels = cvCountNonZero( pMaxImg );
	
	if( nNoOfMaxPels > nNoOfMinPels )
	{
		pRedundacyMap = pMaxImg;
		rnNoOfRadundantPels = nNoOfMaxPels;
		
		cvReleaseImage(&pMinImg);
	}
	else 
	{
		pRedundacyMap = pMinImg;
		rnNoOfRadundantPels = nNoOfMinPels;
		
		cvReleaseImage(&pMaxImg);
	}
	
	return pRedundacyMap;
}

IplImage* RedundancyMapExtractor::EstimRedundancyMap(
		std::string sFileName,
		int nDilationRadi,
		int nAreaThrhld,
		int nNoOfChildrenThrhld)
{
	//Declerations
	int nFrameWidth;
	int nFrameHeight;
	IplImage* pRedundancyMap;
	IplImage* pOrigImg;
	int nContourArea;
	CvMemStorage* pStorage;
	CvSeq* pContours;
	int nNoOfRadundantPels;
	IplConvKernel* pStructuringElem;
	bool bDeleteContour;
	CvSeq* pContoursChilds;
	int nChildCntr;

	//Initializations
	pOrigImg = cvLoadImage(sFileName.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	nFrameWidth = pOrigImg->width;
	nFrameHeight = pOrigImg->height;
	pContours = NULL;
	
	// Estimate the initial redundancy map
	pRedundancyMap = EstimInitRedundancyMap( pOrigImg, nNoOfRadundantPels);
	
	if( nNoOfRadundantPels < nAreaThrhld )
	{
		cvSetZero( pRedundancyMap );
		
		cvReleaseImage(&pOrigImg);
		
		return pRedundancyMap;
	}
		
	// Remove small regions in the initial redundancy map. 
	
	//ATTENTION, nonzero pels are modified by 
	//this function, but surely are not turned to zero :))
	//In fact, the pixels that are at the contour boundaries
	//are set to some value greater than 1, and the internal pixels 
	//are set to the value of 1.
	pStorage = cvCreateMemStorage(0);
	cvFindContours( 
				   pRedundancyMap, 
				   pStorage, 
				   &pContours, 
				   sizeof(CvContour), 
				   CV_RETR_CCOMP, 
				   CV_CHAIN_APPROX_SIMPLE );	
	
	for( ; 
		pContours != 0; 
		pContours = pContours->h_next )
	{
		bDeleteContour = false;
		
		//Calculate the area of the current contour
		nContourArea = (int)(fabs((cvContourArea(pContours))) + 0.5);
		
		if( nContourArea < nAreaThrhld )
		{
			bDeleteContour = true;
		}
		else 
		{
			nChildCntr = 0;
			for(pContoursChilds = pContours->v_next; 
				(pContoursChilds != 0) && (nChildCntr <= nNoOfChildrenThrhld); 
				pContoursChilds = pContoursChilds->h_next )
			{
				nChildCntr++;
			}
			
			if( nChildCntr > nNoOfChildrenThrhld )
			{
				bDeleteContour = true;
			}
		}
		
		if( bDeleteContour )
		{
			cvDrawContours(pRedundancyMap, 
						   pContours,
						   cvScalar(0,0,0), 
						   cvScalar(0,0,0),
						   0, CV_FILLED, 8);			
		}
	}
	
	// Dilate the redundancy map borders by the specified radious
	pStructuringElem = cvCreateStructuringElementEx( 
								2 * nDilationRadi + 1, 
								2 * nDilationRadi + 1,
								nDilationRadi,
								nDilationRadi,
								CV_SHAPE_RECT );
	cvDilate( pRedundancyMap, pRedundancyMap, pStructuringElem);

	// Set all nonzero pixels to 255
	cvSet( pRedundancyMap, cvScalar(255,255,255), pRedundancyMap );
	
	// Deallocations
	cvReleaseStructuringElement(&pStructuringElem);
    cvReleaseMemStorage(&pStorage);
	cvReleaseImage(&pOrigImg);
	

	return pRedundancyMap;
}
