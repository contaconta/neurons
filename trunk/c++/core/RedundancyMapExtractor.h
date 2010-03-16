#ifndef REDUNDANCY_MAP_EXTRACTOR_H
#define REDUNDANCY_MAP_EXTRACTOR_H

#include <cxcore.h>
#include <cv.h>
#include <string>

using namespace std;

class RedundancyMapExtractor
{
public:

	static IplImage* EstimRedundancyMap(std::string sFileName,
										int nDilationRadi,
										int nAreaThrhld = 0,
										int nNoOfChildrenThrhld = INT_MAX);
	
private:
	
	static IplImage* EstimInitRedundancyMap(IplImage* pImg,
											int& rnNoOfRadundantPels);
};

#endif
