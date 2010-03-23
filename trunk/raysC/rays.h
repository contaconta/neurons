#ifndef RAYS_H
#define RAYS_H

#include <list>
#include "cv.h"

using namespace std;

#define F_SOBEL 0
#define F_CANNY 1

// Parameter for canny filter
#define edge_low_thresh_default 15000.0
#define edge_high_thresh_default 30000.0
#define apertureSize 7

int computeDistanceDifferenceRays(const char *pImageName,
                                   int start_angle, int end_angle, int step_angle,
                                   IplImage**& rays2, IplImage** rays1=0,
                                   int filterType=F_CANNY, bool saveImages=false,
                                   int edge_low_threshold=edge_low_thresh_default, int edge_high_threshold=edge_high_thresh_default);

void computeRays(const char *pImageName, double angle,
                 IplImage** ray1, IplImage** ray3, IplImage** ray4,
                 int filterType=F_CANNY, bool saveImages=false,
                 int edge_low_threshold=edge_low_thresh_default, int edge_high_threshold=edge_high_thresh_default);

void linepoints(int img_width, int img_height ,double angle, list<int>& xs, list<int>& ys);

inline void intline(int x1, int x2, int y1, int y2, list<int>& xs, list<int>& ys,int img_width, int img_height);

void save32bitsimage(char* filename, IplImage* img);
void savefloatimage(char* filename, IplImage* img);

#endif //RAYS_H
