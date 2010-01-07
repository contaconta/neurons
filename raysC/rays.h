#include <list>
#include "cv.h"

using namespace std;

#define F_SOBEL 0
#define F_CANNY 1

// Parameter for canny filter
const float edge_low_thresh_default = 15000;
const float edge_high_thresh_default = 30000;
const int apertureSize = 7;

void computeDistanceDifferenceRays(const char *pImageName,
                                  int start_angle, int end_angle, int step_angle,
                                  IplImage** ray1, IplImage** ray2);

void computeRays(const char *pImageName, double sigma, double angle,
                 IplImage** ray1, IplImage** ray3, IplImage** ray4,
                 int filterType=F_CANNY, bool saveImages=false, int edge_low_threshold=edge_low_thresh_default, int edge_high_threshold=edge_high_thresh_default);

void linepoints(int img_width, int img_height ,double angle, list<int>& xs, list<int>& ys);

inline void intline(int x1, int x2, int y1, int y2, list<int>& xs, list<int>& ys,int img_width, int img_height);

void save32bitsimage(char* filename, IplImage* img);
