#include <list>
#include "cv.h"

using namespace std;

#define F_SOBEL 0
#define F_CANNY 1

// Parameter for canny filter
const float edge_low_thresh = 27000;
const float edge_up_thresh = 30000;
const int apertureSize = 7;

void computeRays(const char *pImageName, double sigma, double angle, IplImage** ray1, int filterType=F_CANNY);

void linepoints(int img_width, int img_height ,double angle, list<int>& xs, list<int>& ys);

inline void intline(int x1, int x2, int y1, int y2, list<int>& xs, list<int>& ys,int img_width, int img_height);
