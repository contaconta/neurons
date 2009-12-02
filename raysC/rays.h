#include <list>

using namespace std;

void computeRays(const char *pImageName, double sigma, double angle, IplImage* ray1);

void linepoints(int img_width, int img_height ,double angle, list<int>& xs, list<int>& ys);

inline void intline(int x1, int x2, int y1, int y2, list<int>& xs, list<int>& ys,int img_width, int img_height);
