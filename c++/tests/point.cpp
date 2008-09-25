#include "CubeFactory.h"
#include "Point3D.h"

using namespace std;



int main(int argc, char **argv) {
  Point3D* pt = new Point3D();
  pt->loadFromFile("data/point.pt");
  pt->saveToFile("data/point_save.pt");
  printf("Compare data/point.pt and data/point_save.pt\n");
}
