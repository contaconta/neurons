#ifndef POINT2D_H_
#define POINT2D_H_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;

class Point2D {

public:
  int x;       // in indexes of the image
  int y;
  float scale; // in micrometers
  float theta; // in Radians!
  int type;    // +1 - part of the neuron -1 negative sample


  Point2D(){};

  static Point2D* point2DfromString(string str, bool convert_to_radians = false){
    Point2D* p = new Point2D();
    stringstream ss(str);
    double d;
    ss >> d;
    p->x = (int)d;
    ss >> d;
    p->y = (int)d;
    ss >> d;
    p->scale = (float)d;
    ss >> d;
    p->theta = (float)d;
    ss >> d;
    p->type = (int)d;
    if(convert_to_radians)
      p->theta = p->theta*M_PI/180;
    return p;
  }

  static vector< Point2D* > readFile(string filename, bool convert_to_radians = true){
    vector<Point2D* > points;
    std::ifstream points_in(filename.c_str());
    string line;
    while(getline(points_in, line))
      {
        points.push_back(point2DfromString(line, convert_to_radians));
      }
    return points;
  }

  void print(){
    printf("Point2D: x=%i y=%i scale=%f theta=%f type=%i\n",
           x, y, scale, theta, type);
  }

};



#endif
