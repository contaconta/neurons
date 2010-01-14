#ifndef POINT2DSF_H_
#define POINT2DSF_H_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;

class Point2Dsf {

public:
  int x;       // in indexes of the image
  int y;
  float scale; // in micrometers
  float theta; // in Radians!
  int type;    // +1 - part of the neuron -1 negative sample


  Point2Dsf(){};

  static Point2Dsf* point2DfromString(string str, bool convert_to_radians = false){
    Point2Dsf* p = new Point2Dsf();
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

  static vector< Point2Dsf* > readFile(string filename, bool convert_to_radians = true){
    vector<Point2Dsf* > points;
    std::ifstream points_in(filename.c_str());
    string line;
    while(getline(points_in, line))
      {
        points.push_back(point2DfromString(line, convert_to_radians));
      }
    return points;
  }

  void print(){
    printf("Point2Dsf : x=%i y=%i scale=%f theta=%f type=%i\n",
           x, y, scale, theta, type);
  }

};



#endif
