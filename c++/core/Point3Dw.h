#ifndef POINT3DW_H_
#define POINT3DW_H_

#include "neseg.h"

#include <vector>
#include <fstream>

#include "Point.h"

using namespace std;

/** A point in 3D*/
class Point3Dw : public Point
{
public:

  // vector< float > coords;

  float weight;

  Point3Dw(){}

  Point3Dw(float x, float y, float z=0, double weight = 0)
  {
    coords.resize(3);
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;
    this->weight = weight;
  }

  void draw(){
    glPushMatrix();
    glTranslatef(coords[0], coords[1], coords[2]);
    //    glScalef(weight, weight, weight);
    glutSolidSphere(weight, 10, 10);
    glPopMatrix();
  }

  void draw(float width){
    draw();
  }

  bool load(istream &in){
    coords.resize(3);
    int start = in.tellg();
    for(int i = 0; i < 3; i++){
      in >> coords[i];
      if(in.fail()){
        in.clear();
        in.seekg(start+1); //????????? Why that one
        return false;
      }
    }
    in >> weight;
    if(in.fail()){
      in.clear();
      in.seekg(start+1); //????????? Why that one
      return false;
    }
    return true;
  }

  void save(ostream &out){
    for(int i = 0; i < 2; i++)
      out << coords[i] << " ";
    out << coords[2]  << " " << weight << std::endl;
  }

  double distanceTo(Point* p){return 0.0;}

  virtual string className(){
    return "Point3Dw";
  }
};

#endif
