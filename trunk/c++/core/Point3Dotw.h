#ifndef POINT3DOTW_H_
#define POINT3DOTW_H_

#include "neseg.h"

#include <vector>
#include <fstream>

#include "Point3Dot.h"

using namespace std;

/** A point in 3D*/
class Point3Dotw : public Point3Dot
{
public:

  double weight;

  Point3Dotw() : Point3Dot(){
    weight = 0;
  }

  Point3Dotw(float x, float y, float z,  float theta=0, float phi = 0,
             int type = -1, double _weight = 0.0) :
    Point3Dot(x,y,z,theta,phi,type)
  {
    weight = _weight;
  }

  void draw(float width){
    if(type == 1)
      glColor3f(0.0,0.0,1.0);
    if(type == -1)
      glColor3f(1.0,0.0,0.0);
    Point3Do::draw(width);
  }

  void save(ostream &out){
    for(int i = 0; i < 3; i++)
      out << coords[i] << " ";
    out << theta << " " << phi << " "
        << type << " " << weight << std::endl;
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
    in >> theta;
    if(in.fail()){
      in.clear();
      in.seekg(start+1); //????????? Why that one
      return false;
    }
    in >> phi;
    if(in.fail()){
      in.clear();
      in.seekg(start+1); //????????? Why that one
      return false;
    }
    in >> type;
    if(in.fail()){
      in.clear();
      in.seekg(start+1); //????????? Why that one
      return false;
    }

    in >> weight;
    if(in.fail()){
      in.clear();
      in.seekg(start+1); //????????? Why that one
      return false;
    }

    return true;
  }


  virtual string className(){
    return "Point3Dotw";
  }
};

#endif
