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
             Type type = TrainingNegative, double _weight = 0.0) :
    Point3Dot(x,y,z,theta,phi,type)
  {
    weight = _weight;
  }

  void draw(float width){

    Point3Dot::draw(width);

    if(0){
    if(weight == 0.4)
      glColor3f(1.0,0.0,0.0);
    if(weight == 0.8)
      glColor3f(0.0,1.0,0.0);
    if(weight == 1.2)
      glColor3f(0.0,0.0,1.0);
    if(weight == 1.6)
      glColor3f(1.0,1.0,0.0);

    glPushMatrix();
    glTranslatef(coords[0], coords[1], coords[2]);
    float ox, oy, oz;
    ox = weight*2*cos(theta)*sin(phi);
    oy = weight*2*sin(theta)*sin(phi);
    oz = weight*2*cos(phi);
    glRotatef(-phi*180/3.1416,oy,-ox,0);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(0,0,3);
    // glVertex3f(ox, oy, oz);
    glEnd();
    glScalef(0.5, 0.5, 2);
    glScalef(weight, weight, weight);
    glutSolidSphere(1, 10, 10);
    glPopMatrix();

//     glPushMatrix();
//     glScalef(weight, weight, weight);
//     Point3Do::draw(width);
//     glPopMatrix();
    }
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
    int typeInt;
    in >> typeInt;
    type = (Type)typeInt;
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
