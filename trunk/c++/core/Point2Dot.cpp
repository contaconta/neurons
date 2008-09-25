#include "Point2Dot.h"


Point2Dot::Point2Dot() : Point2Do()
{
  type = 0;
}

Point2Dot::Point2Dot(float x, float y, float theta, int type) :
Point2Do(x, y, theta)
{
  this->type = type;
}

void Point2Dot::draw()
{
  if(type == 1)
    glColor3f(0.0,0.0,1.0);
  if(type == -1)
    glColor3f(1.0,0.0,0.0);
  Point2Do::draw();
 }

void Point2Dot::draw(float width)
{
  if(type == 1)
    glColor3f(0.0,0.0,1.0);
  if(type == -1)
    glColor3f(1.0,0.0,0.0);
  Point2Do::draw(width);
}

void Point2Dot::save(ostream &out){
  out << coords[0] << " "
      << coords[1] << " " 
      << theta     << " "
      << type      << std::endl;
}

bool Point2Dot::load(istream &in){
  coords.resize(3);
  int start = in.tellg();
  for(int i = 0; i < 2; i++){
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
  in >> type;
  if(in.fail()){
    in.clear();
    in.seekg(start+1); //????????? Why that one
    return false;
  }
  return true;
}
