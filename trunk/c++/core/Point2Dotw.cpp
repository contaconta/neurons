#include "Point2Dotw.h"


Point2Dotw::Point2Dotw() : Point2Dot()
{
  type = 0;
}

Point2Dotw::Point2Dotw(float x, float y, float theta, int type, float w) :
  Point2Dot(x, y, theta, type)
{
  this->w = w;
}

void Point2Dotw::draw()
{
  Point2Dot::draw(w);
 }

void Point2Dotw::draw(float width)
{
  Point2Dot::draw(w*width);
}

void Point2Dotw::save(ostream &out){
  out << coords[0] << " "
      << coords[1] << " " 
      << theta     << " "
      << type      << " "
      << w << std::endl;
}

bool Point2Dotw::load(istream &in){
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
  in >> w;
  if(in.fail()){
    in.clear();
    in.seekg(start+1); //????????? Why that one
    return false;
  }
  return true;
}

string Point2Dotw::className(){
  return "Point2Dotw";
}
