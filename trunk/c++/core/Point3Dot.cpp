#include "Point3Dot.h"

Point3Dot::Point3Dot() : Point()
{
  coords.resize(3);
  theta = 0;
  phi   = 0;
}


Point3Dot::Point3Dot
(float x, float y, float z, float _theta, float _phi, int _type) : Point()
{
  coords.resize(3);
  coords[0] = x;
  coords[1] = y;
  coords[2] = z;
  theta = _theta;
  phi =  _phi;
  type = _type;
}

void Point3Dot::draw(){
  if(type == 1)
    glColor3f(0.0,0.0,1.0);
  if(type == -1)
    glColor3f(1.0,0.0,0.0);
  Point3Dot::draw();
}

void Point3Dot::draw(float width){
  if(type == 1)
    glColor3f(0.0,0.0,1.0);
  if(type == -1)
    glColor3f(1.0,0.0,0.0);
  Point3Dot::draw(width);
}

void Point3Dot::save(ostream &out){
  for(int i = 0; i < 3; i++)
    out << coords[i] << " ";
  out << theta << " " << phi << " "
      << type << std::endl;
}

bool Point3Dot::load(istream &in){
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
  return true;
}

double Point3Dot::distanceTo(Point* p)
{
  Point3Dot* p3d = dynamic_cast<Point3Dot* >(p);
  assert(p3d);
  return sqrt( (coords[0]-p3d->coords[0])*(coords[0]-p3d->coords[0]) +
               (coords[1]-p3d->coords[1])*(coords[1]-p3d->coords[1]) +
               (coords[2]-p3d->coords[2])*(coords[2]-p3d->coords[2]) );
}
