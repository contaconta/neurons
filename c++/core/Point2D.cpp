#include "Point2D.h"

Point2D::Point2D() : Point()
{
  coords.resize(3);
}


Point2D::Point2D(float x, float y) : Point()
{
  coords.resize(3);
  coords[0] = x;
  coords[1] = y;
  coords[2] = 0;
}

void Point2D::draw(){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], 0);
  glutSolidSphere(0.5, 10, 10);
  glPopMatrix();
}

void Point2D::draw(float width){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], 0);
  glutSolidSphere(width, 10, 10);
  glPopMatrix();
}

void Point2D::save(ostream &out){
  out << coords[0] << " "
      << coords[1] << std::endl;
}

bool Point2D::load(istream &in){
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
  return true;
}

double Point2D::distanceTo(Point* p)
{
  Point2D* p3d = dynamic_cast<Point2D* >(p);
  assert(p3d);
  return sqrt( (coords[0]-p3d->coords[0])*(coords[0]-p3d->coords[0]) +
               (coords[1]-p3d->coords[1])*(coords[1]-p3d->coords[1]));
}
