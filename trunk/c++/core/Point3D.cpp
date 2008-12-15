#include "Point3D.h"

Point3D::Point3D() : Point()
{
  coords.resize(3);
}


Point3D::Point3D(float x, float y, float z) : Point()
{
  coords.resize(3);
  coords[0] = x;
  coords[1] = y;
  coords[2] = z;
}

void Point3D::draw(){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], coords[2]);
  glutSolidSphere(0.5, 10, 10);
  glPopMatrix();
}

void Point3D::draw(float width){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], coords[2]);
  glutSolidSphere(width, 10, 10);
  glPopMatrix();
}

void Point3D::save(ostream &out){
  // printf("Point3D::save\n");
  for(int i = 0; i < 2; i++)
    out << coords[i] << " ";
  out << coords[2] << std::endl;
}

bool Point3D::load(istream &in){
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
  return true;
}

double Point3D::distanceTo(Point* p)
{
  Point3D* p3d = dynamic_cast<Point3D* >(p);
  assert(p3d);
  return sqrt( (coords[0]-p3d->coords[0])*(coords[0]-p3d->coords[0]) +
               (coords[1]-p3d->coords[1])*(coords[1]-p3d->coords[1]) +
               (coords[2]-p3d->coords[2])*(coords[2]-p3d->coords[2]) );
}
