#include "Point3Dt.h"

Point3Dt::Point3Dt() : Point()
{
  coords.resize(3);
}


Point3Dt::Point3Dt(float x, float y, float z, int _type) : Point()
{
  coords.resize(3);
  coords[0] = x;
  coords[1] = y;
  coords[2] = z;
  type = _type;
}

void Point3Dt::draw(){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], coords[2]);
  if(type == 1)
    glColor3f(1.0,1.0,0.0);
  else
    glColor3f(0.0,1.0,1.0);
  glutSolidSphere(0.5, 10, 10);
  glPopMatrix();
}

void Point3Dt::draw(float width){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], coords[2]);
  if(type == 1)
    glColor3f(1.0,1.0,0.0);
  else
    glColor3f(0.0,1.0,0.0);
  glutSolidSphere(width, 10, 10);
  glPopMatrix();
}

void Point3Dt::save(ostream &out){
  for(int i = 0; i < 2; i++)
    out << coords[i] << " ";
  out << coords[2]  << " " << type<< std::endl;
}

bool Point3Dt::load(istream &in){
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
  in >> type;
  if(in.fail()){
    in.clear();
    in.seekg(start+1); //????????? Why that one
    return false;
  }

  return true;
}

double Point3Dt::distanceTo(Point* p)
{
  Point3Dt* p3d = dynamic_cast<Point3Dt* >(p);
  assert(p3d);
  return sqrt( (coords[0]-p3d->coords[0])*(coords[0]-p3d->coords[0]) +
               (coords[1]-p3d->coords[1])*(coords[1]-p3d->coords[1]) +
               (coords[2]-p3d->coords[2])*(coords[2]-p3d->coords[2]) );
}
