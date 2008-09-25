#include "Point2Do.h"

Point2Do::Point2Do() : Point()
{
  coords.resize(3);
  theta = 0;
}


Point2Do::Point2Do(float x, float y, float theta) : Point()
{
  coords.resize(3);
  coords[0] = x;
  coords[1] = y;
  coords[2] = 0;
  this->theta = theta;
}

void Point2Do::draw(){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], 0);
  glutSolidSphere(0.5, 10, 10);
  glBegin(GL_LINES);
  glVertex3f(0,0,0);
  glVertex3f(20*cos(theta), 20*sin(theta),0);
  glEnd();
  glPopMatrix();
}

void Point2Do::draw(float width){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], 0);
  glutSolidSphere(width, 10, 10);
  glBegin(GL_LINES);
  glVertex3f(0,0,0);
  glVertex3f(width*3*cos(theta), width*3*sin(theta),0);
  glEnd();
  glPopMatrix();
}

void Point2Do::save(ostream &out){
  out << coords[0] << " "
      << coords[1] << " " <<  theta << std::endl;
}

bool Point2Do::load(istream &in){
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
  return true;
}

double Point2Do::distanceTo(Point* p)
{
  Point2Do* p3d = dynamic_cast<Point2Do* >(p);
  assert(p3d);
  return sqrt( (coords[0]-p3d->coords[0])*(coords[0]-p3d->coords[0]) +
               (coords[1]-p3d->coords[1])*(coords[1]-p3d->coords[1]));
}
