#include "Point3Do.h"

Point3Do::Point3Do() : Point()
{
  coords.resize(3);
  theta = 0;
  phi   = 0;
}


Point3Do::Point3Do(float x, float y, float z, float _theta, float _phi) : Point()
{
  coords.resize(3);
  coords[0] = x;
  coords[1] = y;
  coords[2] = z;
  theta = _theta;
  phi =  _phi;
}

void Point3Do::draw(){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], coords[2]);
  float ox, oy, oz;
  ox = cos(theta)*sin(phi);
  oy = sin(theta)*sin(phi);
  oz = cos(phi);
  glBegin(GL_LINES);
  glVertex3f(0,0,0);
  glVertex3f(ox, oy, oz);
  glEnd();
  glutSolidSphere(0.5, 10, 10);
  glPopMatrix();
}

void Point3Do::draw(float width){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], coords[2]);
  float ox, oy, oz;
  ox = width*2*cos(theta)*sin(phi);
  oy = width*2*sin(theta)*sin(phi);
  oz = width*2*cos(phi);
  glBegin(GL_LINES);
  glVertex3f(0,0,0);
  glVertex3f(ox, oy, oz);
  glEnd();
  glutSolidSphere(width, 10, 10);
  glPopMatrix();
}

void Point3Do::save(ostream &out){
  // printf("Point3Do::save\n");
  for(int i = 0; i < 3; i++)
    out << coords[i] << " ";
  out << theta << " " << phi << std::endl;
}

bool Point3Do::load(istream &in){
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
  return true;
}

double Point3Do::distanceTo(Point* p)
{
  Point3Do* p3d = dynamic_cast<Point3Do* >(p);
  assert(p3d);
  return sqrt( (coords[0]-p3d->coords[0])*(coords[0]-p3d->coords[0]) +
               (coords[1]-p3d->coords[1])*(coords[1]-p3d->coords[1]) +
               (coords[2]-p3d->coords[2])*(coords[2]-p3d->coords[2]) );
}
