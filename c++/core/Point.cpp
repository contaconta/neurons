#include "Point.h"

Point::Point() : Visible(){
  coords.resize(3);
  coords[0] = 0;
  coords[1] = 0;
  coords[2] = 0;
}

Point::Point(float x, float y, float z) : Visible()
{
  coords.resize(3);
  coords[0] = x;
  coords[1] = y;
  coords[2] = z;
}

void Point::draw(){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], coords[2]);
  glutSolidSphere(0.5, 10, 10);
  glPopMatrix();
}

void Point::draw(float width){
  glPushMatrix();
  glTranslatef(coords[0], coords[1], coords[2]);
  glutSolidSphere(width, 10, 10);
  glPopMatrix();
}

// void Point::save(ostream &out){
  // for(int i = 0; i < 2; i++)
    // out << coords[i] << " ";
  // out << coords[2] << std::endl;
// }

// bool Point::load(istream &in){
  // coords.resize(3);
  // int start = in.tellg();
  // for(int i = 0; i < 3; i++){
    // in >> coords[i];
    // if(in.fail()){
      // in.clear();
      // in.seekg(start+1); //????????? Why that one
      // return false;
    // }
  // }
  // return true;
// }

ostream& operator <<(ostream &os,const Point &point)
{
//    for(itCoords = point.coords.begin();
//        itCoords != point.coords.end(); itCoords++)
//    {
//        os << *itCoords << " ";
//    }
    for(int i=0;i<point.coords.size();i++)
    {
        if(i==point.coords.size()-1)
            os << point.coords[i]<<endl;
        else
            os << point.coords[i] << " ";
    }
    return os;
}
