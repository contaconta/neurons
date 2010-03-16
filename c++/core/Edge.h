#ifndef EDGE_H_
#define EDGE_H_

#include "Point.h"
#include "Visible.h"

template < class P=Point>
class Edge : public Visible
{
public:

  int p0;
  int p1;
  vector< Point* >* points;

  Edge(){ p0=-1; p1=-1;}

  Edge(vector< Point* >* _points, int _p0, int _p1);

  void draw();

  void save(ostream &out);

  bool load(istream &in);

  virtual string className(){
    return "Edge";
  }

  ~Edge(){}

};

template< class P>
Edge<P>::Edge(vector< Point* >* _points, int _p0, int _p1) : Visible(){
  p0 = _p0;
  p1 = _p1;
  points = _points;
}

template< class P>
void Edge<P>::draw(){
  // Prevent algorithms that put edges connected to -1
  if(0){
    if(p0 == -1){
      P* pp1 = dynamic_cast<P*>((*points)[p1]);
      glPushMatrix();
      glTranslatef(pp1->coords[0],pp1->coords[1],pp1->coords[2]);
      glColor3f(1.0,1.0,0.0);
      glutSolidSphere(2.0, 10,10);
      glPopMatrix();
      return;
    }
    if(p1 == -1){
      P* pp0 = dynamic_cast<P*>((*points)[p0]);
      glPushMatrix();
      glTranslatef(pp0->coords[0],pp0->coords[1],pp0->coords[2]);
      glColor3f(0.5,0.5,0.0);
      glutSolidSphere(2.0, 10,10);
      glPopMatrix();
      return;
    }
  }
  if( (p0==-1) || (p1==-1))
    return;
  P* pp0 = dynamic_cast<P*>((*points)[p0]);
  P* pp1 = dynamic_cast<P*>((*points)[p1]);
  glBegin(GL_LINES);
  glVertex3f(pp0->coords[0],pp0->coords[1],pp0->coords[2]);
  glVertex3f(pp1->coords[0],pp1->coords[1],pp1->coords[2]);
  glEnd();
}

template< class P>
void Edge<P>::save(ostream &out){
  out << p0 << " " << p1 <<  std::endl;
}

template< class P>
bool Edge<P>::load(istream &in){
  int start = in.tellg();
  in >> p0;
  if(in.fail()){
    in.clear();
    in.seekg(start+1);
    return false;
  }
  in >> p1;
  if(in.fail()){
    in.clear();
    in.seekg(start+1);
    return false;
  }
  return true;
}


#endif
