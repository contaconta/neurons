#ifndef EDGEW_H_
#define EDGEW_H_

#include "Edge.h"
#include <iomanip>
#include <limits>

template < class P=Point>
class EdgeW : public Edge< P >
{
public:

  double w;

  EdgeW() : Edge< P >() {w=-1;}

  EdgeW(vector< Point* >* _points, int _p0, int _p1, double _w=-1) :
    Edge< P >(_points, _p0, _p1)
  {
    w = _w;
  }


  void draw();

  void save(ostream &out);

  bool load(istream &in);

  virtual string className(){
    return "EdgeW";
  }

};


template< class P>
void EdgeW< P >::draw(){

  double wth = w;
  GLfloat currCol[4];

  if( w > 1)
    wth = 1;
  else if (w < 0)
    wth = 0.1;

  glGetFloatv(GL_CURRENT_COLOR, currCol);

  // glColor3f(w,
            // 0,
            // 1-w);
  glEnable(GL_LINE_SMOOTH);
  glLineWidth(2.0);
  Edge<P>::draw();
  glLineWidth(1.0);

  if(0){
    glColor3f(140.0/255,40.0/255,40.0/255);
    glLineWidth(12.0);
    Edge<P>::draw();

    glColor3f(225.0/255,155.0/255,1.0);
    glLineWidth(4.0);
    Edge<P>::draw();

    glLineWidth(1.0);
  }

  if(0){  //cvpr10 style
    glColor3f(140.0/255,40.0/255,40.0/255);
    glLineWidth(6.0);
    Edge<P>::draw();

    glColor3f(225.0/255,155.0/255,1.0);
    glLineWidth(3.0);
    Edge<P>::draw();

    glLineWidth(1.0);
//     glColor3f(140.0/255,40.0/255,40.0/255);
//     Edge<P>::draw();
  }


  glColor3f(currCol[0],
            currCol[1],
            currCol[2]);

  // To draw the value of the edge
  if(0){
    vector< double > pt(3);
    glColor3f(0,1.0,0);
    if(this->p0>this->p1){
      pt[0] = ((*this->points)[this->p0]->coords[0] +
               (*this->points)[this->p1]->coords[0])/2;
      pt[1] = ((*this->points)[this->p0]->coords[1] +
               (*this->points)[this->p1]->coords[1])/2;
      pt[2] = ((*this->points)[this->p0]->coords[2] +
               (*this->points)[this->p1]->coords[2])/2;
    }else{
      pt[0] = ((*this->points)[this->p0]->coords[0] +
               (*this->points)[this->p1]->coords[0])/2;
      pt[1] = ((*this->points)[this->p0]->coords[1] + 1 +
               (*this->points)[this->p1]->coords[1])/2;
      pt[2] = ((*this->points)[this->p0]->coords[2] +
               (*this->points)[this->p1]->coords[2])/2;
    }
    glPushMatrix();
    // glLoadIdentity();
    glTranslatef(pt[0], pt[1], pt[2]);
    renderString("%.02f", w);
    glPopMatrix();
    glColor3f(currCol[0],
              currCol[1],
              currCol[2]);
  }
}

template< class P>
void EdgeW<P>::save(ostream &out){
  out << std::setprecision(5) << std::fixed;
  out << this->p0 << " " << this->p1 << " ";
  out << std::setprecision(20) << std::scientific << w <<  std::endl;
}

template< class P>
bool EdgeW<P>::load(istream &in){
  int start = in.tellg();
  in >> this->p0;
  if(in.fail()){
    in.clear();
    in.seekg(start+1);
    return false;
  }
  in >> this->p1;
  if(in.fail()){
    in.clear();
    in.seekg(start+1);
    return false;
  }
  in >> w;
  if(in.fail()){
    in.clear();
    in.seekg(start+1);
    return false;
  }
  return true;
}


#endif
