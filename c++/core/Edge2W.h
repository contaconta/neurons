#ifndef EDGEW2_H_
#define EDGEW2_H_

#include "EdgeW.h"

template < class P>
class Edge2W : public EdgeW< P >
{
public:

  double w1;
  double w2;

  Edge2W() : EdgeW< P >() {w1=-1; w2=-1;}

  Edge2W(vector< Point* >* _points, int _p0, int _p1, double _w1=1, double _w2=0);

  void draw();

  void save(ostream &out);

  bool load(istream &in);

  virtual string className(){
    return "Edge2W";
  }

};

template< class P>
Edge2W<P>::Edge2W
(vector< Point* >* _points, int _p0, int _p1, double _w1, double _w2) :
  EdgeW< P >(_points, _p0, _p1)
{
  // w  = _w1;
  w1 = _w1;
  w2 = _w2;
}


template< class P>
void Edge2W< P >::draw(){

  double wth1 = w1;
  double wth2 = w2;
  GLfloat currCol[4];

  if( w1 > 1)
    wth1 = 1;
  else if (w1 < 0)
    wth1 = 0.1;

  if( w2 > 1)
    wth2 = 1;
  else if (w2 < 0)
    wth2 = 0.1;


  glGetFloatv(GL_CURRENT_COLOR, currCol);

  glColor3f(wth2,
            0,
            wth1);

  Edge<P>::draw();

  glColor3f(currCol[0],
            currCol[1],
            currCol[2]);

  // printf("The Current Color is [%f, %f, %f] and w is %f\n",
         // currCol[0],
         // currCol[1],
         // currCol[2], w);

}

template< class P>
void Edge2W<P>::save(ostream &out){
  out << this->p0 << " " << this->p1 << " " << w1 << " " << w2 <<  std::endl;
}

template< class P>
bool Edge2W<P>::load(istream &in){
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
  in >> w1;
  if(in.fail()){
    in.clear();
    in.seekg(start+1);
    return false;
  }
  in >> w2;
  if(in.fail()){
    in.clear();
    in.seekg(start+1);
    return false;
  }

  return true;
}


#endif
