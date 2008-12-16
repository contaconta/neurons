#ifndef EDGEW_H_
#define EDGEW_H_

#include "Edge.h"

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

  static string className(){
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

  glColor3f(wth,
            0,
            0);

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
void EdgeW<P>::save(ostream &out){
  out << this->p0 << " " << this->p1 << " " << w <<  std::endl;
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
