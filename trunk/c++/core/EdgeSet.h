#ifndef EDGESET_H_
#define EDGESET_H_

#include "Point3D.h"
#include "utils.h"
#include <string>
#include "Object.h"
#include "VisibleE.h"
#include "Edge.h"
#include "EdgeW.h"

using namespace std;

template < class P=Point, class E=Edge<P> >
class EdgeSet : public VisibleE
{
public:

  vector< P* >* points;

  //Points in the cloud
  vector< E* > edges;

  EdgeSet() : VisibleE() {}

  EdgeSet(string filename);

  void setPointVector(vector< P* >* _points);

  void draw();

  bool load(istream &in);

  void save(ostream &out);

  void addEdge(int p1idx, int p2idx);
};


// FOR NOW ON, DEFINITIONS HERE, BUT SHOULD CHANGE

template< class P, class E>
EdgeSet<P,E>::EdgeSet(string filename)
{
  loadFromFile(filename);
}

template< class P, class E>
bool EdgeSet<P,E>::load(istream& in){
  int start = in.tellg();
  string s;
  in >> s;
  int orig = s.find("<EdgeSet");
  if(orig == string::npos){
    printf("EdgeSet::error load called when there is no beginning of Edge\n");
    in.seekg(start);
    return false;
  }
  in >> s;
  orig = s.find(E::className()+">");
  if(orig == string::npos){
    printf("EdgeSet::error load called when there is no type of the class\n");
    in.seekg(start);
    return false;
  }

  if(!VisibleE::load(in))
    return false;
  E* e = new E();
  while(e->load(in)){
    edges.push_back(e);
    e = new E();
  }
  in >> s;
  if(s.find("</EdgeSet>")==string::npos){
    printf("EdgeSet::error load can not find </EdgeSet>\n");
    in.seekg(start);
    return false;
  }

  return true;
}

template< class P, class E>
void EdgeSet<P,E>::save(ostream &out){

  out << "<EdgeSet " << E::className() << ">" << std::endl;
  VisibleE::save(out);
  for(int i = 0; i < edges.size(); i++)
    edges[i]->save(out);
  out << "</EdgeSet>" << std::endl;
}

template< class P, class E>
void EdgeSet<P,E>::draw()
{
  VisibleE::draw();
  // printf("The EdgeSet color is: [%f, %f, %f]\n", v_r, v_g, v_b);
  // glColor3f(v_r, v_g, v_b);
  // GLfloat currCol[4];
  // glGetFloatv(GL_CURRENT_COLOR, currCol);
  // printf("The Current EdgeSet Color is [%f, %f, %f] and w is %f\n",
         // currCol[0],
         // currCol[1],
         // currCol[2], 0.0);
  glLineWidth(this->v_radius);
  for(int i = 0; i < edges.size(); i++)
    edges[i]->draw();


}

template< class P, class E>
void EdgeSet<P,E>::setPointVector(vector< P* >* _points)
{
  points = _points;
  for(int i = 0; i < edges.size(); i++)
    edges[i]->points = points;
}

template< class P, class E>
void EdgeSet<P,E>::addEdge(int p1idx, int p2idx)
{
  edges.push_back(new E(points, p1idx, p2idx));
}



#endif
