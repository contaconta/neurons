#ifndef DOUBLESET_H_
#define DOUBLESET_H_

#include <sstream>
#include "Point3D.h"
#include "Point.h"
#include "VisibleE.h"

class Point3Dc
{
 public:
  //vector<float> w_coords; // world coordinates
  vector<float> coords;

  void save(ostream &out){
    for(int i = 0; i < 2; i++)
      out << coords[i] << " ";
    out << coords[2] << std::endl;
  }

  bool load(istream &in){
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

  void draw(float width){
    glPushMatrix();
    glTranslatef(coords[0], coords[1], coords[2]);
    glutSolidSphere(width, 10, 10);
    glPopMatrix();
  }
};

template < class P=Point>
class DoubleSet : public VisibleE
{
 private:
 static int id;

 void init();

 public:
 string name;

 vector< Point3Dc* > set1;
 vector< Point3Dc* > set2;

 DoubleSet();

 ~DoubleSet();

 void addPoint(Point3Dc* point, int setId);

 void clear();

 void draw(float point_radius=1.0f);

 void save(const string& filename);

 bool load(const string& filename);

 bool load(istream &in);

 static string className(){
   return "DoubleSet";
 }

};

template< class P>
int DoubleSet<P>::id = 0;

template< class P>
DoubleSet<P>::DoubleSet() : VisibleE(){
  init();
}

template< class P>
void DoubleSet<P>::init(){
  std::string s;
  std::stringstream out;
  out << id;
  name = "Double_set " + out.str();
  id++;
}

template< class P>
//DoubleSet<P>::~DoubleSet() : ~Visible(){
DoubleSet<P>::~DoubleSet() {
  for(vector< Point3Dc* >::iterator itPoint3Dcs = set1.begin();
      itPoint3Dcs != set1.end(); itPoint3Dcs++)
    {
      delete *itPoint3Dcs;
    }
  for(vector< Point3Dc* >::iterator itPoint3Dcs = set2.begin();
      itPoint3Dcs != set2.end(); itPoint3Dcs++)
    {
      delete *itPoint3Dcs;
    }
}

template< class P>
void DoubleSet<P>::clear(){
  set1.clear();
  set2.clear();
}

template< class P>
void DoubleSet<P>::draw(float point_radius){
  //glPushAttrib(GL_LINE_BIT);
  //glLineWidth(6.0f);
  //glBegin(GL_LINE_STRIP);
  glColor3f(1,0,0);
  for(vector< Point3Dc* >::iterator itPoint3Dcs = set1.begin();
      itPoint3Dcs != set1.end(); itPoint3Dcs++)
    {
      //glVertex3f((*itPoint3Dcs)->coords[0],(*itPoint3Dcs)->coords[1],(*itPoint3Dcs)->coords[2]);
      (*itPoint3Dcs)->draw(point_radius);
    }
  glColor3f(0,1,0);
  for(vector< Point3Dc* >::iterator itPoint3Dcs = set2.begin();
      itPoint3Dcs != set2.end(); itPoint3Dcs++)
    {
      //glVertex3f((*itPoint3Dcs)->coords[0],(*itPoint3Dcs)->coords[1],(*itPoint3Dcs)->coords[2]);
      (*itPoint3Dcs)->draw(point_radius);
    }
  //glEnd();
  //glPopAttrib();
}

template< class P>
bool DoubleSet<P>::load(const string& filename)
{
  ifstream in(filename.c_str());
  bool bRes = load(in);
  in.close();
  return bRes;
}

template< class P>
bool DoubleSet<P>::load(istream &in)
{
  int start = in.tellg();
  string s;
  in >> s;
  int orig = s.find("<Cloud");
  if(orig == string::npos){
    printf("Cloud::error load called when there is no beginning of Cloud, instead: %s\n",
           s.c_str());
    in.seekg(start);
    return false;
  }
  in >> s;
  orig = s.find(P::className()+">");
  if(orig == string::npos){
    printf("Cloud::error load called when there is no type of the class %s\n", P::className().c_str());
    in.seekg(start);
    return false;
  }

  if(!VisibleE::load(in))
    return false;
  
  Point3Dc* p = new Point3Dc();
  while(p->load(in)){
    float z = p->coords[2];
    printf("p.z : %f\n", z);
    if(z==0)
      set1.push_back(p);
    else if(z==1)
      set2.push_back(p);    
    p = new Point3Dc();
  }
  delete p;

  in >> s;
  if(s.find("</Cloud>")==string::npos){
    printf("Cloud::error load can not find </Cloud>\n");
    in.seekg(start);
    return false;
  }
  return true;
}

template< class P>
void DoubleSet<P>::save(const string& filename){

  ofstream out(filename.c_str());

  if(!out.good())
    {
      printf("Error while creating file %s\n", filename.c_str());
      return;
    }
  
  out << "<Cloud " << P::className() << ">" << std::endl;
  VisibleE::save(out);

  for(vector< Point3Dc* >::iterator itPoint3Dcs = set1.begin();
      itPoint3Dcs != set1.end(); itPoint3Dcs++)
    {
      (*itPoint3Dcs)->save(out);
    }
  for(vector< Point3Dc* >::iterator itPoint3Dcs = set2.begin();
      itPoint3Dcs != set2.end(); itPoint3Dcs++)
    {
      (*itPoint3Dcs)->save(out);
    }

  out << "</Cloud>" << std::endl;
  out.close();
}

template< class P>
void DoubleSet<P>::addPoint(Point3Dc* point, int setId)
{
  if(setId == 1)
    set1.push_back(point);
  else if(setId == 2)
    set2.push_back(point);
}

#endif //DOUBLESET_H_
