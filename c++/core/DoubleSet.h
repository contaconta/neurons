#ifndef DOUBLESET_H_
#define DOUBLESET_H_

#include <sstream>
#include "Point3D.h"
#include "Point.h"
#include "VisibleE.h"

class PointDs
{
 public:
  vector<float> coords;

  void save(ostream &out){
    if(coords.size()>0)
      {
        for(int i = 0; i < coords.size()-1; i++)
          out << coords[i] << " ";
        out << coords[coords.size()-1] << std::endl;
      }
  }

  bool load(istream &in, int ptSize=4){
    coords.resize(ptSize);
    int start = in.tellg();
    for(int i = 0; i < ptSize; i++){
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

 vector< PointDs* > set1;
 vector< PointDs* > set2;

 DoubleSet();

 ~DoubleSet();

 void addPoint(PointDs* point, int setId);

 void clear();

 void draw(float point_radius=1.0f);

 void save(const string& filename);

 bool load(const string& filename);

 bool load(istream &in);

 virtual string className(){
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
  name = "Double_set_" + out.str();
  id++;
}

template< class P>
//DoubleSet<P>::~DoubleSet() : ~Visible(){
DoubleSet<P>::~DoubleSet() {
  for(vector< PointDs* >::iterator itPointDss = set1.begin();
      itPointDss != set1.end(); itPointDss++)
    {
      delete *itPointDss;
    }
  for(vector< PointDs* >::iterator itPointDss = set2.begin();
      itPointDss != set2.end(); itPointDss++)
    {
      delete *itPointDss;
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
  for(vector< PointDs* >::iterator itPointDss = set1.begin();
      itPointDss != set1.end(); itPointDss++)
    {
      //glVertex3f((*itPointDss)->coords[0],(*itPointDss)->coords[1],(*itPointDss)->coords[2]);
      (*itPointDss)->draw(point_radius);
    }
  glColor3f(0,1,0);
  for(vector< PointDs* >::iterator itPointDss = set2.begin();
      itPointDss != set2.end(); itPointDss++)
    {
      //glVertex3f((*itPointDss)->coords[0],(*itPointDss)->coords[1],(*itPointDss)->coords[2]);
      (*itPointDss)->draw(point_radius);
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
  P* tp = new P();
  orig = s.find(tp->className()+">");
  
  if(orig == string::npos){
    printf("Cloud::error load called when there is no type of the class %s\n", tp->className().c_str());
    in.seekg(start);
    delete tp;
    return false;
  }
  delete tp;

  if(!VisibleE::load(in))
    return false;
  
  PointDs* p = new PointDs();
  while(p->load(in)){
    float z = p->coords[p->coords.size()-1];
    printf("p.z : %f\n", z);
    if(z==0)
      set1.push_back(p);
    else if(z==1)
      set2.push_back(p);    
    p = new PointDs();
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
  
  P* tp = new P();
  out << "<Cloud " << tp->className() << ">" << std::endl;
  delete tp;
  VisibleE::save(out);

  for(vector< PointDs* >::iterator itPointDss = set1.begin();
      itPointDss != set1.end(); itPointDss++)
    {
      (*itPointDss)->save(out);
    }
  for(vector< PointDs* >::iterator itPointDss = set2.begin();
      itPointDss != set2.end(); itPointDss++)
    {
      (*itPointDss)->save(out);
    }

  out << "</Cloud>" << std::endl;
  out.close();
}

template< class P>
void DoubleSet<P>::addPoint(PointDs* point, int setId)
{
  if(setId == 1)
    set1.push_back(point);
  else if(setId == 2)
    set2.push_back(point);
}

#endif //DOUBLESET_H_
