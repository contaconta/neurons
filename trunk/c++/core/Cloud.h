#ifndef CLOUD_H_
#define CLOUD_H_

#include "Cloud_P.h"

using namespace std;

template <class T = Point>
class Cloud : public Cloud_P
{
public:

  //Points in the cloud
  // vector< T* > points;
  
  Cloud() : Cloud_P() {}

  Cloud(string filename);

  void draw();

  bool load(istream &in);

  void save(ostream &out);
};


// FOR NOW, DEFINITIONS HERE, BUT SHOULD CHANGE
template <class T>
 Cloud<T>::Cloud(string filename) : Cloud_P()
{
  if (fileExists(filename))
    loadFromFile(filename);
}

template <class T>
 void Cloud<T>::draw(){
  VisibleE::draw();

  if(v_glList == 0){
    // Reduces the number of points to 2000
    float step = points.size() / 2000;
    if(step < 1)
      step = 1;
    else
      printf("Cloud::draw there is a subsampling of the points\n");
    v_glList = glGenLists(1);
    glNewList(v_glList, GL_COMPILE);
    for(float i = 0; i < points.size(); i+=step){
      points[(int)i]->draw(v_radius);
    }
    glEndList();
  }
  else{
    glCallList(v_glList);
  }
}

template <class T>
bool Cloud<T>::load(istream &in){
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
  orig = s.find(T::className()+">");
  if(orig == string::npos){
    printf("Cloud::error load called when there is no type of the class\n");
    in.seekg(start);
    return false;
  }

  if(!VisibleE::load(in))
    return false;
  T* p = new T();
  while(p->load(in)){
    points.push_back(p);
    p = new T();
  }
  in >> s;
  if(s.find("</Cloud>")==string::npos){
    printf("Cloud::error load can not find </Cloud>\n");
    in.seekg(start);
    return false;
  }
  return true;
}

template <class T>
void Cloud<T>::save(ostream &out){
  out << "<Cloud " << T::className() << ">" << std::endl;
  VisibleE::save(out);
  for(int i = 0; i < points.size(); i++)
    points[i]->save(out);
  out << "</Cloud>" << std::endl;
}

#endif
