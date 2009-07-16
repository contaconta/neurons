#ifndef CLOUD_H_
#define CLOUD_H_

#include <fstream>
#include <string>
#include <vector>

using namespace std;

class Point
{
 public:
  vector<float> coords;

  int type;

  string className(){
    return "Point2Dot";
  }

  bool load(istream &in){
    coords.resize(3);
    int start = in.tellg();
    for(int i = 0; i < 2; i++){
      in >> coords[i];
      if(in.fail()){
        in.clear();
        in.seekg(start+1); //????????? Why that one
        return false;
      }
    }
    in >> type;
    if(in.fail()){
      in.clear();
      in.seekg(start+1); //????????? Why that one
      return false;
    }
    return true;
  }

};

class Cloud
{
public:

  vector< Point* > points;

  Cloud(string filename);

  ~Cloud();

  bool loadFromFile(string filename);

  bool load(istream &in);

  //void save(ostream &out);

  virtual string className(){
    return "Cloud";
  }
};

Cloud::~Cloud()
{
  for(vector< Point* >::iterator it = points.begin();
      it != points.end();it++)
    {
      delete *it;
    }
}

Cloud::Cloud(string filename)
{
  loadFromFile(filename);
}

bool Cloud::loadFromFile(string filename){
  bool res = false;
  ifstream in(filename.c_str());
  if(in.fail())
    res = false;
  else
    {
      res = load(in);
      in.close();
    }
  return res;
}

bool Cloud::load(istream &in){
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
  Point* t = new Point();
  orig = s.find(t->className()+">");
  delete t;
  if(orig == string::npos){
    printf("Cloud::error load called when there is no type of the class %s\n", t->className().c_str());
    in.seekg(start);
    return false;
  }

  while(s!="</VisibleE>"){
    in >> s;
  }

  Point* p = new Point();
  while(p->load(in)){
    points.push_back(p);
    p = new Point();
  }
  //delete p;
  in >> s;
  if(s.find("</Cloud>")==string::npos){
    printf("Cloud::error load can not find </Cloud>\n");
    in.seekg(start);
    return false;
  }
  return true;
}

/*
void Cloud::save(ostream &out){
  Point* t = new Point();
    out << "<Cloud " << t->className() << ">" << std::endl;
  VisibleE::save(out);
  for(int i = 0; i < points.size(); i++)
    points[i]->save(out);
  out << "</Cloud>" << std::endl;
  delete t;
}
*/

#endif
