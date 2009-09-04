#include "Cloud.h"

Cloud::~Cloud()
{
  //printf("Cloud:~\n");
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
    {
      printf("Cloud::error in loadFromFile\n");
      res = false;
    }
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
