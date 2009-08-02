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

  string className(){
    return "Cloud";
  }
};

#endif
