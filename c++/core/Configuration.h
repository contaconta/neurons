#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <map.h>
#include "Object.h"

using namespace std;

class Configuration : public Object
{
public:
  string filename;

  map< string, string> stock;

  Configuration(string filename)
  {
    this->filename = filename;
    loadFromFile(filename);
  };

  string retrieve(string name)
  {
    return stock[name];
  };

  void add(string nameVariable, string value){
    stock[nameVariable] = value;
  }

  bool load(istream& in){
    string name, attribute;
    while(in.good()){
      in >> name;
      in >> attribute;
      stock[name] = attribute;
    }
    return true;
  }

  void save(ostream& out){
    for(map<string, string>::iterator it = stock.begin();
        it != stock.end();
        it++){
      out << it->first << " " << it->second << std::endl;
    }
  }

};



#endif
