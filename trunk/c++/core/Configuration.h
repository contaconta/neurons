#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <map>
#include <sstream>
#include "Object.h"
#include "utils.h"

using namespace std;

class Configuration : public Object
{
public:
  static Configuration* pInstance;
  string filename;

  map< string, string> stock;

  static Configuration* Instance() 
  {
    /*
    if (pInstance == 0)  // is it the first call?
      {  
        pInstance = new Configuration(); // create unique instance
      }
    */
    return pInstance; // address of unique instance
  }

  static void setInstance(Configuration* aInstance) 
  {
    pInstance = aInstance;
  }

  Configuration()
  {
  };

  Configuration(string filename)
  {
    assert(fileExists(filename));
    this->filename = filename;
    loadFromFile(filename);
  };

  string retrieve(string name)
  {
    return stock[name];
  };

  int retrieveInt(string name){
    return atoi(stock[name].c_str());
  };

  int retrieveFloat(string name){
    return atof(stock[name].c_str());
  };

  bool retrieveIfExists(string name, int* ptval){
    if(stock[name]!=""){
      *ptval = atoi((stock[name]).c_str());
      return true;
    }
    return false;
  }

  bool retrieveIfExists(string name, float* ptval){
    if(stock[name]!=""){
      *ptval = atof((stock[name]).c_str());
      return true;
    }
    return false;
  }

  bool retrieveIfExists(string name, double* ptval){
    if(stock[name]!=""){
      *ptval = atof((stock[name]).c_str());
      return true;
    }
    return false;
  }

  bool retrieveIfExists(string name, string* ptval){
    if(stock[name]!=""){
      *ptval =stock[name];
      return true;
    }
    return false;
  }


  void add(string nameVariable, string value){
    stock[nameVariable] = value;
  }

  void add(string nameVariable, float value){
    char buff[512];
    sprintf(buff, "%f", value);
    stock[nameVariable] = string(buff);
  }

  void add(string nameVariable, double value){
    char buff[512];
    sprintf(buff, "%f", value);
    stock[nameVariable] = string(buff);
  }

  void add(string nameVariable, int value){
    char buff[512];
    sprintf(buff, "%i", value);
    stock[nameVariable] = string(buff);
  }


  bool load(istream& in){
    string name, attribute;
    char lineb[1024]; //lines of 1024 chars
    while(in.good()){
      in.getline(lineb,1024);
      string line(lineb);
      stringstream linestream(line);
      linestream >> name;
      if(name.length() > 0){
        if (name[0] == '#')
          continue;
      }
      linestream >> attribute;
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
