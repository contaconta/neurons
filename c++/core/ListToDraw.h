#ifndef LISTTODRAW_H_
#define LISTTODRAW_H_


#include "VisibleE.h"
#include "GraphFactory.h"

class ListToDraw : public VisibleE
{

public:
  vector< VisibleE* > objects;
  vector< string >    objectsName;
  int objectToDraw;
  string directory;


  ListToDraw(string filename){
    string extension = getExtension(filename);
    directory = getDirectoryFromPath(filename);
    if(extension != "lst"){
      printf("Trying to create a ListToDraw with a file that does not have the extension lsto, quitting ... \n");
      exit(0);
    }
    objectToDraw  = 0;
    loadFromFile(filename);
  }

  VisibleE* load_object(string name){
    string extension;
    extension = getExtension(name);
    Graph_P* gr = GraphFactory::load(name);
    return gr;
  }

  void draw(){
    for(int i = 0; i < objects.size(); i++)
      objects[i]->draw();
  }

  void load_objects(){
    for(int i = 0; i < objectsName.size(); i++){
      if(fileExists(objectsName[i])){
        objects.push_back(load_object(objectsName[i]));
      } else if (fileExists(directory + "/" + objectsName[i])){
        objects.push_back(load_object(directory + "/" + objectsName[i]));
      } else{
        printf("ListToDraw::load_objects::file does not exist %s\n",
               objectsName[i].c_str());
        exit(0);
      }
    }
  }

  void save(ostream out){
    for(int i = 0; i < objectsName.size(); i++)
      out << objectsName[i] << std::endl;
  }

  bool load(istream &in){
    string s;
    while(getline(in,s))
      {
        printf("%s\n", s.c_str());
        fflush(stdout);
        objectsName.push_back(s);
      }
    load_objects();
    return true;
  }

};



#endif
