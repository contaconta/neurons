#ifndef LISTOFOBJECTS_H_
#define LISTOFOBJECTS_H_

#include "VisibleE.h"
#include "GraphFactory.h"
#include "CloudFactory.h"
#include "ListToDraw.h"

class ListOfObjects : public VisibleE
{

public:
  vector< VisibleE* > objects;
  vector< string >    objectsName;
  vector< uchar  >    object_loaded;
  int objectToDraw;
  string directory;


  ListOfObjects(string filename){
    string extension = getExtension(filename);
    directory = getDirectoryFromPath(filename);
    if(extension != "lsto"){
      printf("Trying to create a ListOfObjects with a file that does not have the extension lsto, quitting ... \n");
      exit(0);
    }
    objectToDraw  = 0;
    loadFromFile(filename);
    objects.resize(objectsName.size());
    objects[objectToDraw] = load_object(objectsName[objectToDraw]);
    object_loaded[objectToDraw] = 1;
    printf("ListOfObjects::Constructor done\n");
  }

  VisibleE* load_object(string name){
    string extension;
    extension = getExtension(name);
    if(extension == "gr"){
      Graph_P* gr = GraphFactory::load(name);
      return gr;
    }
    if(extension == "lst"){
      ListToDraw* lst = new ListToDraw(name);
      return lst;
    }
    if(extension == "cl"){
      Cloud_P* cl = CloudFactory::load(name);
      return cl;
    }

  }

  void draw(){
    if(object_loaded[objectToDraw]==1)
      objects[objectToDraw]->draw();
  }

  void load_objects(){
    for(int i = 0; i < objectsName.size(); i++){
      if(fileExists(objectsName[i])){
        objects.push_back(load_object(objectsName[i]));
      } else if (fileExists(directory + "/" + objectsName[i])){
        objects.push_back(load_object(directory + "/" + objectsName[i]));
      } else{
        printf("ListOfObjects::load_objects::file does not exist %s\n",
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
        object_loaded.push_back(0);
      }
    // load_objects();
    return true;
  }

  void listen(int order){
    switch(order){
    case 1:
      objectToDraw = (objectToDraw+1)%objects.size();
      if(!object_loaded[objectToDraw]){
        objects[objectToDraw] = load_object(objectsName[objectToDraw]);
        object_loaded[objectToDraw] = 1;
      }
      printf("Drawing object %s\n", objectsName[objectToDraw].c_str());
      break;
    case -1:
      objectToDraw = (objectToDraw-1);
      if(objectToDraw < 0) objectToDraw = objects.size()-1;
      if(!object_loaded[objectToDraw]){
        objects[objectToDraw] = load_object(objectsName[objectToDraw]);
        object_loaded[objectToDraw] = 1;
      }
      printf("Drawing object %s\n", objectsName[objectToDraw].c_str());
      break;
    }
  }


};




#endif
