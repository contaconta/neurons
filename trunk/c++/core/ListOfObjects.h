#ifndef LISTOFOBJECTS_H_
#define LISTOFOBJECTS_H_

#include "VisibleE.h"
#include "GraphFactory.h"

class ListOfObjects : public VisibleE
{

public:
  vector< VisibleE* > objects;
  vector< string >    objectsName;
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
  }

  VisibleE* load_object(string name){
    string extension;
    extension = getExtension(name);
    Graph_P* gr = GraphFactory::load(name);
    return gr;
  }

  void draw(){
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

  void listen(int order){
    switch(order){
    case 1:
      objectToDraw = (objectToDraw+1)%objects.size();
      printf("Drawing object %s\n", objectsName[objectToDraw].c_str());
      break;
    case -1:
      objectToDraw = (objectToDraw-1)%objects.size();
      printf("Drawing object %s\n", objectsName[objectToDraw].c_str());
      break;
    }
  }


};




#endif
