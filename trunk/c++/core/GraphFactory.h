#ifndef GRAPH_FACTORY_H_
#define GRAPH_FACTORY_H_

#include "Graph.h"

class GraphFactory
{
public:

  static Graph_P* load(string filename){

    assert(fileExists(filename));
    ifstream pp(filename.c_str());
    string s1, s2;
    pp >> s1;
    if(s1!="<Graph"){
      printf("GraphFactory::loadFromFile error: no <Graph\n");
      pp.close();
      return NULL;
    }
    pp >> s1;
    pp >> s2;
    pp.close();
    if(s1 == "Point3D"){
      if(s2 == "Edge>"){
        return new Graph<Point3D,Edge<Point3D> >(filename);
      }
      else if(s2 == "EdgeW>"){
        return new Graph<Point3D,EdgeW<Point3D> >(filename);
      }
      else{
        printf("GraphFactory::load error: no idea what kind of edge it is %s\n", s2.c_str());
        return NULL;
      }
    }
    else if(s1 == "Point2D"){
      if(s2 == "Edge>"){
        return new Graph<Point2D,Edge<Point2D> >(filename);
      }
      if(s2 == "EdgeW>"){
        return new Graph<Point2D,EdgeW<Point2D> >(filename);
      }
      else{
        printf("GraphFactory::load error: no idea what kind of edge it is %s\n", s2.c_str());
        return NULL;
      }
    }
    else{
      printf("GraphFactory::load error: no idea what kind of point it is %s\n", s1.c_str());
      return NULL;
    }

  }
};



#endif
