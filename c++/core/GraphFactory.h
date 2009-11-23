#ifndef GRAPH_FACTORY_H_
#define GRAPH_FACTORY_H_

#include "Graph.h"
#include "CloudFactory.h"

class GraphFactory
{
public:

  static Graph_P* load(string filename){
//     assert(fileExists(filename));
    if(!fileExists(filename)){
      return new Graph<Point3Dw, EdgeW<Point3Dw> >();
    }
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
    s2 = s2.substr(0, s2.size()-1); // remove the last >
    pp.close();
    Graph_P* gr = graphFromTypes(s1, s2);
    gr->loadFromFile(filename);
    return gr;
  }

  static Graph_P* graphFromTypes(string pointType, string edgeType){
    if(pointType == "Point3D"){
      if(edgeType == "Edge"){
        return new Graph<Point3D,Edge<Point3D> >();
      }
      else if(edgeType == "EdgeW"){
        return new Graph<Point3D,EdgeW<Point3D> >();
      }
      else if(edgeType == "Edge2W"){
        return new Graph<Point3D,Edge2W<Point3D> >();
      }
      else{
        printf("GraphFactory::load error: no idea what kind of edge it is %s\n",
               edgeType.c_str());
        return NULL;
      }
    }
    else if(pointType == "Point3Dt"){
      if(edgeType == "Edge"){
        printf("GraphFactory:: returning Graph<Point3Dt,Edge<Point3Dt> >\n");
        return new Graph<Point3Dt,Edge<Point3Dt> >();
      }
      else if(edgeType == "EdgeW"){
        printf("GraphFactory:: returning Graph<Point3Dt,EdgeW<Point3Dt> >\n");
        return new Graph<Point3Dt,EdgeW<Point3Dt> >();
      }
      else if(edgeType == "Edge2W"){
        printf("GraphFactory:: returning Graph<Point3Dt,Edge2W<Point3Dt> >\n");
        return new Graph<Point3Dt,Edge2W<Point3Dt> >();
      }
      else{
        printf("GraphFactory::graphFromTypes error: no idea what kind of edge it is %s\n",
               edgeType.c_str());
        return NULL;
      }
    }
    else if(pointType == "Point3Dw"){
      if(edgeType == "Edge"){
        printf("GraphFactory:: returning Graph<Point3Dw,Edge<Point3Dw> >\n");
        return new Graph<Point3Dw,Edge<Point3Dw> >();
      }
      else if(edgeType == "EdgeW"){
        printf("GraphFactory:: returning Graph<Point3Dw,EdgeW<Point3Dw> >\n");
        return new Graph<Point3Dw,EdgeW<Point3Dw> >();
      }
      else if(edgeType == "Edge2W"){
        printf("GraphFactory:: returning Graph<Point3Dw,Edge2W<Point3Dw> >\n");
        return new Graph<Point3Dw,Edge2W<Point3Dw> >();
      }
      else{
        printf("GraphFactory::graphFromTypes error: no idea what kind of edge it is %s\n",
               edgeType.c_str());
        return NULL;
      }
    }
    else if(pointType == "Point2D"){
      if(edgeType == "Edge"){
        return new Graph<Point2D,Edge<Point2D> >();
      }
      if(edgeType == "EdgeW"){
        return new Graph<Point2D,EdgeW<Point2D> >();
      }
      else if(edgeType == "Edge2W"){
        return new Graph<Point2D,Edge2W<Point2D> >();
      }
      else{
        printf("GraphFactory::graphFromTypes error: no idea what kind of edge it is%s\n",
               edgeType.c_str());
        return NULL;
      }
    }
    else{
      printf("GraphFactory::graphFromTypes error: no idea what kind of point it is %s\n",
             pointType.c_str());
      return NULL;
    }
  }

  static Graph_P* fromCloud(Cloud_P* cl, string edgeType){
    string pointType = CloudFactory::inferPointType(cl);
    Graph_P* gr = graphFromTypes(pointType, edgeType);
    gr->changeCloud(cl);
    return gr;
  }

};



#endif
