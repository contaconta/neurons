#ifndef SWC_H_
#define SWC_H_

#include "VisibleE.h"
#include "utils.h"
#include "Cloud.h"
#include "Point3Dw.h"
#include "Graph.h"

class SWC : public VisibleE
{
public:

  Cloud<Point3Dw>* allPoints;

  Graph<Point3Dw, Edge<Point3Dw> >* gr;

  //To be done the distinction among points
  vector< Cloud<Point3Dw>* > cloudTypes;
  Cloud<Point3Dw>* undefined;
  Cloud<Point3Dw>* soma;
  Cloud<Point3Dw>* axon;
  Cloud<Point3Dw>* dendrite;
  Cloud<Point3Dw>* apicalDendrite;
  Cloud<Point3Dw>* forkPoint;
  Cloud<Point3Dw>* endPoint;
  Cloud<Point3Dw>* custom;
  vector<double>   offset;


  SWC(string filename){

    string filenameNoExt = getPathWithoutExtension(filename);
    string offsetTXT     = getPathWithoutExtension(filename) + ".txt";
    if(fileExists(offsetTXT)){
      offset = readVectorDouble(offsetTXT);
      printf("Offset read from %s: [%f, %f, %f]\n",
             offsetTXT.c_str(), offset[0], offset[1], offset[2]);
    }
    else {
      offset.resize(0); offset.push_back(0); offset.push_back(0); offset.push_back(0);
    }


    allPoints = new Cloud<Point3Dw>();
    soma = new Cloud<Point3Dw>();
    vector< vector< double > > orig = loadMatrix(filename);
    for(int i = 0; i < orig.size(); i++){
      double width = orig[i][5];
      if (width == 0) width = 1.0;
      allPoints->points.push_back
        (new Point3Dw(orig[i][2] + offset[0],
                      orig[i][3] + offset[1],
                      orig[i][4] + offset[2],
                      width));
      if(orig[i][6] == -1){
        soma->points.push_back
          (new Point3Dw(orig[i][2] + offset[0],
                        orig[i][3] + offset[1],
                        orig[i][4] + offset[2],
                        width));
      }
    }
    gr = new Graph<Point3Dw, Edge<Point3Dw> >(allPoints);
    for(int i = 0; i < orig.size(); i++){
      int endIdx = orig[i][6]-1;
      if( (endIdx >= 0) & (endIdx < allPoints->points.size()))
        gr->eset.addEdge(i, endIdx);
    }
    soma->v_r=1.0; soma->v_g= 1.0;
    // allPoints->v_radius = 0.1;
  }

  void draw(){
    gr->draw();
    soma->draw();
  }

  void save(ostream &out){};

  bool load(istream &in){};

  string className(){return "SWC";}

};
#endif
