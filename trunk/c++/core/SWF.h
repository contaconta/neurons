#ifndef SWF_H_
#define SWF_H_

#include "VisibleE.h"
#include "utils.h"

class SWF : public VisibleE
{
  public;

  Cloud<Point3D>* cloud;

  SWF(string filename){
    vector< vector< double > > orig = loadMatrix(filename);
    for(int i = 0; i < orig.size(); i++){
      cloud->points.puh_back
        (new Point3D(orig[i][2], orig[i][3]. orig[i][4]));
    }
  }

  void draw(){
    cloud->draw();
  }

  void save(ostream &out){};

  bool load(istream &in){};

  string className(){return "SWF";}

};
#endif
