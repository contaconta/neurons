#ifndef SWF_H_
#define SWF_H_

#include "VisibleE.h"
#include "utils.h"

class SWF : public VisibleE
{
public:

  Cloud<Point3D>* cloud;

  SWF(string filename){
//     printf("Here I am loading %s\n", filename.c_str());
    cloud = new Cloud<Point3D>();
    vector< vector< double > > orig = loadMatrix(filename);
    for(int i = 0; i < orig.size(); i++){
//       printf("adding %i : %f, %f, %f\n", i, orig[i][2], orig[i][3], orig[i][4]);
      cloud->points.push_back
        (new Point3D(orig[i][2]/10, orig[i][3]/10, orig[i][4]/10) );
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
