/** Simple class to load cubes.*/


#ifndef CLOUD_FACTORY_H_
#define CLOUD_FACTORY_H_

#include "Cloud.h"

class CloudFactory
{
public:

  static Cloud_P* load(string filename){

    assert(fileExists(filename));
    ifstream pp(filename.c_str());
    string s;
    pp >> s;
    if(s!="<Cloud"){
      printf("CloudFactory::loadFromFile error: no <Cloud\n");
      pp.close();
      return NULL;
    }
    pp >> s;
    pp.close();
    if( s == "Point3D>")
      return new Cloud<Point3D>(filename);
    else if( s == "Point2D>")
      return new Cloud<Point2D>(filename);
    else if( s == "Point2Do>")
      return new Cloud<Point2Do>(filename);
    else if( s == "Point2Dot>")
      return new Cloud<Point2Dot>(filename);
    else{
      printf("CloudFactory:: not such type of point %s\n", s.c_str());
      return NULL;
    }
  }

};


#endif
