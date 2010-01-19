/** Simple class to load cubes.*/


#ifndef CLOUD_FACTORY_H_
#define CLOUD_FACTORY_H_

#include "Cloud_P.h"
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
    s = s.substr(0, s.size()-1); // remove the last >
    Cloud_P* cl = cloudFromType(s);
    cl->loadFromFile(filename);
    return cl;
  }

  static Cloud_P* cloudFromType(string s){
    if( s == "Point3D")
      return new Cloud<Point3D>();
    else if( s == "Point2D")
      return new Cloud<Point2D>();
    else if( s == "Point2Do")
      return new Cloud<Point2Do>();
    else if( s == "Point2Dot")
      return new Cloud<Point2Dot>();
    else if( s == "Point2Dotw")
      return new Cloud<Point2Dotw>();
    else if( s == "Point3Do")
      return new Cloud<Point3Do>();
    else if( s == "Point3Dt")
      return new Cloud<Point3Dt>();
    else if( s == "Point3Dot")
      return new Cloud<Point3Dot>();
    else if( s == "Point3Dotw")
      return new Cloud<Point3Dotw>();
    else{
      printf("CloudFactory:: not such type of point %s\n", s.c_str());
      return NULL;
    }
  }

  static string inferPointType(Cloud_P* cloudOrig){
    string pointType;
    if(typeid(*cloudOrig) == typeid(Cloud<Point2D>))
      pointType = "Point2D";
    else if(typeid(*cloudOrig) == typeid(Cloud<Point2Do>))
      pointType = "Point2Do";
    else if(typeid(*cloudOrig) == typeid(Cloud<Point2Dot>))
      pointType = "Point2Dot";
    else if(typeid(*cloudOrig) == typeid(Cloud<Point2Dotw>))
      pointType = "Point2Dotw";
    else if(typeid(*cloudOrig) == typeid(Cloud<Point3D>))
      pointType = "Point3D";
    else if(typeid(*cloudOrig) == typeid(Cloud<Point3Do>))
      pointType = "Point3Do";
    else if(typeid(*cloudOrig) == typeid(Cloud<Point3Dt>))
      pointType = "Point3Dt";
    else if(typeid(*cloudOrig) == typeid(Cloud<Point3Dot>))
      pointType = "Point3Dot";
    else if(typeid(*cloudOrig) == typeid(Cloud<Point3Dotw>))
      pointType = "Point3Dotw";
    else{
      printf("CloudFactory::No clue what the cloud is made of\n");
      exit(0);
    }
    return pointType;
  }

  static Cloud_P* newCloudSameClass(Cloud_P* cloudOrig){
    return cloudFromType(inferPointType(cloudOrig));
  }

  static Cloud_P* newCloudWithType(Cloud_P* cloudOrig){
    string pointType = inferPointType(cloudOrig);
    string pointTypeR;
    if      (pointType == "Point3D") pointTypeR = "Point3Dt";
    else if (pointType == "Point3Dt")  pointTypeR = "Point3Dt";
    else if (pointType == "Point3Do")  pointTypeR = "Point3Dot";
    else if (pointType == "Point3Dot") pointTypeR = "Point3Dot";
    return cloudFromType(pointTypeR);
  }

};


#endif
