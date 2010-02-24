#ifndef STEERABLEFILTER2DMRADIUS_H_
#define STEERABLEFILTER2DMRADIUS_H_

#include "SteerableFilter2DM.h"

class RadialFunction
{
public:
  string name;
  float radius;
  RadialFunction()
  {this->name = "RadialFunction";}
  virtual vector<vector< double > > returnMask()
  {return vector<vector< double > >();}
};

class Po0 : public RadialFunction{
public:
  Po0(float radius) : RadialFunction()
  { this->radius = radius;
    this->name  = "Po0";
  }
  vector<vector< double > > returnMask(){
    vector< vector< double> > mask = allocateMatrix(2*radius+1, 2*radius+1);
    int x0 = radius;
    int y0 = radius;
    for(int x = 0; x < 2*radius+1; x++)
      for(int y = 0; y < 2*radius+1; y++){
        float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
        if(d <= radius)
          mask[x][y] = 1;
      }
  return mask;
  }
};

class Po1 : public RadialFunction{
public:
  Po1(float radius) : RadialFunction()
  { this->radius = radius;
    this->name  = "Po1";
  }
  vector<vector< double > > returnMask(){
    vector< vector< double> > mask = allocateMatrix(2*radius+1, 2*radius+1);
    int x0 = radius;
    int y0 = radius;
    for(int x = 0; x < 2*radius+1; x++)
      for(int y = 0; y < 2*radius+1; y++){
        float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
        if(d <= radius)
          mask[x][y] = d;
      }
  return mask;
  }
};

class Po2 : public RadialFunction{
public:
  Po2(float radius) : RadialFunction()
  { this->radius = radius;
    this->name  = "Po2";
  }
  vector<vector< double > > returnMask(){
    vector< vector< double> > mask = allocateMatrix(2*radius+1, 2*radius+1);
    int x0 = radius;
    int y0 = radius;
    for(int x = 0; x < 2*radius+1; x++)
      for(int y = 0; y < 2*radius+1; y++){
        float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
        if(d <= radius)
          mask[x][y] = d*d;
      }
  return mask;
  }
};

class Po3 : public RadialFunction{
public:
  Po3(float radius) : RadialFunction()
  { this->radius = radius;
    this->name  = "Po3";
  }
  vector<vector< double > > returnMask(){
    vector< vector< double> > mask = allocateMatrix(2*radius+1, 2*radius+1);
    int x0 = radius;
    int y0 = radius;
    for(int x = 0; x < 2*radius+1; x++)
      for(int y = 0; y < 2*radius+1; y++){
        float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
        if(d <= radius)
          mask[x][y] = d*d*d;
      }
  return mask;
  }
};

class Po4 : public RadialFunction{
public:
  Po4(float radius) : RadialFunction()
  { this->radius = radius;
    this->name  = "Po4";
  }
  vector<vector< double > > returnMask(){
    vector< vector< double> > mask = allocateMatrix(2*radius+1, 2*radius+1);
    int x0 = radius;
    int y0 = radius;
    for(int x = 0; x < 2*radius+1; x++)
      for(int y = 0; y < 2*radius+1; y++){
        float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
        if(d <= radius)
          mask[x][y] = d*d*d*d;
      }
  return mask;
  }
};

class Po5 : public RadialFunction{
public:
  Po5(float radius) : RadialFunction()
  { this->radius = radius;
    this->name  = "Po5";
  }
  vector<vector< double > > returnMask(){
    vector< vector< double> > mask = allocateMatrix(2*radius+1, 2*radius+1);
    int x0 = radius;
    int y0 = radius;
    for(int x = 0; x < 2*radius+1; x++)
      for(int y = 0; y < 2*radius+1; y++){
        float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
        if(d <= radius)
          mask[x][y] = d*d*d*d*d;
      }
  return mask;
  }
};

class Po6 : public RadialFunction{
public:
  Po6(float radius) : RadialFunction()
  { this->radius = radius;
    this->name  = "Po6";
  }
  vector<vector< double > > returnMask(){
    vector< vector< double> > mask = allocateMatrix(2*radius+1, 2*radius+1);
    int x0 = radius;
    int y0 = radius;
    for(int x = 0; x < 2*radius+1; x++)
      for(int y = 0; y < 2*radius+1; y++){
        float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
        if(d <= radius)
          mask[x][y] = d*d*d*d*d*d;
      }
  return mask;
  }
};

class Po7 : public RadialFunction{
public:
  Po7(float radius) : RadialFunction()
  { this->radius = radius;
    this->name  = "Po7";
  }
  vector<vector< double > > returnMask(){
    vector< vector< double> > mask = allocateMatrix(2*radius+1, 2*radius+1);
    int x0 = radius;
    int y0 = radius;
    for(int x = 0; x < 2*radius+1; x++)
      for(int y = 0; y < 2*radius+1; y++){
        float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
        if(d <= radius)
          mask[x][y] = d*d*d*d*d*d*d;
      }
  return mask;
  }
};



class Torus : public RadialFunction{
public:
  float rmax;
  float rmin;
  Torus(float rmax, float rmin) : RadialFunction()
  { this->radius = rmax;
    this->rmax = rmax;
    this->rmin = rmin;
    this->name  = "Torus";
  }
  vector<vector< double > > returnMask(){
    vector< vector< double> > mask = allocateMatrix(2*rmax+1, 2*rmax+1);
    int x0 = rmax;
    int y0 = rmax;
    for(int x = 0; x < 2*rmax+1; x++)
      for(int y = 0; y < 2*rmax+1; y++){
        float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
        if( (d <= rmax) && (d > rmin))
          mask[x][y] = 1;
        else
          mask[x][y] = -1;
      }
    return mask;
  }
};



class SteerableFilter2DMR : public SteerableFilter2DM
{
public:

  vector< RadialFunction* > rdf;
  vector< SteerableFilter2DM* > stfs;

  SteerableFilter2DMR(string imageName, int M, double sigma,
                      string resultName,
                      vector< RadialFunction* > radialFunctions,
                      bool includeOddTerms = true,
                      bool includeEvenTerms = true, bool includeOrder0 = false);

  vector< float > getDerivativeCoordinatesRotated(int x, int y, float theta);

};

SteerableFilter2DMR::SteerableFilter2DMR
(string imageName, int _M, double _sigma,
 string resultName,
 vector< RadialFunction* > radialFunctions,
 bool _includeOddTerms,
 bool _includeEvenTerms,
 bool _includeOrder0) :
  SteerableFilter2DM(imageName, _M, _sigma, resultName, _includeOddTerms, _includeEvenTerms, _includeOrder0)
{
  this->rdf = radialFunctions;

  string imageDir = getDirectoryFromPath(imageName);

  for(int i = 0; i < radialFunctions.size(); i++){
    char basisDirBuff[1024];
    sprintf(basisDirBuff, "%s/r_%i_%s/", imageDir.c_str(), (int)radialFunctions[i]->radius, radialFunctions[i]->name.c_str());
    string basisDir(basisDirBuff);
    makeDirectory(basisDir);
    string basisName = basisDir + radialFunctions[i]->name + ".jpg" ;
    if(!fileExists(basisName)){
      Image<float>* basis = image->create_blank_image_float(basisName);
      image->convolve_2D(radialFunctions[i]->returnMask(), basis);
    }
    string resultBasisName = "result.jpg";
    stfs.push_back
      (new SteerableFilter2DM(basisName, _M, _sigma, resultBasisName,
                              _includeOddTerms, _includeEvenTerms, _includeOrder0));
  }
}

vector< float > SteerableFilter2DMR::getDerivativeCoordinatesRotated
(int x, int y, float theta)
{
  vector< float > toReturn;

  for(int i = 0; i < stfs.size(); i++){
    vector< float > tmp = stfs[i]->getDerivativeCoordinatesRotated(x,y,theta);
    for(int j = 0; j < tmp.size(); j++){
      toReturn.push_back(tmp[j]);
    }
  }


  return toReturn;
}



#endif
