#ifndef STEERABLEFILTER2DM_H_
#define STEERABLEFILTER2DM_H_

#include "SteerableFilter2D.h"
#include "CloudFactory.h"
#include <gsl/gsl_blas.h>

class SteerableFilter2DM : public SteerableFilter2D
{

public:

  bool includeOddOrders;
  bool includeEvenOrders;
  bool includeOrder0;

  SteerableFilter2DM(string imageName, int M, double sigma,
                     string resultName, bool includeOddTerms = true,
                     bool includeEvenTerms = true,
                     bool includeOrder0   = false);

  void outputCoordinates(string trainingPoints, string filename, bool rotated = true);

  virtual vector< float > getDerivativeCoordinatesRotated(int x, int y, float theta);


};


SteerableFilter2DM::SteerableFilter2DM
(string imageName, int _M, double _sigma,
 string resultName, bool _includeOddTerms,
 bool _includeEvenTerms,
 bool _includeOrder0) :
  SteerableFilter2D(imageName, _M, _sigma)
{

  includeOddOrders  = _includeOddTerms;
  includeEvenOrders = _includeEvenTerms;
  includeOrder0     = _includeOrder0;

}



void SteerableFilter2DM::outputCoordinates
(string trainingPoints, string filename, bool rotated)
{
  Cloud< Point2Dot >* training_set = new Cloud< Point2Dot >(trainingPoints);

  std::ofstream out(filename.c_str());
  vector< int > indexes(2);
  vector< float > coordinates_sr =
    getDerivativeCoordinatesRotated(0,0,0);

  bool torch_format = true;
  if(torch_format){
    out << training_set->points.size() << " "
        <<  coordinates_sr.size()+1 << std::endl;
  }

  for(int nP = 0; nP < training_set->points.size(); nP++){
    Point2Dot* pp = dynamic_cast<Point2Dot*>(training_set->points[nP]);
    vector< float > micrometers(2);
    //Get the derivative coordinates
    micrometers[0] = pp->coords[0];
    micrometers[1] = pp->coords[1];
    image->micrometersToIndexes(micrometers, indexes);
    vector< float > coordinates;
    coordinates = getDerivativeCoordinatesRotated(indexes[0], indexes[1],
                                                     pp->theta);

    //Output them
    for(int i = 0 ; i < coordinates.size(); i++)
      out << coordinates[i] << " " ;

    if(torch_format)
      out << pp->type;
    out << std::endl;
  }
  out.close();
}


vector< float > SteerableFilter2DM::getDerivativeCoordinatesRotated
(int x, int y, float theta)
{
  /* Not working for some reason!!
  for(int i = 0; i < derivatives.size(); i++){
    gsl_vector_set(alpha, i, derivatives[i]->at(x,y));
  }
  calculate_steering_coefficients(-theta);
  vector< float > toRet;
  for(int i = 0; i < derivatives.size(); i++){
    toRet.push_back(gsl_vector_get(b_theta, i));
  }
  return toRet;
  */


  vector< float > toReturn;

  gsl_vector* derivs = gsl_vector_alloc(alpha->size);
  gsl_vector* coords = gsl_vector_alloc(alpha->size);

  gsl_matrix* rot = get_rotation_matrix(-theta);
  for(int i = 0; i < derivatives.size(); i++)
    gsl_vector_set(derivs, i,
                   derivatives[i]->at(x,y) );

  gsl_blas_dgemv(CblasTrans, 1.0, rot, derivs, 0, coords);

  for(int m = 1; m <= M; m++){
    if((m%2==0) && (!includeEvenOrders))
      continue;
    if((m%2==1) && (!includeOddOrders))
      continue;
    int offset = (m+2)*(m-1)/2;
    for(int i = 0; i <= m; i++){
      //      printf("M=%i m=%i, i=%i, idx = %i\n", M, m, i, offset+i);
      toReturn.push_back(gsl_vector_get(coords, offset + i));
    }
  }

  gsl_matrix_free(rot);
  gsl_vector_free(derivs);
  gsl_vector_free(coords);
  return toReturn;
}

#endif
