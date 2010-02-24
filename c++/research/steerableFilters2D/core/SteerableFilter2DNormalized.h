
#ifndef STEERABLEFILTER2DNORMALIZED_H_
#define STEERABLEFILTER2DNORMALIZED_H_

#include "SteerableFilter2D.h"

class SteerableFilter2DNormalized : public SteerableFilter2D
{
public:
  /** Constructor.*/
  SteerableFilter2DNormalized(string filename_image, int M = 2, float sigma = 1.0);

  SteerableFilter2DNormalized(string filename_image, string filename_parameters, float sigma);

  /** Calculate the  coefficients.*/
  void calculate_coefficients();

  void calculate_steering_coefficients(double theta);

};


#endif
