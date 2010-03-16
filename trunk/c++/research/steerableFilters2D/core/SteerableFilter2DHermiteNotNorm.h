
#ifndef STEERABLEFILTER2DHERMITENOTNORM_H_
#define STEERABLEFILTER2DHERMITENOTNORM_H_

#include "SteerableFilter2D.h"
#include "utils.h"

class SteerableFilter2DHermiteNotNorm : public SteerableFilter2D
{
public:
  /** Constructor.*/
  SteerableFilter2DHermiteNotNorm(string filename_image, int M = 2, float sigma = 1.0);

  SteerableFilter2DHermiteNotNorm(string filename_image, string filename_parameters, float sigma);

  double response(double theta, int x, int y);

  /** Calculate the hermite coefficients.*/
  void calculate_coefficients();

};


#endif
