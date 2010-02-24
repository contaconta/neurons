
#ifndef STEERABLEFILTER2DORTHOGONAL_H_
#define STEERABLEFILTER2DORTHOGONAL_H_

#include "SteerableFilter2D.h"

class SteerableFilter2DOrthogonal : public SteerableFilter2D
{
public:
  /** Constructor.*/
  SteerableFilter2DOrthogonal(string filename_image, int M = 2, float sigma = 1.0);

  SteerableFilter2DOrthogonal(string filename_image, string filename_parameters, float sigma);

  /** Calculate the hermite coefficients.*/
  void calculate_coefficients();

  /** Puts in b_theta the coefficients for the new orientation.*/
  virtual void calculate_steering_coefficients(double theta);
};


#endif
