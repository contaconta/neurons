
#ifndef STEERABLEFILTER2DHERMITE_H_
#define STEERABLEFILTER2DHERMITE_H_

#include "SteerableFilter2D.h"

class SteerableFilter2DHermite : public SteerableFilter2D
{
public:
  /** Constructor.*/
  SteerableFilter2DHermite(string filename_image, int M = 2, float sigma = 1.0);

  SteerableFilter2DHermite(string filename_image, string filename_parameters, float sigma);

  /** Calculate the hermite coefficients.*/
  void calculate_coefficients();

};


#endif
