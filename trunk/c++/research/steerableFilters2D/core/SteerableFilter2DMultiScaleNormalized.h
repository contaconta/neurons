#ifndef STEERABLEFILTER2DMULTISCALENORMALIZED_H_
#define STEERABLEFILTER2DMULTISCALENORMALIZED_H_

#include "SteerableFilter2DMultiScale.h"
#include "SteerableFilter2DNormalized.h"

class SteerableFilter2DMultiScaleNormalized : public SteerableFilter2DMultiScale
{
public:
  SteerableFilter2DMultiScaleNormalized
  (string _filename_image, int _M,
   float _scale_init, float _scale_end, float _scale_step,
   string name_output = "result.jpg", string name_output_orientation = "orientation.jpg");

  vector< double > getDerivativeCoordinatesRotated(int x, int y, double theta);


};
#endif
