#ifndef STEERABLEFILTER2DMULTISCALEORTHOGONAL_H_
#define STEERABLEFILTER2DMULTISCALEORTHOGONAL_H_

#include "SteerableFilter2DMultiScale.h"
#include "SteerableFilter2DOrthogonal.h"

class SteerableFilter2DMultiScaleOrthogonal : public SteerableFilter2DMultiScale
{
public:
  SteerableFilter2DMultiScaleOrthogonal
  (string _filename_image, int _M,
   float _scale_init, float _scale_end, float _scale_step,
   string name_output = "result.jpg", string name_output_orientation = "orientation.jpg");
};
#endif
