#include "SteerableFilter2DMultiScaleOrthogonal.h"


SteerableFilter2DMultiScaleOrthogonal::SteerableFilter2DMultiScaleOrthogonal
(string _filename_image, int _M,
 float _scale_init, float _scale_end, float _scale_step,
 string name_output, string name_output_orientation)
{
  filename_image = _filename_image;
  M = _M;
  scale_init = _scale_init;
  scale_end  = _scale_end;
  scale_step = _scale_step;

  for(float s = scale_init; s <= scale_end; s+=scale_step)
    stf.push_back( new SteerableFilter2DOrthogonal
                   ( filename_image, M, s));

  alphas = gsl_vector_alloc(stf.size()*M*(M+3)/2);

  directory = getDirectoryFromPath(filename_image);
  image = new Image<float>(filename_image,true);
  result = image->create_blank_image_float(directory + name_output);
  orientation = image->create_blank_image_float(directory + name_output_orientation);
}
