#include "SteerableFilter2DMultiScaleNormalized.h"


SteerableFilter2DMultiScaleNormalized::SteerableFilter2DMultiScaleNormalized
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
    stf.push_back( new SteerableFilter2DNormalized
                   ( filename_image, M, s));

  alphas = gsl_vector_alloc(stf.size()*M*(M+3)/2);

  directory = getDirectoryFromPath(filename_image);
  image = new Image<float>(filename_image,true);
  result = image->create_blank_image_float(directory + name_output);
  orientation = image->create_blank_image_float(directory + name_output_orientation);
}


vector< double >
SteerableFilter2DMultiScaleNormalized::getDerivativeCoordinatesRotated
(int x, int y, double theta)
{
  vector< double > toReturn;

  gsl_vector* derivs = gsl_vector_alloc(stf[0]->alpha->size);
  gsl_vector* coords = gsl_vector_alloc(stf[0]->alpha->size);

  for(int nStf = 0; nStf < stf.size(); nStf++){
    gsl_matrix* rot = stf[nStf]->get_rotation_matrix(-theta);
    for(int i = 0; i < stf[nStf]->derivatives.size(); i++)
      gsl_vector_set(derivs, i,
                     stf[nStf]->derivatives[i]->at(x,y)
                     );

    gsl_blas_dgemv(CblasTrans, 1.0, rot, derivs, 0, coords);

    // Look at the normalization by sigma in here
    for(int i = 0; i < coords->size; i++)
      toReturn.push_back(gsl_vector_get(coords, i)
                         / stf[nStf]->sigma);

    gsl_matrix_free(rot);
  }

  gsl_vector_free(derivs);
  gsl_vector_free(coords);
  return toReturn;
}
