#ifndef STEERABLEFILTER2DMULTISCALE_H_
#define STEERABLEFILTER2DMULTISCALE_H_

#include "SteerableFilter2D.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include "Point2Dsf.h"
#include "CloudFactory.h"

/** Implements a steerable filter with different scales.*/

class SteerableFilter2DMultiScale
{
public:

  /** Each scale will be an individual steerable filter.*/
  vector< SteerableFilter2D* > stf;

  /** Output of the algorithm.*/
  Image<float>* result;

  Image<float>* orientation;


  /** Original image.*/
  Image<float>* image;

  string directory;

  int M;
  float scale_init;
  float scale_end;
  float scale_step;
  string filename_image;
  gsl_vector* alphas;

  SteerableFilter2DMultiScale(){};

  SteerableFilter2DMultiScale
  (string _filename_image,
   int _M,
   float _scale_init,
   float _scale_end,
   float _scale_step,
   string filename_output = "result.jpg",
   string filename_orientation = "resultTheta.jpg");

  SteerableFilter2DMultiScale
  (string _filename_image,
   string _filename_alphas,
   float _scale_init,
   float _scale_end,
   float _scale_step,
   string filename_output = "result.jpg",
   string filename_orientation = "resultTheta.jpg");


  /** Outputs the training coordinates for the training samples.
      The formart of the output is compatible with torch.
      -- output_file ---
      nSamples nDimensions+type
      g_x_s0 g_y_s0 ... g_yyyy_s0 g_x_s1 .... g_yyyy_s1 .... g_yyyy_sn type
  **/
  void outputCoordinates
   (string training_points,
    string filename,
    bool convert_to_radians = false);


  /** Outputs the training coordinates for the training samples at angles 00:10:180.
      The formart of the output is compatible with torch.
      -- output_file ---
      nSamples*18 nDimensions+type
      g_x_s0_00 g_y_s0_00 ... g_yyyy_s0_00 g_x_s1_00 .... g_yyyy_s1_00 .... g_yyyy_sn_00 type
      ...
      g_x_s0_00 g_y_s0_10 ... g_yyyy_s0_10 g_x_s1_10 .... g_yyyy_s1_10 .... g_yyyy_sn_10 type
  **/
  void outputCoordinatesAllOrientations
   (string training_points,
    string filename,
    bool convert_to_radians = false);



  /** Load the alpha coefficients stored in a file. The format of the alphas is the same as before.*/
  void loadAlphas(string filename);

  double response(double theta, int x, int y);

  /** Filter the image with the direction theta*/
  void filter(double theta);

  virtual vector< double > getDerivativeCoordinatesRotated(int x, int y, double theta);

  // A simple factorial (should be somewhere else).
  int factorial(int i);

};



#endif
