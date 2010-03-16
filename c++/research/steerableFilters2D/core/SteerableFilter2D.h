#ifndef STEERABLEFILTER2D_H_
#define STEERABLEFILTER2D_H_

//Units in radians


#include "Image.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

class SteerableFilter2D
{
public:

  /** Pointer to the image derivatives. It has the form [gx gy gxx gxy gyy gxxx gxxy gxyy gyyy ... ]*/
  vector< Image<float>* > derivatives;

  /** Original image.*/
  Image<float>* image;

  /** Output of the algorithm.*/
  Image<float>* result;

  /** Orientation of the output.*/
  Image<float>* orientation;

  /** Vector of the coefficients of the filter for the orientation 0.*/
  gsl_vector* alpha;

  /**  Vector for the coefficients of the filter at a certain orientation.*/
  gsl_vector* b_theta;

  /** Order of the filter.*/
  int M; //derivative order of the filter.

  /** String with the name of the image.*/
  string image_to_steer;

  /** String with the directory of the image.*/
  string directory;

  /** Variance of the gaussian filter used.*/
  double sigma;

  double theta; // Last orientation used for any calculation

  bool includeOddOrders;

  bool includeEvenOrders;

  /** Null constructor.*/
  SteerableFilter2D(){
    includeOddOrders = true;
    includeEvenOrders = true;
  }

  /** Constructor.*/
  SteerableFilter2D(string filename_image, string filename_parameters, float sigma = 1.0);

  /** Constructor.*/
  SteerableFilter2D(string filename_image, int M = 2, float sigma = 1.0, bool includeOddOrders = true, bool includeEvenOrders = true);

  /** Puts in b_theta the coefficients for the new orientation.*/
  virtual void calculate_steering_coefficients(double theta);

  /** Calculate the derivatives of the image.*/
  void calculate_derivatives();

  /** Filter the image with the direction theta*/
  void filter(double theta);

  virtual double response(double theta, int x, int y);

  /** Loads the alpha matrix from the filename.*/
  void load_alphas(string filename);

  vector< double > load_vector(string filename);

  gsl_matrix* get_rotation_matrix(double angle);
};


#endif
