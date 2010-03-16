#ifndef GABORFILTER2D_H_
#define GABORFILTER2D_H_

//Units in radians

#include "Image.h"

class GaborFilter2D
{
public:

  enum eMode {MODE_AVG=0, MODE_MAX} m_mode;

  /** Original image.*/
  Image<float>* m_image;

  /** Output of the algorithm.*/
  Image<float>* m_result;
  Image<float>* m_resultNoDC;

  /** Coefficients of the filter for the last computed parameters.*/
  gsl_matrix* m_gbFilter;

  int m_sz_x;
  int m_sz_y;

  /** String with the name of the image.*/
  string m_input_image;

  /** String with the directory of the image.*/
  string m_directory;

  /** Variance of the gaussian filter used.*/
  //double sigma;

  //double orientation; // orientation of the normal to the parallel stripes of a Gabor function

  //double wavelength; // represents the wavelength of the cosine factor,

  //double phaseOffset; // phase offset

  //double aspectRatio; // spatial aspect ratio (specifies the ellipticity of the support of the Gabor function)

  /** Constructors*/
  GaborFilter2D(string filename_image, eMode mode=MODE_AVG);
  GaborFilter2D(Image<float>* image, Image<float>* result=0, eMode mode=MODE_AVG);

  /** Filter the image */
  void filter();

  double response(int x, int y);

  /** compute the filter */
  gsl_matrix* compute(
		      const double sigma,
		      const double wavelength,
		      const double psi);

  /** compute the filter at a given angle */
  gsl_matrix* compute(
		      const double angle, // radians
		      const double sigma,
		      const double wavelength,
		      const double psi,
                      bool outputCoeffs=false);

 private:
  inline void compute_filter(const double angle,
                             const double sigma,
                             const double wavelength,
                             const double psi);
  inline void update_sigmas(const double sigma);

  void init();

};

#endif
