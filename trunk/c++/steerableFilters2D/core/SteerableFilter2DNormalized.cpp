#include "SteerableFilter2DNormalized.h"


SteerableFilter2DNormalized::SteerableFilter2DNormalized
(string filename_image, int M, float sigma)
{
  directory = getDirectoryFromPath(filename_image);
  image_to_steer = filename_image;
  this->sigma = sigma;

  image = new Image<float>(filename_image,true);
  result = image->create_blank_image_float(directory + "result.jpg");
  orientation = image->create_blank_image_float(directory + "orientation.jpg");

  this->M = M;
  alpha = gsl_vector_alloc((M+3)*M/2);

  derivatives.resize((M+3)*M/2);
  b_theta = gsl_vector_alloc((M+3)*M/2);
  gsl_vector_set_all (b_theta, 0.0);
  theta = -30.109;

  calculate_steering_coefficients(theta);
  calculate_coefficients();
}

SteerableFilter2DNormalized::SteerableFilter2DNormalized(string filename_image, string filename_parameters, float sigma)
{
  directory = filename_image.substr(0,filename_image.find_last_of("/\\")+1);
  image_to_steer = filename_image;
  this->sigma = sigma;

  image = new Image<float>(filename_image,true);
  result = image->create_blank_image_float(directory + "result.jpg");
  orientation = image->create_blank_image_float(directory + "orientation.jpg");

  load_alphas(filename_parameters);

  derivatives.resize((M+3)*M/2);
  b_theta = gsl_vector_alloc((M+3)*M/2);
  gsl_vector_set_all (b_theta, 0.0);
  theta = -30.109;
//   calculate_steering_coefficients(3.14159/4);
  calculate_steering_coefficients(theta);
  calculate_coefficients();
}


void SteerableFilter2DNormalized::calculate_coefficients()
{
  ifstream inp;
  char buff[512];
  sprintf(buff, "_%02.02f.jpg",sigma);
  string name_b = directory + "gn" + buff;

  int k_idx = 0;
  for(int k = 1; k <= M; k++){
    for(int j = 0; j <= k; j++){
      string name = directory + "gn_";
      for(int l = 0; l < k-j; l++)
        name = name + "x";
      for(int l = 0; l < j; l++)
        name = name + "y";
      name = name + buff;
      //Check for the existance of the file
      inp.open(name.c_str(), ifstream::in);
      if(inp.fail()){
        derivatives[k_idx + j] =
          image->calculate_gaussian_normalized(k-j,j,sigma, name);
      }else{
        derivatives[k_idx +j] = new Image<float>(name);}
      inp.close();
    }
    k_idx += (k+1);
  }
}


void SteerableFilter2DNormalized::calculate_steering_coefficients(double theta)
{
  double ct = cos(theta);
  double st = sin(theta);

  gsl_vector_set(b_theta,0,
                 gsl_vector_get(alpha,0));

  //Outer loop for all the derivatives
  int k_idx = 0;
  for(int k = 1; k <= M; k++){
    for(int j = 0; j <= k; j++){
      //Inner loop for each coefficient
      double b_kj = 0;
      for(int i = 0; i <= k; i++){
        double sum = 0;
        for(int m = 0; m <= i; m++){
          for(int l = 0; l <= k-i; l++){
            if( k-(l+m) != j){
              continue;}
            double prod = 1;
            prod *= combinatorial(k-i,l);
            prod *= combinatorial(i,m);
            prod *= pow(-1.0,m);
            prod *= pow(ct,i+l-m);
            prod *= pow(st,k-i-l+m);
            sum = sum + prod;
          }
        }
        b_kj += gsl_vector_get(alpha,k_idx + i)*sum/
          Mask::energy2DGaussianMask(k-i,i,sigma);
      }
      b_kj *= Mask::energy2DGaussianMask(k-j,j,sigma);
      gsl_vector_set(b_theta,k_idx+j,b_kj);
    } // j loop
    k_idx += (k+1);
  } // k loop
}
