#include "SteerableFilter2DHermiteNotNorm.h"


SteerableFilter2DHermiteNotNorm::SteerableFilter2DHermiteNotNorm
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

SteerableFilter2DHermiteNotNorm::SteerableFilter2DHermiteNotNorm(string filename_image, string filename_parameters, float sigma)
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


void SteerableFilter2DHermiteNotNorm::calculate_coefficients()
{
  ifstream inp;
  char buff[512];
  sprintf(buff, "_%02.02f.jpg",sigma);
  string name_b = directory + "h" + buff;

  int k_idx = 0;
  for(int k = 1; k <= M; k++){
    for(int j = 0; j <= k; j++){
      string name = directory + "h_";
      for(int l = 0; l < k-j; l++)
        name = name + "x";
      for(int l = 0; l < j; l++)
        name = name + "y";
      name = name + buff;
      //Check for the existance of the file
      inp.open(name.c_str(), ifstream::in);
      if(inp.fail()){
        derivatives[k_idx + j] = image->calculate_hermite(k-j,j,sigma, name);
      }else{
        derivatives[k_idx +j] = new Image<float>(name);}
      inp.close();
    }
    k_idx += (k+1);
  }
}


double SteerableFilter2DHermiteNotNorm::response(double theta, int x, int y)
{
  if(theta!= this->theta){
    calculate_steering_coefficients(theta);
    this->theta = theta;

  }
  // printf("Response to %i %i\n", x, y);
  double ret = 0;
  int k_idx = 0;
  for(int k = 1; k <=M; k++){
    for(int j = 0; j <= k; j++){
      ret = ret +
        gsl_vector_get(b_theta, k_idx + j)*
        derivatives[k_idx+j]->at(x,y)*
        sqrt(pow(2,k)*(factorial_n(k-j)*factorial_n(j)));
//       printf(" times %s \n",
//              gsl_vector_get(b_theta, k_idx + j),
//              derivatives[k_idx+j]->name.c_str());
    }
    k_idx += (k+1);
  }
  return ret;
}
