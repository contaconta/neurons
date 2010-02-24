#include "SteerableFilter2D.h"

SteerableFilter2D::SteerableFilter2D(string filename_image, string filename_parameters, float sigma)
{
  includeOddOrders = true;
  includeEvenOrders = true;

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
  calculate_derivatives();
}

SteerableFilter2D::SteerableFilter2D(string filename_image, int M, float sigma,  bool _includeOddOrders, bool _includeEvenOrders)
{
  includeEvenOrders = _includeEvenOrders;
  includeOddOrders = _includeOddOrders;

  printf("Called this constructor includeOddOrders = %i, includeEvenOrders = %i\n",
         includeOddOrders, includeEvenOrders);

  directory = filename_image.substr(0,filename_image.find_last_of("/\\")+1);
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
  calculate_derivatives();
}



void SteerableFilter2D::calculate_derivatives()
{
  printf("Here!!!! includeOddOrders = %i, includeEvenOrders = %i\n",
         includeOddOrders, includeEvenOrders);
  ifstream inp;
  char buff[512];
  sprintf(buff, "_%02.02f.jpg",sigma);
  string name_b = directory + "g" + buff;

  int k_idx = 0;
  for(int k = 1; k <= M; k++){
    if((k%2==0) && (!includeEvenOrders)){
      continue;
    }
    if((k%2==1) && (!includeOddOrders)){
      continue;
    }
    for(int j = 0; j <= k; j++){
      string name = directory + "g_";
      for(int l = 0; l < k-j; l++)
        name = name + "x";
      for(int l = 0; l < j; l++)
        name = name + "y";
      name = name + buff;
      //Check for the existance of the file
      inp.open(name.c_str(), ifstream::in);
      if(inp.fail()){
        derivatives[k_idx + j] = image->calculate_derivative(k-j,j,sigma, name);
      }else{
        derivatives[k_idx +j] = new Image<float>(name);}
      inp.close();
//       printf("%s\n", name.c_str());
    }
    k_idx += (k+1);
  }
}


double SteerableFilter2D::response(double theta, int x, int y)
{
  if(theta!= this->theta){
    calculate_steering_coefficients(theta);
    this->theta = theta;

  }
  double ret = 0;
  int k_idx = 0;
  for(int k = 1; k <=M; k++){
    for(int j = 0; j <= k; j++){
      ret = ret +
        gsl_vector_get(b_theta, k_idx + j)*
        derivatives[k_idx+j]->at(x,y);
//       printf(" times %s \n",
//              gsl_vector_get(b_theta, k_idx + j),
//              derivatives[k_idx+j]->name.c_str());
    }
    k_idx += (k+1);
  }
  return ret;
}

void SteerableFilter2D::filter(double theta)
{
  for(int x = 0; x < image->width; x++){
    for(int y = 0; y < image->height; y++){
      result->put(x,y,response(theta,x,y));
    }
  }
  result->save();
}

void SteerableFilter2D::load_alphas(string filename)
{
  FILE* f = fopen(filename.c_str(), "r");

  //This is a hack to automatically know the dimension of the vector
  char number[1024];
  int nNumbers = 0;
  while(fgets(number, 1024, f) != NULL){
    nNumbers++;
  }
  fclose(f);
//   nNumbers--;

  //Gets the order of the derivatives
  for(M = 0; M < 10; M++){
//     printf("M = %i, %i %i\n", M, (M+3)*M/2, nNumbers);
    if(nNumbers == (M+3)*M/2)
      break;
  }

  f = fopen(filename.c_str(), "r");

  alpha = gsl_vector_alloc(nNumbers);
  int err = gsl_vector_fscanf(f, alpha);
  if(err == GSL_EFAILED){
    printf("Error reading the vectorx in %s\n", filename.c_str());
    exit(0);
  }

//   printf("Alpha:\n");
//   gsl_vector_fprintf (stdout, alpha, "%0.03f");
}



void SteerableFilter2D::calculate_steering_coefficients(double theta)
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
//       printf("k=%i j=%i\n", k, j);
      double b_kj = 0;
      for(int i = 0; i <= k; i++){
        double sum = 0;
        for(int m = 0; m <= i; m++){
          for(int l = 0; l <= k-i; l++){
//             printf("  i=%i l=%i m=%i: ",i,l,m);
            if( k-(l+m) != j){
//               printf("continue \n");
              continue;}
//             printf(" s=%f ", sum);
            double prod = 1;
            prod *= combinatorial(k-i,l);
            prod *= combinatorial(i,m);
            prod *= pow(-1.0,m);
            prod *= pow(ct,i+l-m);
            prod *= pow(st,k-i-l+m);

            sum = sum + prod;
//             printf("  c1=%f c2=%f p1=%f p2=%f p3=%f p=%f s=s+p=%f\n",
//                    combinatorial(k-i,l), combinatorial(i,m),
//                    pow(-1.0,m),pow(ct,i+l-m),pow(st,k-i-l+m),prod,sum);
          }
        }
//         printf("b_%i%i = %f + %f*%f\n", k-j,j,b_kj,gsl_matrix_get(alpha,k-j,j),sum );
        b_kj += gsl_vector_get(alpha,k_idx + i)*sum;
//         b_kj += gsl_matrix_get(alpha,k-i,i)*sum*pow(3.0,k-1);
      }
//       printf("b_%i%i = %f\n", k-j,j,b_kj);
      gsl_vector_set(b_theta,k_idx+j,b_kj);
    }
    k_idx += (k+1);
  }
//   printf("Coefficients:\n");
//   gsl_matrix_fprintf (stdout, b_theta, "%0.03f");
}


gsl_matrix* SteerableFilter2D::get_rotation_matrix(double angle){

  int m_size = (M+3)*M/2;
  gsl_matrix* toRet = gsl_matrix_alloc(m_size, m_size);
  gsl_matrix_set_all(toRet,0);
  int k_idx = 0;
  int j_idx = 0;
  double ct = cos(angle);
  double st = sin(angle);


  for(int k = 1; k <=M; k++){
    for(int j = 0; j <=k; j++){
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
        gsl_matrix_set(toRet, k_idx + j , j_idx + i, sum);
      }
    }
    k_idx = k_idx + k+1;
    j_idx = j_idx + k+1;
  }

  return toRet;


}



