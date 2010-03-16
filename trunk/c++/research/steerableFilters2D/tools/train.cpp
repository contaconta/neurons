/////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or       //
// modify it under the terms of the GNU General Public License         //
// version 2 as published by the Free Software Foundation.             //
//                                                                     //
// This program is distributed in the hope that it will be useful, but //
// WITHOUT ANY WARRANTY; without even the implied warranty of          //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   //
// General Public License for more details.                            //
//                                                                     //
// Written and (C) by German Gonzalez                                  //
// Contact <ggonzale@atenea> for comments & bug reports                //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "highgui.h"
#include "Image.h"
#include "polynomial.h"
#include "SteerableFilter2D.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

using namespace std;

typedef struct
{
  int x;
  int y;
  float scale;
  float theta; // in Radians!
  int type;
} Point;


SteerableFilter2D* stf;
double old_theta = 0;
float sigma = 0;

vector< Point > points;

void save_gsl_matrix(gsl_matrix* m,  string filename){
  std::ofstream out(filename.c_str());
  for(int i = 0; i<m->size1; i++){
    for(int j = 0; j < m->size2; j++){
      out << gsl_matrix_get(m,i,j) << " ";
    }
    out <<std::endl;
  }
  out.close();
}

void save_gsl_vector(gsl_vector* v, string filename){
  FILE* f = fopen(filename.c_str(),"w");
  gsl_vector_fprintf(f, v, "%f");
  fclose(f);
}

void output_points_in_alpha_space(string filename){

  std::ofstream out(filename.c_str());
  gsl_vector* coords = gsl_vector_alloc(stf->alpha->size);
  gsl_vector* derivs = gsl_vector_alloc(stf->alpha->size);
  for(int nP = 0; nP < points.size(); nP++){
    gsl_matrix* rot = stf->get_rotation_matrix(points[nP].theta);
    for(int i = 0; i < stf->derivatives.size(); i++)
      gsl_vector_set(derivs, i, stf->derivatives[i]->at(points[nP].x, points[nP].y));

    gsl_blas_dgemv(CblasTrans, 1.0, rot, derivs, 0, coords);

    for(int i = 0; i < coords->size; i++)
      out << gsl_vector_get(coords, i) << " " ;
    out << std::endl;
  }
  out.close();
}

void load_points(string points_name, float convert_to_radians = true){
  std::ifstream points_in(points_name.c_str());
  string line;
  while(getline(points_in, line))
    {
      stringstream ss(line);
      double d;
      Point p;
      ss >> d;
      p.x = (int)d;
      ss >> d;
      p.y = (int)d;
      ss >> d;
      p.scale = (float)d;
      ss >> d;
      p.theta = (float)d;
      ss >> d;
      p.type = (int)d;
      if(convert_to_radians)
        p.theta = p.theta*3.1416/180;
      points.push_back(p);
//       printf("%i %i %f %f %i\n", p.x, p.y, p.scale, p.theta, p.type);
    }
  points_in.close();
}

void load_points_from_image(string img_mask, string img_orientation){

  //Pseudo random number generator
  gsl_rng* rng;
  const gsl_rng_type * T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  rng = gsl_rng_alloc (T);


  points.resize(0);
  int nPNeg = 0;
  int nPPos = 0;
  int x,y;

  IplImage* mask = cvLoadImage(img_mask.c_str(),0);
  Image<float>* orientation = new Image<float>(img_orientation);

  int mask_val= 0;
  //Loads the positive points
//   for(int y = mask->height-1; y >0; y--){
  while(nPPos < 1000){
    y = (int)(gsl_rng_uniform(rng)*mask->height);
    for(int x = 0; x < mask->width; x++){
      mask_val = ((uchar *)(mask->imageData + y*mask->widthStep))[x];
      //It is a positive point
      if(mask_val < 100){
        Point p;
        p.x = x;
        p.y = y;
        p.scale = 1;
        p.theta = orientation->at(x,y);
        p.type = 1;
        points.push_back(p);
        nPPos++;
      }
    }
//     if(nPPos > 1000)
//       break;
  }

  //Generates the same number of negative samples by sampling from the image at random.
  nPPos = points.size();
  while(nPNeg < nPPos){
    x = (int)(gsl_rng_uniform(rng)*mask->width);
    y = (int)(gsl_rng_uniform(rng)*mask->height);
    if ( ((uchar *)(mask->imageData + y*mask->widthStep))[x] < 100)
      continue;
    Point p;
    p.x = x;
    p.y = y;
    p.theta = gsl_rng_uniform(rng)*3.1416;
    p.scale = 1;
    p.type = -1;
    points.push_back(p);
    nPNeg++;
  }
}



void output_points(string points_name){
  std::ofstream points_out(points_name.c_str());
  char line[2048];
  for(int i = 0; i < points.size(); i++)
    {
      points_out << points[i].x << " " << points[i].y << " " << points[i].scale << " " 
                 << points[i].theta*180/M_PI << " " << points[i].type << std::endl;
    }
  points_out.close();
}


void output_response(int x, int y){
  double r;
  for(int i = 0; i < 360; i+= 5){
    r = stf->response(i*3.1416/180, x, y);
    printf("%i %f\n", i, r);
  }
}

void output_response_all(){
  double r;
  for(int i = 0; i < 360; i+= 5){
    printf("%i ", i);
    for(int j = 0; j < points.size(); j++){
      printf("%f ",stf->response(i*3.1416/180, points[j].x, points[j].y));
    }
    printf("\n");
  }
}

gsl_matrix* get_energy_matrix()
{
  gsl_matrix* energy = gsl_matrix_alloc((stf->M+3)*stf->M/2,
                                        (stf->M+3)*stf->M/2);
  gsl_matrix_set_all(energy, 0);
  int k_idx = 0;
  for(int k = 1; k <= stf->M; k++){
    for(int j = 0; j <= k; j++){
      int k2_idx = 0;
      for(int k2 = 1; k2 <= stf->M; k2++){
        for(int j2 = 0; j2 <= k2; j2++){
          double en = 0;
          vector<float> mask0_x = Mask::gaussian_mask(k-j,sigma,false);
          vector<float> mask0_y = Mask::gaussian_mask(j,sigma,false);
          vector<float> mask1_x = Mask::gaussian_mask(k2-j2,sigma,false);
          vector<float> mask1_y = Mask::gaussian_mask(j2,sigma,false);

          for(int l = 0; l < mask0_y.size(); l++){
            for(int m = 0; m < mask0_x.size(); m++){
              en = en + mask0_y[l]*mask0_x[m]*mask1_y[l]*mask1_x[m];
            }
          }
          gsl_matrix_set(energy, k_idx+j, k2_idx+j2, en);
        }
        k2_idx += (k2+1);
      }
    }
    k_idx += (k+1);
  }
  return energy;
}

gsl_matrix* invert_matrix(gsl_matrix* orig){
  gsl_matrix* result;
  gsl_matrix* LU;
  if(orig->size1 != orig->size2){
    printf("invert_matrix: error: matrix not square\n");
    exit(0);
  }
  LU = gsl_matrix_alloc(orig->size1, orig->size2);
  result = gsl_matrix_alloc(orig->size1, orig->size2);
  gsl_matrix_memcpy(LU, orig);
  gsl_permutation* p = gsl_permutation_alloc(orig->size1);
  int signum = 0;
  gsl_linalg_LU_decomp(LU, p, &signum);
  gsl_linalg_LU_invert (LU, p, result);
  gsl_matrix_free(LU);
  gsl_permutation_free(p);
  return result;
}

gsl_vector* optimize_coefficients_with_energy()
{
  gsl_vector* coefficients = gsl_vector_alloc((stf->M+3)*stf->M/2);
  gsl_vector* w_coefficients = gsl_vector_alloc((stf->M+3)*stf->M/2);
  gsl_vector* v_t = gsl_vector_alloc((stf->M+3)*stf->M/2);
  gsl_vector_set_all(coefficients, 0);
  gsl_vector_set_all(w_coefficients, 0);
  gsl_matrix* energy = get_energy_matrix();
  gsl_matrix_set(energy, 0, 0, 0);
  gsl_matrix* en_inv = invert_matrix(energy);

  //First calculate the weights w_i
  int k_idx = 0;
  for(int k = 1; k <= stf->M; k++){
    for(int i = 0; i <= k; i++){
      double w_points = 0;
      for(int n = 0; n < points.size(); n++){
        double p_contribution = 0;
        for(int j = 0; j <= k; j++){
          double b = 0;
          for(int l = 0; l <= k-i; l++){
            for(int m = 0; m <= i; m++){
              if(!(k-(l+m)==j)) continue;
              b += combinatorial(k-i,l)*
                combinatorial(i,m)*pow(-1.0,m)*
                pow(cos(points[n].theta),i+l-m)*
                pow(sin(points[n].theta), k+m-i-l);
            }//m
          }//l
          p_contribution += b*
            stf->derivatives[k_idx+j]->at(points[n].x,points[n].y);
        }//j
        p_contribution *= points[n].type;
        w_points += p_contribution;
      }//n
      gsl_vector_set(w_coefficients, k_idx+i, w_points);
    }//i
    k_idx += (k+1);
  }//k

//   gsl_vector_fprintf(stdout, w_coefficients, "%03.3f");

  gsl_blas_dgemv(CblasNoTrans, 1.0, en_inv, w_coefficients, 0, coefficients);
  //Calculates the denominator

  gsl_blas_dgemv(CblasTrans, 1.0, en_inv, w_coefficients, 0, v_t);
  double den = 0;
  gsl_blas_ddot(v_t, w_coefficients, &den);
  gsl_blas_dscal (1.0/den, coefficients);

  double function = 0;
  gsl_blas_ddot(w_coefficients, coefficients, &function);
  printf("%f\n", function);

  gsl_vector_free(v_t);
  gsl_vector_free(w_coefficients);
  gsl_matrix_free(en_inv);
  gsl_matrix_free(energy);

  return coefficients;
}

void load_all_training_set(string name)
{

  gsl_rng* rng;
  const gsl_rng_type * T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  rng = gsl_rng_alloc (T);

  int nPP = 0;
  int nPN = 0;
  int tPP = 0;
  int tPN = 0;

  IplImage* training = cvLoadImage(name.c_str());
  int r,g,b;

  //Counts the number of positive and negative training samples
  for(int x = 0; x < training->width; x++){
    for(int y = 0; y < training->height; y++){
      b =  ((uchar *)(training->imageData + y*training->widthStep))[x*3];
      g =  ((uchar *)(training->imageData + y*training->widthStep))[x*3+1];
      r =  ((uchar *)(training->imageData + y*training->widthStep))[x*3+2];
      if( ((r<100)&&(b<100)&&(g<100)) ||
          ((r<100)&&(b>100)&&(g<100)) )
        nPP++;
      if( ((r>100)&&(b<100)&&(g<100)) )
        nPN++;
    }
  }

  for(int x = 0; x < training->width; x++){
    for(int y = 0; y < training->height; y++){
      b =  ((uchar *)(training->imageData + y*training->widthStep))[x*3];
      g =  ((uchar *)(training->imageData + y*training->widthStep))[x*3+1];
      r =  ((uchar *)(training->imageData + y*training->widthStep))[x*3+2];
//       printf("%i %i %i\n", r, g, b);
      if( (r < 100) && (g < 100) && (b < 100)){
        for(int i = 0; i < nPN/(nPP); i++){
          Point p;
          p.x = x;
          p.y = y;
          p.scale = 3;
          p.theta = 0;
          p.type = 1;
          points.push_back(p);
          tPP++;
        }
      }
      if( (r < 100) && (g < 100) && (b > 100)){
        for(int i = 0; i < nPN/(nPP); i++){
          Point p;
          p.x = x;
          p.y = y;
          p.scale = 3;
          p.theta = 3.1316/2;
          p.type = 1;
          points.push_back(p);
          tPP++;
        }
      }
      if( (r > 100) && (g < 100) && (b < 100)){
        Point p;
        p.x = x;
        p.y = y;
        p.scale = 3;
        p.theta = gsl_rng_uniform(rng)*3.1416;
        p.type = -1;
        points.push_back(p);
        tPN++;
      }
    }
  }
  printf("Number of training points positives = %i, negatives = %i\n",tPP, tPN);
}

void output_training_set(){
  for(int i = 0; i < points.size(); i++)
    printf("%i %i %f %f %i\n", points[i].x, points[i].y, points[i].scale,
           points[i].theta, points[i].type);
}


int main(int argc, char **argv) {


  if( (argc!=7)){
    printf("Usage: train <image> <order> <sigma> <points> <matrix_out> <lds=1,normal=0>\n");
    exit(0);
  }


  if(atoi(argv[6]) > 0.5){
    stf = new SteerableFilter2D(argv[1], atoi(argv[2]) , atof(argv[3]));
    string points_f = argv[4];
    if(points_f.find("txt")!=string::npos)
      load_points(argv[4]);
    else
      load_all_training_set(argv[4]);
    output_points("points.txt");
    output_points_in_alpha_space("alpha_coords.txt");
    exit(0);
    // //Here for debugging the matrices
//     gsl_matrix* rotation = stf->get_rotation_matrix(M_PI/3);
//     save_gsl_matrix(rotation, "rotation.txt");
//     stf->calculate_steering_coefficients(M_PI/3);
//     save_gsl_vector(stf->b_theta, "b_theta.txt");
//     gsl_blas_dgemv(CblasNoTrans, 1.0, rotation, stf->alpha, 0, stf->b_theta);
//     save_gsl_vector(stf->b_theta, "b_theta2.txt");
  }
  else{
    sigma = atof(argv[3]);
    stf = new SteerableFilter2D(argv[1], atoi(argv[2]), sigma);
    string points_f = argv[4];
    if(points_f.find("txt")!=string::npos)
      load_points(argv[4]);
    else
      load_all_training_set(argv[4]);
    gsl_vector* coeffs = optimize_coefficients_with_energy();
    save_gsl_vector(coeffs, argv[5]);
  }
}



/*** CODE FOR THE GRADIENT DESCENT. NOT IN USE.

double get_e()
{
  double e = 0;
  for(int i = 0; i < points.size(); i++)
    {
      e += stf->response(points[i].theta, points[i].x, points[i].y)*
        points[i].type;
//       printf("%f\n", stf->response(points[i].theta, points[i].x, points[i].y)*
//              points[i].type);
    }
  return -e;
}

double get_e_regularized(double lambda){
  double e = 0;
  for(int i = 0; i < points.size(); i++)
    {
      e += stf->response(points[i].theta, points[i].x, points[i].y)*
        points[i].type;
//       printf("%f\n", stf->response(points[i].theta, points[i].x, points[i].y)*
//              points[i].type);
    }

    double mod_a = 0;
    for(int k = 1; k <=stf->M; k++){
      for(int j = 0; j <= k; j++){
        mod_a +=
          gsl_matrix_get(stf->alpha,k-j,j)*
          gsl_matrix_get(stf->alpha,k-j,j);
      }
    }

  return -e + lambda*mod_a;

}



void gradient_descent()
{
  // First will be to calculate the gradient as finnite differences. I will increase each parameter with 0.01 and calculate the error there.
  gsl_matrix* a_save   = gsl_matrix_alloc(stf->M+1, stf->M+1);
  gsl_matrix* gradient = gsl_matrix_alloc(stf->M+1, stf->M+1);
  gsl_matrix_set_all(gradient, 0);
  gsl_matrix_memcpy(a_save, stf->alpha);

  double lambda = 1;

  double e_init = get_e_regularized(lambda);
  double e_new = 0;

  double step_size = 0.01;

  //The gradient descent starts here

  for(int t = 1; t < 1000; t++){

    e_init = get_e_regularized(lambda);
    e_new = 0;

    //Calculates the gradient
    for(int k = 1; k <=stf->M; k++){
      for(int j = 0; j <= k; j++){
        gsl_matrix_set(stf->alpha, k-j, j,
                       gsl_matrix_get(a_save, k-j,j) + 0.01);
        e_new = get_e_regularized(lambda);
        gsl_matrix_set(gradient,  k-j, j, (e_new - e_init)/0.01);
        gsl_matrix_set(stf->alpha, k-j, j,
                       gsl_matrix_get(a_save, k-j,j));
      }
    }

    // Moves the matrix to the new state
    for(int k = 1; k <=stf->M; k++){
      for(int j = 0; j <= k; j++){
        gsl_matrix_set(stf->alpha, k-j, j,
                       gsl_matrix_get(a_save, k-j,j) -
                       step_size*gsl_matrix_get(gradient, k-j,j));
      }
    }

    double grad_mod = 0;
    for(int i = 0; i < stf->M+1; i++)
      for(int j = 0; j < stf->M+1; j++)
        grad_mod += gsl_matrix_get(gradient, i, j)*
          gsl_matrix_get(gradient, i, j);

    double mod_a = 0;
    for(int k = 1; k <=stf->M; k++){
      for(int j = 0; j <= k; j++){
        mod_a +=
          gsl_matrix_get(stf->alpha,k-j,j)*
          gsl_matrix_get(stf->alpha,k-j,j);
      }
    }

    e_new = get_e_regularized(lambda);

    printf("%i %f %f %f\n",t,e_new, grad_mod, mod_a);
    if( (e_new > e_init) || (fabs(e_new - e_init) < 1e-3) ) {
//       save_gsl_matrix(a_save,stf->M+1,"matrix_before_collapse.txt");
//       save_gsl_matrix(stf->alpha,stf->M+1,"matrix_after_collapse.txt");
//       save_gsl_matrix(gradient,stf->M+1,"matrix_gradient_collapse.txt");
      break;
    }
    //Saves the matrix
    gsl_matrix_memcpy(a_save, stf->alpha);
  }
  save_gsl_matrix(stf->alpha, stf->M+1, "matrix_out.txt");
//   gsl_matrix_fprintf (stdout, stf->alpha, "%0.03f");
}

gsl_matrix* optimize_coefficients()
{
  gsl_matrix* coefficients = gsl_matrix_alloc(stf->M+1, stf->M+1);
  gsl_matrix_set_all(coefficients, 0);
  for(int k = 1; k <= stf->M; k++){
//   for(int k = 8; k <= 8; k++){
    if(k%2!=0) continue;
    for(int i = 0; i <= k; i++){
      if(i%2!=0) continue;
      double w_points = 0;
      for(int n = 0; n < points.size(); n++){
        double p_contribution = 0;
        for(int j = 0; j <= k; j++){
          double b = 0;
          for(int l = 0; l <= k-i; l++){
            for(int m = 0; m <= i; m++){
              if(!(k-(l+m)==j)) continue;
              b += stf->combinatorial(k-i,l)*
                stf->combinatorial(i,m)*pow(-1.0,m)*
                pow(cos(points[n].theta),i+l-m)*
                pow(sin(points[n].theta), k+m-i-l);
            }//m
          }//l
          p_contribution += b*
            stf->derivatives[k-j][j]->at(points[n].x,points[n].y);
        }//j
        p_contribution *= points[n].type;
        w_points += p_contribution;
      }//n
      gsl_matrix_set(coefficients, k-i, i,
                     gsl_matrix_get(coefficients, k-i, i)+
                     w_points);
    }//i
  }//k

  save_gsl_matrix(coefficients, stf->M+1, "w_coefficients_n_e.txt");


  double norm = 0;
  for(int k = 0; k <= stf->M; k++){
    for(int i = 0; i <= k; i++){
      norm += gsl_matrix_get(coefficients, k-i, i)*
        gsl_matrix_get(coefficients, k-i, i);
    }
  }
  norm = sqrt(norm);
//   printf("Norm = %g\n", norm);

  for(int k = 0; k <= stf->M; k++){
    for(int i = 0; i <= k; i++){
      gsl_matrix_set(coefficients, k-i,i,
                     gsl_matrix_get(coefficients, k-i, i)/norm);
    }
  }
//   gsl_matrix_fprintf(stdout, coefficients, "%0.03f");
  return coefficients;
}



*/
