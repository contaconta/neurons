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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Cube.h"
#include "Neuron.h"
#include "utils.h"
#include <gsl/gsl_multimin.h>

using namespace std;

Cube<float,double>* param;
Cloud<Point3Dot>* cl;
vector<double> t;
vector<double> xv;


double
my_f (const gsl_vector *v, void *params)
{
  double a = gsl_vector_get(v, 0);
  double b = gsl_vector_get(v, 1);
  double toReturn = 0;
  double ex = 0;

  for(int i = 0; i < xv.size(); i++){
    ex = a*xv[i] + b;
    toReturn +=
      (1-t[i])*ex +
      log(1+exp(-ex));
  }

  return toReturn;
}

void
my_df (const gsl_vector *v, void *params,
       gsl_vector *df)
{
  double a = gsl_vector_get(v, 0);
  double b = gsl_vector_get(v, 1);
  double ex = 0;
  double partial_a = 0;
  double partial_b = 0;

  for(int i = 0; i < xv.size(); i++){
    ex = exp(-(a*xv[i] + b));
    ex = 1-t[i] - ex/(1+ex);
    partial_a += xv[i]*ex;
    partial_b += ex;
  }

  gsl_vector_set(df, 0, partial_a);
  gsl_vector_set(df, 1, partial_b);
}

void
my_fdf (const gsl_vector *v, void *params,
        double *f, gsl_vector *df)
{
  double a = gsl_vector_get(v, 0);
  double b = gsl_vector_get(v, 1);
  double ex = 0;
  double ex2 = 0;
  double partial_a = 0;
  double partial_b = 0;
  double toReturn = 0;

  for(int i = 0; i < xv.size(); i++){
    ex = exp(-(a*xv[i] + b));
    ex2 = 1+ex;
    ex = 1-t[i] - ex/(1+ex);
    partial_a += xv[i]*ex;
    partial_b += ex;
    toReturn +=
      (1-t[i])*(a*xv[i] + b) +
      log(ex2);
  }

  gsl_vector_set(df, 0, partial_a);
  gsl_vector_set(df, 1, partial_b);
  *f = toReturn;
}

/*
double
my_f (const gsl_vector *v, void *params)
{
  double x, y;
  double *p = (double *)params;

  x = gsl_vector_get(v, 0);
  y = gsl_vector_get(v, 1);

  return p[2] * (x - p[0]) * (x - p[0]) +
    p[3] * (y - p[1]) * (y - p[1]) + p[4];
}
void
my_df (const gsl_vector *v, void *params,
       gsl_vector *df)
{
  double x, y;
  double *p = (double *)params;

  x = gsl_vector_get(v, 0);
  y = gsl_vector_get(v, 1);

  gsl_vector_set(df, 0, 2.0 * p[2] * (x - p[0]));
  gsl_vector_set(df, 1, 2.0 * p[3] * (y - p[1]));
}
void
my_fdf (const gsl_vector *x, void *params,
        double *f, gsl_vector *df)
{
  *f = my_f(x, params);
  my_df(x, params, df);
}
*/




int main(int argc, char **argv) {

  if(argc!=5){
    printf("Usage: cubeLogistic cubeFromWhichGetTheValues cloudWithPoints OriginalCube SigmoidCubeName\n");
    exit(0);
  }
  string cubeDataName(argv[1]);
  string cloudName   (argv[2]);
  string cubeName    (argv[3]);
  string outputName  (argv[4]);

  param  = new Cube<float,double>(cubeDataName);
//     ("/media/neurons/steerableFilters3D/resultVolumes/result-4-e-2-2-groundTruth-1-estimated.nfo");

  cl = new Cloud<Point3Dot>(cloudName);
//     ("/media/neurons/steerableFilters3D/test.cl");

  Cube<float,double>* orig  = new Cube<float,double>(cubeName);
//     ("/home/ggonzale/mount/cvlabfiler/n7_4/cut.nfo");

  Cube<float,double>* flts = orig->create_blank_cube(outputName);
//     ("cuts");


  // t.resize(cl->points.size());
  // xv.resize(cl->points.size());

  t.resize(5000);
  xv.resize(5000);

  vector<int>    idx(3);
  vector<float> mic(3);

  int nPp = 0;
  int nPn = 0;
  // for(int i = 0; i<cl->points.size(); i++){
  for(int i = 0; i<5000; i++){
    Point3Dot* pp = dynamic_cast<Point3Dot*>(cl->points[i]);
    if(pp->type == 1){
      t[i] = 1;
      nPp++;
    }
    else{
      t[i] = 0;
      nPn++;
    }
    mic[0] = cl->points[i]->coords[0];
    mic[1] = cl->points[i]->coords[1];
    mic[2] = cl->points[i]->coords[2];
    param->micrometersToIndexes(mic, idx);
    xv[i] = param->at(idx[0], idx[1], idx[2]);
  }

  printf("nPp=%i, nPn=%i\n",nPp, nPn);
  saveVectorDouble(xv, "data.txt");

  size_t iter = 0;
  int status = 0;

  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *s;

  /* Position of the minimum (1,2), scale factors
     10,20, height 30. */
  double par[5] = { 1.0, 2.0, 10.0, 20.0, 30.0 };

  gsl_vector *x;
  gsl_multimin_function_fdf my_func;

  my_func.n = 2;
  my_func.f = &my_f;
  my_func.df = &my_df;
  my_func.fdf = &my_fdf;
  my_func.params = &par;

  /* Starting point, x = (5,7) */
  x = gsl_vector_alloc (2);
  gsl_vector_set (x, 0, 1); //a
  gsl_vector_set (x, 1, 0.0); //b

  // T = gsl_multimin_fdfminimizer_conjugate_fr;
  T = gsl_multimin_fdfminimizer_conjugate_pr;
  // T = gsl_multimin_fdfminimizer_steepest_descent;
  s = gsl_multimin_fdfminimizer_alloc (T, 2);

  // Done to debug
  // gsl_vector* x2 = gsl_vector_alloc (2);
  // gsl_vector* x3 = gsl_vector_alloc (2);
  // double      val;
  // for(float a = 0.1; a < 1; a+=0.1){
    // gsl_vector_set (x, 0, a); //a
    // my_df(x, par, x3);
    // my_fdf(x, par, &val, x2);
    // printf("The function at a = %f is %f and %f\n",
           // a, my_f(x, par), val);
    // printf("  the gradients are: g_a = %f and g_a2 = %f\n",
           // gsl_vector_get(x2,0), gsl_vector_get(x3,0));
    // printf("  the gradients are: g_b = %f and g_b2 = %f\n",
           // gsl_vector_get(x2,1), gsl_vector_get(x3,1));
  // }
  // exit(0);

  gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.005, 1e-8);

  printf("#it   a          b          f          status\n"); fflush(stdout);
  do
    {
      printf ("%5d %010.5f %010.5f %010.5f %i\n", (int)iter,
              gsl_vector_get (s->x, 0),
              gsl_vector_get (s->x, 1),
              s->f, status);

      iter++;
      status = gsl_multimin_fdfminimizer_iterate (s);

      if (status){
        printf ("%5d %010.5f %010.5f %010.5f %i\n", (int)iter,
                gsl_vector_get (s->x, 0),
                gsl_vector_get (s->x, 1),
                s->f, status);
        break;
      }

      status = gsl_multimin_test_gradient (s->gradient, 1e-3);

      if (status == GSL_SUCCESS){
        printf ("\nMinimum found at:\n");
        printf("   #it   a          b          f          status\n"); fflush(stdout);
        printf("   %5d %010.5f %010.5f %010.5f %i\n", (int)iter,
                gsl_vector_get (s->x, 0),
                gsl_vector_get (s->x, 1),
                s->f, status);
      }

    }
  while (status == GSL_CONTINUE && iter < 10000);

  double a = gsl_vector_get (s->x, 0);
  double b = gsl_vector_get (s->x, 1);

  printf("Copying the values  [");
  for(int z=0; z < orig->cubeDepth; z++){
    for(int y = 0; y < orig->cubeHeight; y++)
      for(int x = 0; x < orig->cubeWidth; x++)
        flts->put(x,y,z,1.0/(1+exp(-orig->at(x,y,z)*a -b)) );
    printf("#");fflush(stdout);
  }
  printf("]\n");

  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);


}
