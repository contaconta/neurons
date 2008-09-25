
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
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <stdio.h>
#include <vector>

using namespace std;

int nsamples = 1000000;
int nAccess  = 500;
vector< double > a;

void at1(vector< double > p){
  vector< double > v = p;
  // printf("v at 1000 = %f\n", v[nsamples -1]);
}

void at2(vector< double > &p){
  vector< double > v = p;
  // printf("v at 1000 = %f\n", v[nsamples -1]);
}

void at3(vector< double >* p){
  vector< double >* v = p;
  // printf("v at 1000 = %f\n", (*v)[nsamples -1]);
}


void fat1(){
 for(int i = 0; i < nAccess; i++)
    at1(a);
}

void fat2(){
 for(int i = 0; i < nAccess; i++)
    at2(a);
}

void fat3(){
 for(int i = 0; i < nAccess; i++)
    at3(&a);
}


int main(int argc, char **argv) {

  printf("Creating the vector\n");
  a.resize(nsamples);
  for(int i = 0; i < nsamples; i++){
    a[i] = i;
  }

  printf("accessing the vector with method 1\n");
  fat1();
  printf("accessing the vector with method 2\n");
  fat2();
  printf("accessing the vector with method 3\n");
  fat3();
  printf("done\n");

  // printf ("a1 =  %.10lf seconds.\n", dif );



}
