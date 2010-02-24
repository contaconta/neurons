/***************************************************************************
                          Random.cc  -  description
                             -------------------
    begin                : Fri Nov 10 2000
    copyright            : (C) 2000 by Christian Blum
    email                : cblum@ulb.ac.be
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
#include "Random.h"
#include <stdio.h>
#define VERBOSE(x) x
#define VERYVERBOSE(x)

/* constants for a pseudo-random number generator, 
 for details see numerical recipes in C 
 */
static const int IA = 16807;
static const int IM = 2147483647;
static const double AM = (1.0/IM);
static const int IQ = 127773;
static const int IR = 2836;

/* pseudo-random number generator as proposed in numerical recipes in C 
   Input:   a long value; has to be the seed variable
   Output:  a pseudo-random number uniformly distributed in [0,1]
   Side effects: changes the value of the input variable, must be this way
*/
double Random::ran01( long *idum )
{
  long k;
  double ans;

  k =(*idum)/IQ;
	*idum = IA * (*idum - k * IQ) - IR * k;
  if (*idum < 0 ) *idum += IM;
	ans = AM * (*idum);
  return ans;
}

long int* Random::generate_array(const int& size) {
   long int  i, j, help;
   long int  *v;

   v = (long int*) malloc( size * sizeof(long int) );

   for ( i = 0 ; i < size; i++ ) 
     v[i] = i;

   for ( i = 0 ; i < size-1 ; i++) {
     j = (long int) ( ran01( &seed ) * (size - i)); 
     assert( i + j < size );
     help = v[i];
     v[i] = v[i+j];
     v[i+j] = help;
   }
   VERYVERBOSE ( printf("Random vector:\n");
   for (i = 0 ; i < size ; i++ ) 
     printf(" %ld ",v[i]);
   printf("\n"); )
   return v;
}
