#ifndef RANDOM_H
#define RANDOM_H

#include "config.h"

#include <algorithm>
#include <vector>
#include <stdio.h>
#include <assert.h>

class Random {
private:
  
  double ran01(long *idum);

public:
  Random(const int& arg) : seed(arg) {}
  long int seed;
  double next() { return ran01(&seed);}
  /*    
      FUNCTION:      generates a random vector, quick and dirty
      INPUT:         vector dimension
      OUTPUT:        returns pointer to vector, 
                     free memory after using the vector
      (SIDE)EFFECTS: none
  */
  long int* generate_array(const int & size);
  /*
   * alternative
   */
  vector<long int> generate_random_vector(const int& size ) {
    vector<long int> V(size);
    generate(V.begin(), V.end(), rand);
    return V;
  }

};
#endif
