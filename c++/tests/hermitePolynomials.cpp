
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

#include <iostream>
#include <fstream>
// #include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include "Mask.h"
#include "Cube.h"
using namespace std;




int main(int argc, char **argv) {

  // Coefficients according to wikipedia.
  vector< string > wiki;
  wiki.push_back("1.000x^0");
  wiki.push_back("0.000x^0  2.000x^1");
  wiki.push_back("-2.000x^0  0.000x^1  4.000x^2");
  for(int i = 0; i < 8; i++){
    if(i < 3)
      std::cout << "Wiki:  " << wiki[i] << std::endl;
    else
      std::cout << "Wiki: Check http://en.wikipedia.org/wiki/Hermite_polynomials" << std::endl;
    std::cout << "Ours: ";
    Polynomial* p = Mask::hermite_polynomial_p(i);
    p->print();
  }

  std::cout << "Checking for values at certain points for the second order polynomial\n";
  Polynomial* p = Mask::hermite_polynomial_p(2);
  std::cout << "X   Expected   Obtained\n";
  std::cout << "0     -2        " << p->evaluate(0) << std::endl;
  std::cout << "1      2        " << p->evaluate(1) << std::endl;
  std::cout << "-1     2        " << p->evaluate(-1) << std::endl;
  std::cout << "0.5   -1        " << p->evaluate(0.5) << std::endl;
  std::cout << "-0.5  -1        " << p->evaluate(-0.5) << std::endl;
}
