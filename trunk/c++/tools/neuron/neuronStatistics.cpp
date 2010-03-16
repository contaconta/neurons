
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
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include "Neuron.h"

using namespace std;

int main(int argc, char** argv)
{

  if(argc!=2){
    printf("neuronStatistics neuron.asc\n");
    exit(0);
  }

  Neuron* neuronita = new Neuron(argv[1]);
  neuronita->printStatistics();

  return 0;
}
