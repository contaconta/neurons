
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
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "CubeLiveWire.h"
#include "CubeFactory.h"

using namespace std;

int main(int argc, char **argv) {
  Cube<float, double>* cubeAguet = new Cube<float, double>
    ("/media/neurons/cut2/aguet_4.00_2.00.nfo");
  Cube<float, double>* cubeAguetTheta = new Cube<float, double>
    ("/media/neurons/cut2/aguet_4.00_2.00_theta.nfo");
  Cube<float, double>* cubeAguetPhi = new Cube<float, double>
    ("/media/neurons/cut2/aguet_4.00_2.00_phi.nfo");
  Cube<uchar, ulong>* cube = new Cube<uchar, ulong>("/media/neurons/cut2/cut2.nfo");

  DistanceDijkstraColorInverse* djkc = new DistanceDijkstraColorInverse
    (cube);

  CubeLiveWire* cubeLiveWire = new CubeLiveWire(cube, djkc);

  cubeLiveWire->computeDistances(87,39,11);
  Cloud<Point3D>* cd = cubeLiveWire->findShortestPath(87,39,11, 118,81,9);
  cd->saveToFile("cloud.cl");

}
