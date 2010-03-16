

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
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Cube.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!= 3){
    printf("Usage: cubeCombineFromList.cpp list.txt out_name\n");
    exit(0);
  }

  string directory = getDirectoryFromPath(argv[1]);

  vector< Cube< uchar, ulong>* > cubes;
  vector< string > cubeNames;
  vector< int    > offsetX;
  vector< int    > offsetY;
  vector< int    > offsetZ;

  std::ifstream in (argv[1]);
  string s;
  int i = 0;
  while(getline(in, s)){
    stringstream ss(s);
    int X,Y,Z;
    string cubeName;
    ss >> cubeName;
    ss >> X;
    ss >> Y;
    ss >> Z;
    cubeNames.push_back(cubeName);
    offsetX.push_back(X);
    offsetY.push_back(Y);
    offsetZ.push_back(Z);
    std::cout << cubeNames[i] << " " << offsetX[i] << " " << offsetY[i]
              << " " << offsetZ[i] << std::endl;
    i++;
  }
  in.close();


  //Gets the min and max coordinates
  int Xmin = 1e6;
  int Xmax = 0;
  int Ymin = 1e6;
  int Ymax = 0;
  int Zmin = 1e6;
  int Zmax = 0;
  printf("The size of the vectors is [%i %i %i %i]\n",
         cubeNames.size(), offsetX.size(), offsetY.size(), offsetZ.size());
  for(int i = 0 ; i < cubeNames.size(); i++){
    std::cout << i << " " <<  cubeNames[i] << " " << offsetX[i] << " " << offsetY[i]
              << " " << offsetZ[i] << std::endl;
    if(offsetX[i] <= Xmin)
      Xmin = offsetX[i];
    if(offsetY[i] <= Ymin)
      Ymin = offsetY[i];
    if(offsetZ[i] <= Zmin)
      Zmin = offsetZ[i];
    cubes.push_back
      (new Cube<uchar, ulong>(directory + cubeNames[i]));
    if(Xmax <= offsetX[i] + cubes[i]->cubeWidth-1)
      Xmax = offsetX[i] + cubes[i]->cubeWidth-1;
    if(Ymax <= offsetY[i] + cubes[i]->cubeHeight-1)
      Ymax = offsetY[i] + cubes[i]->cubeHeight-1;
    if(Zmax <= offsetZ[i] + cubes[i]->cubeDepth-1)
      Zmax = offsetZ[i] + cubes[i]->cubeDepth-1;
  }


  printf("The cube limits are [%i,%i,%i]->[%i,%i,%i]\n",
         Xmin, Ymin, Zmin, Xmax, Ymax, Zmax);

  printf("The new cube should have a dimension of [%i,%i,%i]\n",
         Xmax-Xmin+1, Ymax-Ymin+1, Zmax-Zmin+1);

  // Xmin = 0; Ymin = 0; Zmin = 0;

  Cube<uchar, ulong>* dest = new Cube<uchar, ulong>
    (Xmax-Xmin+1, Ymax-Ymin+1, Zmax-Zmin+1, directory + argv[2]);

  for(int i = 0; i < cubes.size(); i++){
    printf("doing cube %i\n", i);
    for(int z = 0; z < cubes[i]->cubeDepth; z++)
      for(int y = 0; y < cubes[i]->cubeHeight; y++)
        for(int x = 0; x < cubes[i]->cubeWidth; x++)
          if( (x+offsetX[i]-Xmin >= 0) &&
              (y+offsetY[i]-Ymin >= 0) &&
              (z+offsetZ[i]-Zmin >= 0) ){
            dest->put(x+offsetX[i]-Xmin,
                      y+offsetY[i]-Ymin,
                      z+offsetZ[i]-Zmin,
                      cubes[i]->at(x,y,z));
          }
  }




}
