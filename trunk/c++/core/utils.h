
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


#ifndef _STUPID_UTILS_H
#define _STUPID_UTILS_H 

#ifdef WITH_GLEW
  #include <GL/glew.h>
#endif
#include <GL/glut.h>


#include <string>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdarg>
// #include "neseg.h"

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <string>
#include <errno.h>

#include "Mask.h"

using namespace std;

bool fileExists(string filename);

string getDirectoryFromPath(string path);

string getNameFromPath(string path);

string getNameFromPathWithoutExtension(string path);

string getExtension(string path);

int get_files_in_dir(string dir, vector<string> &files);

string getDerivativeName(int order_x, int order_y, int order_z,
                         float sigma_x, float sigma_y, float sigma_z,
                         string directory = "");

//returns the file size in bytes
int   getFileSize(string path);

vector< vector< double > > loadMatrix(string filename);

vector< vector< double > > loadMatrix(istream &in);

void saveMatrix(vector< vector< double > > & matrix, string filename);

int factorial_n(int n);

/** Returns the double factorial of an int. That is, the product of n*(n-2)*(n-4).... by definition, dfactorial(0) = dfactorial(-1) = dfactorial(1) = 1.*/
int dfactorial_n(int n);

/** Returns the combinatorial of two integers.*/
double combinatorial(int over, int under);

bool isNumber(string s);

/** Saves a vector of doubles to a file*/
void saveVectorDouble(vector< double >& vc, string filename);

vector< double > readVectorDouble(string filename);

void saveFloatVector( vector< float >& vc, string filename);

void renderString(const char* format, ...);

void secondStatistics(vector< double > data, double* mean, double* variance);

// This class will be used to plot vectors using matlab from the C++ code.

class MATLABDRAW
{
public:

  static void runMatlabSlave() {

    // This is an awful hack to see if matlabSlave is running
    int err = system("ps -ef | grep matlabSlave | wc > /tmp/matlabRunning");
    std::ifstream in("/tmp/matlabRunning");
    int isMatRun = 0;
    in >> isMatRun;
    in.close();
    //Starts the listener
    if(isMatRun < 2){
      remove("/tmp/matlab");
      mkfifo("/tmp/matlab",0666);
      int error = system("gnome-terminal -e matlabSlave");
      usleep(1000000);
    }
  }

  static void createFigure() {
    runMatlabSlave();
    FILE* stream = fopen("/tmp/matlab", "w");
    fprintf (stream, "figure\n");
    fclose(stream);
  }

  static void sendCommand(string command) {
    FILE* stream = fopen("/tmp/matlab", "w");
    fputs (command.c_str(), stream);
    fputs ("\n", stream);
    fflush(stream);
    fclose(stream);
    // To give time to the server to open and close the pipe
    usleep(1000);
  }

  template <class T>
  static void drawVector(vector< T > vc) {
    runMatlabSlave();
    std::ofstream out("/tmp/matlabVector");
    for(int i = 0; i < vc.size(); i++)
      out << vc[i] << std::endl;
    out.close();
    // FILE* stream = fopen("/tmp/matlab", "w");
    sendCommand ("load(\'/tmp/matlabVector\')");
    sendCommand ("plot(matlabVector)");
    // fclose(stream);
  }
};




#endif //STUPID_UTILS_H
