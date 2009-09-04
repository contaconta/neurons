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
// Written and (C) by Aurelien Lucchi and Kevin Smith                  //
// Contact aurelien.lucchi (at) gmail.com or kevin.smith (at) epfl.ch  // 
// for comments & bug reports                                          //
/////////////////////////////////////////////////////////////////////////

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

using namespace std;

void store_weak_learners(char* learner_type, int index,
                         const char* data, int data_size);

string getExtension(string path);

string getNameFromPathWithoutExtension(string path);

int get_files_in_dir(string dir, vector<string> &files,string extension="");

#endif //UTILS_H
