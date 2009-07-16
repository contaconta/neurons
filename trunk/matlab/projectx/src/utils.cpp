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

#include <iostream>
#include <sstream>
#include <fstream>
#include "utils.h"
#include <dirent.h>
#include <errno.h>

using namespace std;

void store_weak_learners(char* learner_type, int index,
                         const char* data, int data_size)
{
    stringstream out;
    out << learner_type << "_" << index;
    cout << "Store" << out.str() << endl;
    ofstream outputFile(out.str().c_str(),ios::out);
    outputFile.write(data,data_size);
    outputFile.close();
}

string getExtension(string path){
  return path.substr(path.find_last_of(".")+1);
}

int get_files_in_dir(string dir, vector<string> &files,string extension)
{
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
    cout << "Error(" << errno << ") opening " << dir << endl;
    return errno;
  }

  while ((dirp = readdir(dp)) != NULL) {
    if(extension != "")
      {
        if(getExtension(dirp->d_name)==extension)
          files.push_back(string(dirp->d_name));
      }
    else
      files.push_back(string(dirp->d_name));
  }
  closedir(dp);
  return 0;
}
