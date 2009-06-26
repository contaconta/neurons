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
#include <map>
#include <string.h>
#include <cstdlib>
#include <sstream>
#include "enumerateLearners.h"

#define DEBUG

struct sHA_params
{
  int step_size_x;
  int step_size_y;
};

vector<string> list_weak_learners;

int enumerate_learners(char *learner_type, int max_width, int max_height, char**& weak_learners)
{
  char temp[20];
  char* left_token = learner_type;
  char* right_token = 0;
  int size;
  int step_x = 1;
  int step_y = 1;

  // Matlab passes width and height, not coordinates
  max_width++;
  max_height++;

  if(learner_type[0] == 'H' && learner_type[1] == 'A')
    {
      sHA_params params;
      while(left_token=strchr(left_token,'_'))
        {
          right_token=strchr(left_token+1,'_');
          // if null, point at the end of the string
          if(right_token == 0)
            right_token = learner_type+strlen(learner_type);

          size = right_token - left_token - 2;

          //printf("left_token %s\n",left_token);
          //printf("right_token %s\n",right_token);

          switch(*(left_token+1))
            {
            case 'x':
              strncpy(temp,left_token+2,size);
              temp[size] = 0;
              //printf("x %s\n",temp);
              params.step_size_x = atoi(temp);
              break;
            case 'y':
              strncpy(temp,left_token+2,size);
              temp[size] = 0;
              //printf("y %s\n",temp);
              params.step_size_y = atoi(temp);
              break;
            case 'u':
              strncpy(temp,left_token+2,size);
              temp[size] = 0;
              step_x = atoi(temp);
              break;
            case 'v':
              strncpy(temp,left_token+2,size);
              temp[size] = 0;
              step_y = atoi(temp);
              break;
            }
          left_token = right_token;
        }

      // Generate all the weak learner for this type
      for(int sx=1;sx<max_width;sx+=params.step_size_x)
        {
          // minimum height for HA is 2
          for(int sy=2;sy<max_height;sy+=params.step_size_y)
            {
              for(int ix=1;ix<=max_width;ix+=step_x)
                for(int iy=1;iy<=max_height;iy+=step_y)
                  {
                    stringstream learner_id;
                    //learner_id << learner_type[0] << learner_type[1] << "_W_ax0ay0bx" << sx << "by" << sy/2 << "_B_ax0ay" << sy/2 << "bx" << sx << "by" << sy;
                    learner_id << learner_type[0] << learner_type[1]
                               << "_Wax" << ix << "ay" << iy << "bx" << (ix + sx) << "by" << (iy + sy/2)
                               << "_Bax" << ix << "ay" << (iy + sy/2) << "bx" << (ix + sx) << "by" << (iy + sy);
                    //cout << learner_id.str() << endl;
                    list_weak_learners.push_back(learner_id.str());
                  }
            }
        }
    }

  // Exporting the list
  weak_learners = new char*[list_weak_learners.size()];
  int idx = 0;
  for( vector<string>::iterator iter = list_weak_learners.begin();
       iter != list_weak_learners.end(); ++iter ) {
    weak_learners[idx] = new char[iter->length()+1];
    strcpy(weak_learners[idx], iter->c_str());
    idx++;
    //cout << *iter  << endl;
  }

  return list_weak_learners.size();
}
