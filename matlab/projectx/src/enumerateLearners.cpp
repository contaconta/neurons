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
  int id;
  int step_x;
  int step_y;
};

//map<const char*, char*> m_weak_learners;
vector<string> list_weak_learners;

int enumerate_learners(char *learner_type, int max_width, int max_height,char**& weak_learners)
{
  char temp[20];
  char* left_token = learner_type;
  char* right_token = 0;
  int size;

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

          printf("left_token %s\n",left_token);
          printf("right_token %s\n",right_token);
          printf("Size %d\n",size);

          switch(*(left_token+1))
            {
            case 'i':
              strncpy(temp,left_token+2,size);
              temp[size] = 0;
              printf("i %s\n",temp);
              params.id = atoi(temp);
              break;
            case 'x':
              strncpy(temp,left_token+2,size);
              temp[size] = 0;
              printf("s %s\n",temp);
              params.step_x = atoi(temp);
              break;
            case 'y':
              strncpy(temp,left_token+2,size);
              temp[size] = 0;
              printf("s %s\n",temp);
              params.step_y = atoi(temp);
              break;
            }
          left_token = right_token;
        }

      // TODO : enumerate all possible positions starting at (1,1)
      // Generate all the weak learner for this type
      for(int sx=1;sx<max_width;sx+=params.step_x)
        //for(int sx=1;sx<=max_width/params.step_x;sx++)
        {
          //for(int sy=1;sy<=max_width/params.step_y;sy++)
            for(int sy=2;sy<max_height;sy+=params.step_y)
            {
              //string learner_id;
              stringstream learner_id;
              learner_id << learner_type[0] << learner_type[1] << "_W_ax0ay0bx" << sx << "by" << sy/2 << "_B_ax0ay" << sy/2 << "bx" << sx << "by" << sy;

              //sprintf("W_ax0ay0bx%dby%d_B_ax0ay%dbx%dby%d",sx,sy/2,sy/2,sx,sy);
              //sWeak_learner_params weak_learner_params;
              //weak_learner_params.x_size = sx;
              //weak_learner_params.y_size = sy;
              //m_weak_learners[learner_id] = weak_learner_params;
              list_weak_learners.push_back(learner_id.str());
            }
        }
    }

  weak_learners = new char*[list_weak_learners.size()];
  int idx = 0;
  for( vector<string>::iterator iter = list_weak_learners.begin();
       iter != list_weak_learners.end(); ++iter ) {
    weak_learners[idx] = new char[iter->length()];
    strcpy(weak_learners[idx], iter->c_str());
    idx++;
    //cout << *iter  << endl;
  }

  return list_weak_learners.size();
}
