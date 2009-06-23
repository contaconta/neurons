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

using namespace std;

struct s_params
{
  int id;
  int size;
};

map<const char*, s_params> m;

int main(int argc,char*argv[])
{
  char str[]="aze_i23_s34";
  char temp[20];
  char* left_token = str;
  char* right_token = 0;
  s_params param;
  int size;

  while(left_token=strchr(left_token,'_'))
    {
      right_token=strchr(left_token+1,'_');
      // if null, point at the end of the string
      if(right_token == 0)
        right_token = str+strlen(str);

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
          param.id = atoi(temp);
          break;
        case 's':
          strncpy(temp,left_token+2,size);
          temp[size] = 0;
          printf("s %s\n",temp);
          param.size = atoi(temp);
          break;
        }
      left_token = right_token;
    }
  m[str] = param;

  for( map<const char*, s_params>::iterator iter = m.begin();
       iter != m.end(); ++iter ) {
    cout << (*iter).first << " " << (*iter).second.id << endl;
  }

  return 0;
}
