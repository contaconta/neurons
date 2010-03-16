
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

#ifndef PARSER_H_
#define PARSER_H_

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

class Parser
{
public:

  //Where the file is
  std::istream*  istr;
  string        fileName;
  string        oneCharTokens;

  Parser(string oneCharTokens);
  Parser(string filename, string oneCharTokens, bool filenameIsFile = true);

  string getNextToken();
  vector< string > getAllTokens();
};



#endif
