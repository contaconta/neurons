
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

#include "Parser.h"


Parser::Parser(string oneCharTokens)
{
  this->oneCharTokens = oneCharTokens;
}

Parser::Parser
(string fileName,
 string oneCharTokens,
 bool filenameIsFile)
{
  this->oneCharTokens = oneCharTokens;
  if(filenameIsFile){
    istr = new ifstream(fileName.c_str());
    this->fileName = fileName;
  } else {
    this->fileName = "";
    istr = new istringstream(fileName);
  }
}

string Parser::getNextToken()
{
  if(!istr->good())
    return "";

  char ch;
  ch = istr->get();

  //elliminate the blank spaces and open line and gets the next characters
  while((ch==' ') || (ch=='\n') || (ch == '\t') || (ch == '\r')  )
    {
      ch = istr->get();
      if(istr->eof()) return "";
    }

  //If it is a one char token, return it
  if(oneCharTokens.find(ch)!=string::npos)
    {
      string s = "";
      s = s + ch;
      return s;
    }
  else
    {
      //Read until next one char token or the end of file or blank space
      string s;
      s = s + ch;
      ch = istr->get();
      string numbers = "0123456789.";
      bool first_char = true;
      bool is_number    = false;
      while(!(
              // It is a oneCharToken and it is not a '-'
              istr->eof()	 		     ||
              !istr->good()	 		     ||
              ( (ch != '-') && (oneCharTokens.find(ch)!=string::npos) ) ||
              (ch == ' ')			     ||
              (ch == '\n')			     ||
              (ch == '\t')			     ||
              (ch == '\r')                           ||
              ( (ch == '-') && is_number ) //We allow names with '-'
               )
            )
        {
          //if the first char is not a number, we assume we are dealing with a name
          if(first_char){
            if(numbers.find(ch)!=string::npos){
              is_number = true;
            }
            first_char = false;
          }
          s = s + ch;
          ch = istr->get();
        }
      istr->putback(ch);
      return s;
    }
}



vector< string > Parser::getAllTokens()
{
  vector< string > tokens;
  string tk = getNextToken();
  while(tk!=""){
    tokens.push_back(tk);
    tk = getNextToken();
  }
  return tokens;
}
