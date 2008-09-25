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
#include "Parser.h"

using namespace std;

//Node class
class Map
{
public:
  virtual void  print(ostream &out)=0;
  virtual float eval()=0;
};

//Constant
class Cst: public Map{
  float value;
public:
  Cst(float v) : value(v) {}
  void print(ostream &o) { o << value; }
  float eval() { return value; }
};


//Name (future image)
class Name: public Map{
  string name;
public:
  Name(string n) : name(n) {}
  void print(ostream &o) { o << name; }
  float eval() { return 0; }
};

//Multiplication
class Mult: public Map{
  Map *f1, *f2;
public:
  Mult(Map* g1, Map* g2) : f1(g1), f2(g2) {}
  void print(ostream &out) {
    f1->print(out); out << " * "; f2->print(out);}
  float eval() {return f1->eval() * f2->eval();}
};

//Sine
class Sin: public Map{
  Map *f1;
public:
  Sin(Map* g1) : f1(g1) {}
  void print(ostream &out) {
    out << "sin(";  f1->print(out); out << ")";}
  float eval() {return sin(f1->eval());}
};

Parser* psc;
string oneCharTokens = ",+-*/()";

int positionOperatorWithPriority(string s)
{
  Parser* psc = new Parser(s, oneCharTokens, false);

  int idx = 0;
  int min_idx = 0;
  float level = 0;
  float min_level = 100;
  string tk = psc->getNextToken();

  while(tk != ""){
    // std::cout << idx << ": l: " << level << " : " << min_level << " : " << tk << std::endl;
    // std::cout << idx << ": " << tk << std::endl;
    if((tk == "+") || (tk == "-")){
      if(level + 0.4 < min_level){
        min_level = level+0.4;
        min_idx = idx;
      }
    }
    if((tk == "*") || (tk == "/")){
      if(level + 0.6 < min_level){
        min_level = level+0.6;
        min_idx = idx;
      }
    }
    if( tk == "("){
      level += 1;
    }
    if( tk == ")"){
      level -= 1;
    }
    if ( (tk == "sin") || (tk == "pow") ){
      level += 1;
      if(level < min_level){
        min_level = level;
        min_idx   = idx;
      }
      tk = psc->getNextToken();
      if(tk != "("){
        std::cout << "Error, operation without (\n" << std::endl;
        exit(0);
      }
    }
    tk = psc->getNextToken();
    idx+=1;
  }
  return min_idx;
}


Map* parseString(string s)
{

  int split = positionOperatorWithPriority(s);
  Parser* psc = new Parser(s, oneCharTokens, false);

  //If we are in a leave or in an operator with just one argument
  if(split == 0){
    string tk = psc->getNextToken();
    // Elliminates the parenthesis at the beginning
    while( tk == "("){
      tk = psc->getNextToken();
    }
    if( tk == "sin"){
      string rest = "";
      while(tk != ""){
        tk = psc->getNextToken();
        rest = rest + tk + " ";

      }
      return new Sin( parseString(rest) );
    }
    else{
      return new Name(tk);
    }
  }

  if(split != 0){
    string left;
    string right;
    string op;
    int idx = 0;
    string tk = psc->getNextToken();
    while( tk != ""){
      if(idx < split){
        left = left + " " + tk;
      }
      if( idx == split)
        op = tk;
      if(idx > split)
        right = right + " " + tk;
      tk = psc->getNextToken();
      idx++;
    }
    if(op=="*"){
      return new Mult(parseString(left), parseString(right));
    }
  }

  // return new Cst(100);
}


int main(int argc, char **argv) {

  string eq = argv[1];

  Parser* psc = new Parser(eq,
                   oneCharTokens,false);
  int idx = 0;
  string s = psc->getNextToken();
  while( s != ""){
    std::cout << idx << ": " << s << std::endl;
    s = psc->getNextToken();
    idx++;
  }


  Map* root = parseString(eq);
  root->print(std::cout);


  // Map* root = parseString();
  // root->print(std::cout);
  // std::cout << std::endl;
}
