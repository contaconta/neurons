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
#include "utils_neseg.h"

using namespace std;

//Node class
class Map
{
public:
  virtual void  print(ostream &out)=0;
  virtual float eval()=0;
  virtual string type()=0;
};

//Constant
class Cst: public Map{
  float value;
public:
  Cst(float v) : value(v) {}
  Cst(string str){
    std::istringstream inpStream(str);
    inpStream >> value;
  }
  void print(ostream &o) { o << value; }
  float eval() { return value; }
  string type(){return "Cst";}
};


//Name (future image)
class Name: public Map{
  string name;
public:
  Name(string n) : name(n) {}
  void print(ostream &o) { o << name; }
  float eval() { return 0; }
  string type(){return "Name";}
};

//Multiplication
class Mult: public Map{
  Map *f1, *f2;
public:
  Mult(Map* g1, Map* g2) : f1(g1), f2(g2) {}
  void print(ostream &out) {
    f1->print(out); out << " * "; f2->print(out);}
  float eval() {return f1->eval() * f2->eval();}
  string type(){return "Mult";}
};

//Adittion
class Add: public Map{
  Map *f1, *f2;
public:
  Add(Map* g1, Map* g2) : f1(g1), f2(g2) {}
  void print(ostream &out) {
    f1->print(out); out << " + "; f2->print(out);}
  float eval() {return f1->eval() + f2->eval();}
  string type(){return "Add";}
};


//Sine
class Sin: public Map{
  Map *f1;
public:
  Sin(Map* g1) : f1(g1) {}
  void print(ostream &out) {
    out << "sin(";  f1->print(out); out << ")";}
  float eval() {return sin(f1->eval());}
  string type(){return "Sin";}
};

Parser* psc;
string oneCharTokens = ",+-*/()";


//FIXME
int positionOperatorWithPriority(string s)
{
  Parser* psc = new Parser(s, oneCharTokens, false);
  vector< string > tokens = psc->getAllTokens();

  int min_idx = 0;
  float level = 0;
  float min_level = 100;
  string tk;

  // while(tk != ""){
  for(int idx = 0; idx < tokens.size(); idx++){
    tk = tokens[idx];

    // Prints the tokens and the levels
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
      //To avoid incrementing twice the level
      tk = tokens[++idx];
      if(tk != "("){
        std::cout << "Error, " << tk << " "
                  << " without (\n" << std::endl;
        exit(0);
      }
      ++idx;
    }
    // tk = psc->getNextToken();
    // idx+=1;
  }
  return min_idx;
}


Map* parseString(string s)
{

  // Find the idx of the token that has the priority
  int split = positionOperatorWithPriority(s);
  Parser* psc = new Parser(s, oneCharTokens, false);
  vector< string > tokens = psc->getAllTokens();

  //If we are in a leave or in an operator with just one argument
  //FIXME -> erase the parenthesis at the end, problem with multiple enclosing parenthesis
  if(split == 0){
    string tk = tokens[0];
    // Elliminates the parenthesis at the beginning and at the end
    int idx_p = 0;
    while( tk == "("){
      tk = tokens[++idx_p];
    }
    if (isNumber(tk)){
      return new Cst(tk);
    }
    if( tk == "sin"){
      if(tokens[++idx_p] != "("){
        std::cout << "Error, operation without (\n" << std::endl;
        exit(0);
      }
      string rest = "";
      int level = 0;
      while(!((tokens[++idx_p] == ")") && (level == 0))){
        tk = tokens[idx_p];
        rest = rest + tk + " ";
        if(tk == "(")
          level ++;
        if(tk == ")")
          level --;
      }
      std::cout << "Inside Sin " << rest << std::endl;
      return new Sin( parseString(rest) );
    }

    //Final case, we do not know what it is
    return new Name(tk);
  }

  if(split != 0){
    string left;
    string right;
    string op = tokens[split];
    int idx = 0;
    for(int i = 0; i < split; i++)
      left = left + tokens[i];
    for(int i = split+1; i < tokens.size(); i++)
      right = right + tokens[i];
    if(op=="*"){
      return new Mult(parseString(left), parseString(right));
    }
    if(op=="+"){
      return new Add(parseString(left), parseString(right));
    }
  }

  // return new Cst(100);
}


int main(int argc, char **argv) {

  string eq = argv[1];

  Parser* psc = new Parser(eq,
                   oneCharTokens,false);
  int idx = 0;
  vector< string > tokens = psc->getAllTokens();
  for(int i = 0; i < tokens.size(); i++)
    std::cout << idx++ << " " << tokens[i] << std::endl;


  // string s = psc->getNextToken();
  // while( s != ""){
    // std::cout << idx << ": " << s << std::endl;
    // s = psc->getNextToken();
    // idx++;
  // }


  Map* root = parseString(eq);
  root->print(std::cout);
  printf(" = %f\n", root->eval());


  // Map* root = parseString();
  // root->print(std::cout);
  std::cout << std::endl;
}
