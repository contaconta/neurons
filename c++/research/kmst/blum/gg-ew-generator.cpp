/***************************************************************************
                          gg-ew-generator.cpp  -  description
                             -------------------
    begin                : Thu Dec 13 15:07:28 CET 2000
    copyright            : (C) 2001 by Christian Blum
    email                : cblum@ulb.ac.be
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include "UndirectedGraph.h"
#include "Vertex.h"
#include "Edge.h"
#include "Random.h"
// using namespace std;

#define LINE_BUF_LEN 512
#define PROG_ID_STR "Grid Graph generator, V0.1"
#define CALL_MISSFILE_STR "You have to specify an output file. Example: -o output.f"
#define CALL_MISSCOLS_STR "You have to specify the number of columns. Example: -cols 10"
#define CALL_MISSROWS_STR "You have to specify the number of rows. Example: -rows 10"
#define CALL_MISSMAX_STR "You have to specify the maximum weight. Example: -max 100"

UndirectedGraph* graph;
Random* rg;
long int n_of_vertices;
long int n_of_edges;

time_t t;

char ofile[LINE_BUF_LEN];
bool ofile_is_given = false;
bool rows_given = false;
bool cols_given = false;
bool max_given = false;
bool xml = false;
int rows;
int cols;
int max_weight;
fstream fout;

bool comments = false;

Random* rg_gen;


inline int stoi(string &s) {

	return atoi(s.c_str());
}

inline double stof(string &s) {

	return atof(s.c_str());
}

void copySubstr(char *dest, char *src, int first, int last) {

	int i;
	for (i = 0; i<=last-first; i++) {
		dest[i] = src[i+first];
	}
	dest[i] = '\0';
}

string concatIntToString(string s, int i) {

	char c[10];
	sprintf (c, "%i", i);
	s = s + c;
	return s;
}

int random(int start, int end) {

  int ret_val = (int)(((double)start) + (( ((double)end) - ((double)start) + 1 ) * rg->next()));
  return ret_val;
}

UndirectedGraph* generate_graph() {
	   
  UndirectedGraph* aGraph = new UndirectedGraph();
  aGraph->setToGridGraph();
  
  Vertex* v = NULL;
  Edge* e = NULL;
  int key1 = -1;
  int key2 = -1;
  int key3 = -1;
  double d = 0.0;
  for (int i = 0; i < (rows * cols); i++) {
    v = new Vertex(i+1);
    aGraph->addVertex(v);
  }
  int count = 0;
  for (int i = 1; i < (rows + 1); i++) {
    for (int j = 1; j < (cols + 1); j++) {
      count = count + 1;
      if (!((j == cols) && (i == rows))) {
	key1 = j + ((i-1) * cols);
      }
      if (j != cols) {
	key2 = 1+ j + ((i-1) * cols);
      }
      if (i != rows) {
	key3 = j + ((i-1) * cols) + cols;
      }
      if (!(key1 == -1)) {
	if (!(key2 == -1)) {
	  e = new Edge(aGraph->vertex(key1),aGraph->vertex(key2));
	  d = (double)random(1,max_weight);
	  e->setWeight(d);
	  e->setID(count);
	  aGraph->addEdge(e);
	}
	if (!(key3 == -1)) {
	  e = new Edge(aGraph->vertex(key1),aGraph->vertex(key3));
	  d = (double)random(1,max_weight);
	  e->setWeight(d);
	  e->setID(count);
	  aGraph->addEdge(e);
	}
      }
      key1 = -1;
      key2 = -1;
      key3 = -1;
    }
  }
  return aGraph;
}

void read_parameters(int argc, char **argv) {

  int iarg=1;

  while (iarg < argc)
  {
    if (strcmp(argv[iarg],"-o")==0) {
      strcpy(ofile,argv[++iarg]);
      ofile_is_given = true;
    }
    else if (strcmp(argv[iarg],"-rows")==0) { 
      rows=atoi(argv[++iarg]);
      rows_given = true;
    }
    else if (strcmp(argv[iarg],"-cols")==0) {
      cols=atoi(argv[++iarg]);
      cols_given = true;
    }
    else if (strcmp(argv[iarg],"-max")==0) {
      max_weight=atoi(argv[++iarg]);
      max_given = true;
    }
    else if (strcmp(argv[iarg],"-xml")==0) {
      xml = true;
    }
    iarg++;
  }
  if (rows_given == false) {
    cout << CALL_MISSROWS_STR << endl;
    exit(1);
  }
  if (cols_given == false) {
    cout << CALL_MISSCOLS_STR << endl;
    exit(1);
  }
  if (max_given == false) {
    cout << CALL_MISSMAX_STR << endl;
    exit(1);
  }
  if (ofile_is_given == false) {
    cout << CALL_MISSFILE_STR << endl;
    exit(1);
  }
  else {
    fout.open(ofile,std::ios::out);
    if (!fout.is_open()) {
      cout << "Problems with writing to file." << endl;
      exit(1);
    }
  }
}

void write_to_file(UndirectedGraph* aGraph) {

  fout << rows * cols << endl;
  fout << (rows * (cols - 1)) + ((rows - 1) * cols) << endl;
  for (list<Edge*>::iterator e = (aGraph->edges).begin(); e != (aGraph->edges).end(); e++) {
    fout << ((*e)->fromVertex())->id();
    fout << "\t";
    fout << ((*e)->toVertex())->id();
    fout << "\t";
    fout << (*e)->weight() << endl;
  }
  fout.close();
}


int main(int argc, char *argv[])
{

  read_parameters(argc, argv);

  rg = new Random((unsigned) time(&t));

  UndirectedGraph* agraph = generate_graph();
  write_to_file(agraph);

  return EXIT_SUCCESS;
}

