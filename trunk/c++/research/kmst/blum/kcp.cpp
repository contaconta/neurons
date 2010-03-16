/***************************************************************************
                          kcp.cpp  -  description
                             -------------------
    begin                : Thu Dec 20 15:46:28 CET 2001
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
#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <stdio.h>
#include "Timer.h"
#include "Random.h"
#include <string>
#include "utilstuff.h"
#include "UndirectedGraph.h"
#include "GreedyHeuristic.h"

#define LINE_BUF_LEN 512
#define PROG_ID_STR "K-Card-Prim (KCP) for the k-cardinality tree problem, V0.1"
#define CALL_SYNTAX_STR "Parameter problems !!"
#define CALL_MISSFILE_STR "You have to specify an input file (i.e., -i instance.dat)."
#define CALL_MISSCARD_STR "You have to specify a cardinality (i.e., -cardb 10)."


char i_file[LINE_BUF_LEN];
char o_file[LINE_BUF_LEN];

UndirectedGraph* graph;

long int n_of_vertices;
long int n_of_edges;

long int cardb;
long int carde;

int cardmod = 1;

int cardinality;

/* parameter (partly initialized to a default value) */
string file;

/* possible values for daemon_action: "none", "simple" or "multi-nh" */
string ls = "none";

/* possible values for type: "vertex_based", "edge_based" */
string type = "vertex_based";

bool cardb_is_given = false;
bool carde_is_given = false;

bool input_file_given = false;
bool output_file_given = false;

time_t t;

Random* rnd;

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

void print_help() {

  printf("\n");
  printf("Usage: hc-aco-kct <options>\n");
  printf("<options> are: \n");
  printf("\n");
  printf("-i \t(i.e. -i instance.dat)\t-> the file to read the instance from.\n");
  printf("-cardb \t(i.e. -cardb 3)\t\t-> the start of the cardinality range for which to solve the problem.\n");
  printf("-carde \t(i.e. -cardb 3)\t\t-> the end of the cardinality range for which to solve the problem.\n");
  printf("-o \t(i.e. -o output.dat)\t\t-> the file you want to write the results to.\n");
  printf("-ls \t(i.e. -ls simple)\t\t-> options are 'simple' and 'none'.\n");
  printf("\n");
  printf("At least you have to specify an input file and either a time limit or a maximum iteration number!!.\n");
  printf("\n");
}

void read_parameters(int argc, char **argv)
{
  int iarg=1;

  while (iarg < argc)
  {
    if (strcmp(argv[iarg],"-i")==0) {
      strcpy(i_file,argv[++iarg]);
      input_file_given = true;
    }
    else if (strcmp(argv[iarg],"-o")==0) {
      strcpy(o_file,argv[++iarg]);
      output_file_given = true;
    }
    else if (strcmp(argv[iarg],"-ls")==0) {
      if (strcmp(argv[++iarg],"leafs")==0) {
	ls = "leafs";
      }
      else {
	if (strcmp(argv[iarg],"cycles_leafs")==0) {
	  ls = "cycles_leafs";
	}
	else {
	  ls = "none";
	}
      }
    }
    else if (strcmp(argv[iarg],"-type")==0) {
      if (strcmp(argv[++iarg],"edge_based")==0) {
	type = "edge_based";
      }
      else {
        type = "vertex_based";
      }
    }
    else if (strcmp(argv[iarg],"-cardb")==0) {
      cardb=atoi(argv[++iarg]);
      cardb_is_given = true;
    }
    else if (strcmp(argv[iarg],"-carde")==0) {
      carde=atoi(argv[++iarg]);
      carde_is_given = true;
    }
    else if (strcmp(argv[iarg],"-cardmod")==0) {
      cardmod = atoi(argv[++iarg]);
    }

    iarg++;
  }
  if (input_file_given == false) {
    printf(CALL_MISSFILE_STR);
    printf("\n");
    exit(1);
  }
  if (cardb_is_given == false) {
    printf(CALL_MISSCARD_STR);
    printf("\n");
    exit(1);
  }
  else {
    if (carde_is_given == false) {
      carde = cardb;
    }
  }
}

int main( int argc, char **argv ) {

  if ( argc < 2 ) {
    print_help();
    exit(1);
  }
  else {
    read_parameters(argc,argv);
  }

  // initialize random number generator
	
  rnd = new Random((unsigned) time(&t));

  // initialize and read instance from file

  graph = new UndirectedGraph(i_file);
  GreedyHeuristic gho(graph);

  for (int i = cardb; i <= carde; i++) {
    cardinality = i;
    if ((cardinality == cardb) || (cardinality == carde) || ((cardinality % cardmod) == 0)) {
      Timer timer;
      
      UndirectedGraph* greedyTree = new UndirectedGraph();
      if (type == "edge_based") {
	gho.getGreedyHeuristicResult(greedyTree,cardinality,ls);
      }
      else {
	if (type == "vertex_based") {
	  gho.getVertexBasedGreedyHeuristicResult(greedyTree,cardinality,ls);
	}
      }
      
      printf("%d\t%f\t%f\n",cardinality,greedyTree->weight(),timer.elapsed_time(Timer::VIRTUAL));
      
      delete(greedyTree);
    }
  }
  delete(graph);
  delete rnd;
}
