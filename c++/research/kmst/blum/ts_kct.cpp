/***************************************************************************
                          ts_kct.cpp  -  description
                             -------------------
    begin                : Mon Sep  24 15:07:28 CET 2002
    copyright            : (C) 2002 by Christian Blum
    email                : cblum@ulb.ac.be
    home page            : http://iridia.ulb.ac.be/~cblum/
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version. In case of usage for publication  *
 *   purposes, it is obligatory to ask the permission of the author        *                           
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
#include <list>
#include <string>
#include <math.h>
#include "UndirectedGraph.h"
#include "Vertex.h"
#include "Edge.h"
#include "Random.h"
#include "utilstuff.h"
#include "Timer.h"
#include "GreedyHeuristic.h"
#include "LocalSearch.h"
#include "LocalSearchB.h"
// using namespace std;

#define LINE_BUF_LEN 512
#define PROG_ID_STR "Tree-Crossbreeding-Algorithm (TCA) for the k-cardinality tree (KCT) problem, V0.1"
#define CALL_SYNTAX_STR "Parameter problems !!"
#define CALL_MISSFILE_STR "You have to specify an input file (i.e., -i instance.dat)."
#define CALL_MISSCARD_STR "You have to specify a cardinality (i.e., -cardb 10)."

UndirectedGraph* graph;
Random* rg;
long int n_of_vertices;
long int n_of_edges;

long int cardb;
long int carde;
long int cardmod = 1;
double tlimit = 0.0;
int n_of_iter = 0;

int n_of_trials = 1;

time_t t;
map<int,double> times;
char ifile[LINE_BUF_LEN];
char mstfile[LINE_BUF_LEN];
char tfile[LINE_BUF_LEN];
char oput[5];
char ls_cstring[10];
bool ifile_is_given = false;
bool mstfile_is_given = false;
bool tfile_is_given = false;
bool cardb_is_given = false;
bool carde_is_given = false;
bool iter_limit_given = false;
bool time_limit_given = false;
bool help_needed = false;

/* output can be minimal (min) or detailed (det) */
string output = "det";

bool comments = false;

map<Edge*,int> in_list_map;
map<Edge*,int> out_list_map;
list<Edge*> in_list;
list<Edge*> out_list;
int in_length = 50;
int out_length = 50;

list<Leaf*> neighborhood;
list<Leaf*> leafs;

string moveType = "first_improvement";
string cycleMoves = "no";

bool seed_is_given = false;
int seed;

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

void readTimesFile() {
  
  FILE* timesFile=fopen(tfile, "r");

  //int counter = 0;
  //bool goon = true;
  //while(feof(timesFile)==0) {
  while(true) {
    //counter++;
    //cout << counter << endl;
    int card;
    double time;
    int first = fscanf(timesFile, "%ld", &card);
    if (first == EOF) {
      break;
    }
    else {
      if (first < 0) {
	printf("error reading cardinality in times-file\n");
	exit(1);
      }
    }
    int second = fscanf(timesFile, "%lf", &time);
    if (second == EOF) {
      break; 
    }
    else {
      if (second < 0) {
	printf("error reading time in times-file\n");
	exit(1);
      }
    }
    //cout << "card " << card << "\ttime" << time << endl;
    times[card] = time;
  }
  fclose(timesFile);
}

void init_program(const int& argc, char* argv[]) {

  int iarg=1;

  while (iarg < argc)
  {
    if (strcmp(argv[iarg],"-i")==0) {
      strcpy(ifile,argv[++iarg]);
      ifile_is_given = true;
    }
	else if (strcmp(argv[iarg],"-m")==0) {
		strcpy(mstfile,argv[++iarg]);
		mstfile_is_given = true;
    }
    else if (strcmp(argv[iarg],"-tfile")==0) {
      strcpy(tfile,argv[++iarg]);
      readTimesFile();
      tfile_is_given = true;
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
      cardmod=atoi(argv[++iarg]);
    }
    else if (strcmp(argv[iarg],"-n")==0) {
      n_of_trials=atoi(argv[++iarg]);
    }    
    else if (strcmp(argv[iarg],"-maxiter")==0) {
      n_of_iter=atoi(argv[++iarg]);
      iter_limit_given = true;
    }
    else if (strcmp(argv[iarg],"-t")==0) {
      tlimit=atof(argv[++iarg]);
      time_limit_given = true;
    }
    else if (strcmp(argv[iarg],"-seed")==0) {
      seed=atoi(argv[++iarg]);
      seed_is_given = true;
    }
    else if (strcmp(argv[iarg],"-h")==0) {
      help_needed = true;
    }
    else if (strcmp(argv[iarg],"-output")==0) {
      strcpy(oput,argv[++iarg]);
      if (strcmp(oput,"det") == 0) {
	output = "det";
      }
      else {
	output = "min";
      }
    }
    else if (strcmp(argv[iarg],"-improvement")==0) {
      strcpy(oput,argv[++iarg]);
      if (strcmp(oput,"best") == 0) {
	moveType = "best_improvement";
      }
      else {
	moveType = "first_improvement";
      }
    }
    else if (strcmp(argv[iarg],"-cyclemoves")==0) {
      strcpy(oput,argv[++iarg]);
      if (strcmp(oput,"yes") == 0) {
	cycleMoves = "yes";
      }
      else {
	cycleMoves = "no";
      }
    }
    else if (strcmp(argv[iarg],"-inlength")==0) {
      in_length=atoi(argv[++iarg]);
    }
    else if (strcmp(argv[iarg],"-outlength")==0) {
      out_length=atoi(argv[++iarg]);
    }
    
    iarg++;
  }

  /* initialize random generator */
  if (seed_is_given) {
    cout << "initializing with seed" << endl;
    rg = new Random((unsigned) seed);
  }
  else {
    rg = new Random((unsigned) time(&t));
    rg->next();
  }

  if (help_needed == true) {
    //print_help();
    exit(1);
  }
  else {
    if (ifile_is_given == false) {
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
    if ((time_limit_given == false) && (iter_limit_given == false) && (tfile_is_given == false)) {
      cout << endl;
      cout << "please specify:" << endl;
      cout << endl;
      cout << "* a time limit (i.e., -t 20.0), or" << endl;
      cout << "* an iteration limit (i.e., -maxiter 1000), or" << endl;
      cout << "* both" << endl;
      cout << endl;
      exit(1);
    }
  }
}

bool treesDisjunct(UndirectedGraph* t1, UndirectedGraph* t2) {

  bool result = true;
  list<Edge*>::iterator e1 = ((*t1).edges).begin();
  while ((e1 != ((*t1).edges).end()) && (result == true)) {
    if (t2->contains(*e1)) {
      result = false;
    }
    e1++;
  }
  return result;
}

bool treesEqual(UndirectedGraph* t1, UndirectedGraph* t2) {

  bool result = true;
  list<Edge*>::iterator e1 = ((*t1).edges).begin();
  while ((e1 != ((*t1).edges).end()) && (result == true)) {
    if (!(t2->contains(*e1))) {
      result = false;
    }
    e1++;
  }
  return result;
}

int random(int start, int end) {

  int ret_val = (int)(((double)start) + (( ((double)end) - ((double)start) + 1 ) * rg->next()));
  return ret_val;
}

void initializeTabuLists() {

  in_list.clear();
  out_list.clear();
  for (list<Edge*>::iterator e = ((*graph).edges).begin(); e != ((*graph).edges).end(); e++) {
    in_list_map[*e] = 0;
    out_list_map[*e] = 0;
  }
}

void cutTabuLists() {

  while (in_list.size() > in_length) {
    Edge* fEdge = in_list.front();
    in_list_map[fEdge] = 0;
    in_list.pop_front();
  }
  while (out_list.size() > out_length) {
    Edge* fEdge = out_list.front();
    out_list_map[fEdge] = 0;
    out_list.pop_front();
  }
}


void adaptTabuLists(Edge* inEdge, Edge* outEdge) {

  if (out_list_map[inEdge] == 0) {
    if (out_list.size() == out_length) {
      Edge* anEdge = out_list.front();
      out_list.pop_front();
      out_list_map[anEdge] = 0;
    }
    out_list.push_back(inEdge);
    out_list_map[inEdge] = 1;
  }
  else {
    out_list.remove(inEdge);
    out_list.push_back(inEdge);
  }
  if (in_list_map[outEdge] == 0) {
    if (in_list.size() == in_length) {
      Edge* anEdge = in_list.front();
      in_list.pop_front();
      in_list_map[anEdge] = 0;
    }
    in_list.push_back(outEdge);
    in_list_map[outEdge] = 1;
  }
  else {
    in_list.remove(outEdge);
    in_list.push_back(outEdge);
  }
}

bool isTabu(Edge* inEdge, Edge* outEdge) {

  bool return_val = false;
  if ((in_list_map[inEdge] == 1) && (out_list_map[outEdge] == 1)) {
    return_val = true;
  }
  return return_val;
}

LSMove* getBestMove(double bestValue, double currentValue) {

  LSMove* bm = NULL;
  double weight_diff = 0.0;
  bool started = false;
  Leaf* inl = NULL;
  Leaf* outl = NULL;
  for (list<Leaf*>::iterator anIn = neighborhood.begin(); anIn != neighborhood.end(); anIn++) {
    for (list<Leaf*>::iterator anOut = leafs.begin(); anOut != leafs.end(); anOut++) {
      if (((*anIn)->lEdge)->otherVertex((*anIn)->lVertex) != (*anOut)->lVertex) {
	bool istabu = isTabu((*anIn)->lEdge,(*anOut)->lEdge);
	double help = ((*anIn)->lEdge)->weight() - ((*anOut)->lEdge)->weight();
	if ((istabu && ((currentValue + help) < bestValue)) || (!istabu)) {
	  if (started == false) {
	    started = true;
	    inl = *anIn;
	    outl = *anOut;
	    weight_diff = help;
	  }
	  else {
	    if (help < weight_diff) {
	      inl = *anIn;
	      outl = *anOut;
	      weight_diff = help;
	    }
	  }
	}
      }
    }
  }
  if ((inl != NULL) && (outl != NULL)) {
    bm = new LSMove(inl,outl);
  }
  return bm;
}

LSMove* getFirstMove(double bestValue, double currentValue) {

  LSMove* bm = NULL;
  double weight_diff = 0.0;
  bool started = false;
  Leaf* inl = NULL;
  Leaf* outl = NULL;
  bool stop = false;
  for (list<Leaf*>::iterator anIn = neighborhood.begin(); (anIn != neighborhood.end()) && (!stop); anIn++) {
    for (list<Leaf*>::iterator anOut = leafs.begin(); (anOut != leafs.end()) && (!stop); anOut++) {
      if (((*anIn)->lEdge)->otherVertex((*anIn)->lVertex) != (*anOut)->lVertex) {
	bool istabu = isTabu((*anIn)->lEdge,(*anOut)->lEdge);
	double help = ((*anIn)->lEdge)->weight() - ((*anOut)->lEdge)->weight();
	if ((istabu && ((currentValue + help) < bestValue)) || (!istabu)) {
	  if (started == false) {
	    started = true;
	    inl = *anIn;
	    outl = *anOut;
	    weight_diff = help;
	    if (weight_diff < 0.0) {
	      stop = true;
	    }
	  }
	  else {
	    if (help < weight_diff) {
	      inl = *anIn;
	      outl = *anOut;
	      weight_diff = help;
	      if (weight_diff < 0.0) {
		stop = true;
	      }
	    }
	  }
	}
      }
    }
  }
  if ((inl != NULL) && (outl != NULL)) {
    bm = new LSMove(inl,outl);
  }
  return bm;
}

void generateSortedNeighborhood(UndirectedGraph* tree) {

  for (list<Leaf*>::iterator aL = neighborhood.begin(); aL != neighborhood.end(); aL++) {
    delete(*aL);
  }
  neighborhood.clear();
  Leaf* theLeaf = NULL;
  for (list<Edge*>::iterator i = ((*graph).edges).begin(); i != ((*graph).edges).end(); i++) {
    if (!(tree->contains(*i))) {
      bool doit = false;
      if (tree->contains((*i)->fromVertex()) && (!tree->contains((*i)->toVertex()))) {
	theLeaf = new Leaf(*i,(*i)->toVertex());
	doit = true;
      }
      if (tree->contains((*i)->toVertex()) && (!tree->contains((*i)->fromVertex()))) {
	theLeaf = new Leaf(*i,(*i)->fromVertex());
	doit = true;
      }
      if (doit) {
	bool inserted = false;
	list<Leaf*>::iterator aLeaf;
	for (aLeaf = neighborhood.begin(); aLeaf != neighborhood.end(); aLeaf++) {
	  if (((*theLeaf).getEdge())->weight() >= ((*aLeaf)->getEdge())->weight()) {
	    break;
	    inserted = true;
	  }
	}
	if (inserted == true) {
	  neighborhood.insert(aLeaf,theLeaf);
	}
	else {
	  neighborhood.push_back(theLeaf);
	}
      }
    }
  }
}

void generateSortedLeafs(UndirectedGraph* tree) {

  for (list<Leaf*>::iterator aL = leafs.begin(); aL != leafs.end(); aL++) {
    delete(*aL);
  }
  leafs.clear();
  for (list<Vertex*>::iterator iv = ((*tree).vertices).begin(); iv != ((*tree).vertices).end(); iv++) {
    if (((*tree).incidentEdges(*iv))->size() == 1) {
      Edge* le = *(((*tree).incidentEdges(*iv))->begin());
      Leaf* newLeaf = new Leaf(le,*iv);
      bool inserted = false;
      list<Leaf*>::iterator aLeaf;
      for (aLeaf = leafs.begin(); aLeaf != leafs.end(); aLeaf++) {
	Edge* cle = (*aLeaf)->getEdge();
	if (le->weight() >= cle->weight()) {
	  break;
	  inserted = true;
	}
      }
      if (inserted == true) {
	leafs.insert(aLeaf,newLeaf);
      }
      else {
	leafs.push_back(newLeaf);
      }
    }
  }
}

list<Leaf*> generateLeafs(UndirectedGraph* tree) {

  list<Leaf*> shrinkLeafs;
  for (list<Vertex*>::iterator iv = ((*tree).vertices).begin(); iv != ((*tree).vertices).end(); iv++) {
    if (((*tree).incidentEdges(*iv))->size() == 1) {
      Edge* le = *(((*tree).incidentEdges(*iv))->begin());
      Leaf* newLeaf = new Leaf(le,*iv);
      shrinkLeafs.push_back(newLeaf);
    }
  }
  return shrinkLeafs;
}

void initializeLeafsAndNeighborhood(UndirectedGraph* tree) {

  generateSortedNeighborhood(tree);
  generateSortedLeafs(tree);
}

void adaptLeafs(LSMove* aMove, UndirectedGraph* tree) {

  leafs.remove(aMove->out);

  Leaf* toRemove = NULL;
  for (list<Leaf*>::iterator aLeaf = leafs.begin(); aLeaf != leafs.end(); aLeaf++) {
    Vertex* other = ((aMove->in)->getEdge())->otherVertex((aMove->in)->getVertex());
    if ((*aLeaf)->getVertex() == other) {
      toRemove = *aLeaf;
      break;
    }
  }
  if (toRemove != NULL) {
    leafs.remove(toRemove);
    delete(toRemove);
  }

  Vertex* ov = ((aMove->out)->getEdge())->otherVertex((aMove->out)->getVertex());
  if (((*tree).incidentEdges(ov))->size() == 1) {
    Edge* le = *(((*tree).incidentEdges(ov))->begin());
    if ((*tree).isLeave(ov)) {
      bool inserted = false;
      list<Leaf*>::iterator aLeaf;
      for (aLeaf = leafs.begin(); aLeaf != leafs.end(); aLeaf++) {
	Edge* cle = (*aLeaf)->getEdge();
	if (le->weight() >= cle->weight()) {
	  break;
	  inserted = true;
	}
      }
      if (inserted == true) {
	leafs.insert(aLeaf,new Leaf(le,ov));
      }
      else {
	leafs.push_back(new Leaf(le,ov));
      }
    }
  }
  
  Leaf* newLeaf = (aMove->in)->copy();
  Edge* le = newLeaf->getEdge();
  bool inserted = false;
  list<Leaf*>::iterator aLeaf;
  for (aLeaf = leafs.begin(); aLeaf != leafs.end(); aLeaf++) {
    Edge* cle = (*aLeaf)->getEdge();
    if (le->weight() >= cle->weight()) {
      break;
      inserted = true;
    }
  }
  if (inserted == true) {
    leafs.insert(aLeaf,newLeaf);
  }
  else {
    leafs.push_back(newLeaf);
  }
}

void adaptNeighborhood(LSMove* aMove, UndirectedGraph* tree) {

  neighborhood.remove(aMove->in);

  list<Leaf*> toRemove;
  for (list<Leaf*>::iterator aLeaf = neighborhood.begin(); aLeaf != neighborhood.end(); aLeaf++) {
    Edge* moveEdge = (*aLeaf)->getEdge();
    Vertex* other = moveEdge->otherVertex((*aLeaf)->getVertex());
    if ((*tree).contains(moveEdge->fromVertex()) && (*tree).contains(moveEdge->toVertex())) {
      toRemove.push_back(*aLeaf);
    }
    else {
      if ((!(*tree).contains(moveEdge->fromVertex())) && (!(*tree).contains(moveEdge->toVertex()))) {
	toRemove.push_back(*aLeaf);
      }
      else {
	if (other == (aMove->out)->getVertex()) {
	  toRemove.push_back(*aLeaf);
	}
      }
    }
  }
  for (list<Leaf*>::iterator aLeaf = toRemove.begin(); aLeaf != toRemove.end(); aLeaf++) {
    neighborhood.remove(*aLeaf);
    delete(*aLeaf);
  }
  toRemove.clear();
  
  list<Edge*>* incidents = (*graph).incidentEdges((aMove->in)->getVertex());
  for (list<Edge*>::iterator anEdge = (*incidents).begin(); anEdge != (*incidents).end(); anEdge++) {
    if (!((*tree).contains(*anEdge))) {
      Vertex* nv = (*anEdge)->otherVertex((aMove->in)->getVertex());
      if (!((*tree).contains(nv))) {
	Leaf* nn = new Leaf(*anEdge,nv);
	bool inserted = false;
	list<Leaf*>::iterator aN;
	for (aN = neighborhood.begin(); aN != neighborhood.end(); aN++) {
	  Edge* cle = (*aN)->getEdge();
	  if ((*anEdge)->weight() >= cle->weight()) {
	    break;
	    inserted = true;
	  }
	}
	if (inserted == true) {
	  neighborhood.insert(aN,nn);
	}
	else {
	  neighborhood.push_back(nn);
	}
      }
    }
  }
  
  Vertex* inVertex = (aMove->out)->getVertex();
  incidents = (*graph).incidentEdges(inVertex);
  for (list<Edge*>::iterator anEdge = (*incidents).begin(); anEdge != (*incidents).end(); anEdge++) {
    if ((*tree).contains((*anEdge)->otherVertex(inVertex))) {
      Leaf* nn = new Leaf(*anEdge,inVertex);
      bool inserted = false;
      list<Leaf*>::iterator aN;
      for (aN = neighborhood.begin(); aN != neighborhood.end(); aN++) {
	Edge* cle = (*aN)->getEdge();
	if ((*anEdge)->weight() >= cle->weight()) {
	  break;
	  inserted = true;
	}
      }
      if (inserted == true) {
	neighborhood.insert(aN,nn);
      }
      else {
	neighborhood.push_back(nn);
      }
    }
  }
}

void shrinkTree(UndirectedGraph* aTree) {

  list<Leaf*> leafs_copy = generateLeafs(aTree);
  while (!(leafs_copy.size() == 0)) {
    Leaf* delLeaf = leafs_copy.front();    
    Vertex* rv = delLeaf->getVertex();
    Edge* re = delLeaf->getEdge();
    (*aTree).remove(re);
    (*aTree).remove(rv);
    leafs_copy.pop_front();
    delete(delLeaf);
    Vertex* ov = re->otherVertex(rv);
    if (((*aTree).incidentEdges(ov))->size() == 1) {
      Edge* le = *(((*aTree).incidentEdges(ov))->begin());
      if ((*aTree).isLeave(ov)) {
	Leaf* newLeaf = new Leaf(le,ov);
	leafs_copy.push_back(newLeaf);
      }
    }
  }
}

int compareEdges(Edge* first, Edge* second) {

  int result = 0;
  if (first->weight() < second->weight()) {
    result = -1;
  }
  else {
    if (first->weight() == second->weight()) {
      result = 0;
    }
    else {
      result = 1;
    }
  }
  return result;
}

bool applyFirstBestCycleMove(UndirectedGraph* tree) {

  list<Edge*> cc;
  for (list<Vertex*>::iterator aV = (*tree).vertices.begin(); aV != (*tree).vertices.end(); aV++) {
    list<Edge*>* incidents = (*graph).incidentEdges(*aV);
    for (list<Edge*>::iterator anE = (*incidents).begin(); anE != (*incidents).end(); anE++) {
      if ((*tree).contains((*anE)->otherVertex(*aV))) {
	if (!(*tree).contains(*anE)) {
	  cc.push_back(*anE);
	}
      }
    }
  }
  cc.sort(compareEdges);
  bool stop = false;
  Edge* inEdge = NULL;
  Edge* outEdge = NULL;
  UndirectedGraph* tCopy = new UndirectedGraph();
  for (list<Edge*>::iterator anE = cc.begin(); (anE != cc.end()) && (!stop); anE++) {
    tCopy->copy(tree);
    tCopy->addEdge(*anE);
    shrinkTree(tCopy);
    Edge* remEdge = NULL;
    double max_weight = 0.0;
    bool started = false;
    for (list<Edge*>::iterator e = (*tCopy).edges.begin(); e != (*tCopy).edges.end(); e++) {
      if (started == false) {
	started = true;
	max_weight = (*e)->weight();
	remEdge = *e;
      }
      else {
	if ((*e)->weight() > max_weight) {
	  max_weight = (*e)->weight();
	  remEdge = *e;	  
	}
      }
    }
    if (max_weight > (*anE)->weight()) {
      stop = true;
      inEdge = *anE;
      outEdge = remEdge;
    }
  }
  delete(tCopy);
  bool improved = false;
  if ((inEdge != NULL) && (outEdge != NULL)) {
    tree->remove(outEdge);
    tree->addEdge(inEdge);
    tree->setWeight(tree->weight() - outEdge->weight() + inEdge->weight());
    adaptTabuLists(inEdge,outEdge);
    improved = true;
  }
  return improved;
}

int main(int argc, char *argv[])
{

  Timer initialization_timer;

  init_program(argc, argv);
  graph = new UndirectedGraph(ifile);

  GreedyHeuristic gho(graph);
  UndirectedGraph* globalBest = NULL;
  UndirectedGraph* restartBest = NULL;
  UndirectedGraph* currentSol = NULL;

  double globalBestValue = 0.0;

  for (int cardinality = cardb; cardinality <= carde; cardinality++) {
    if ((cardinality == cardb) || (cardinality == carde) || ((cardinality % cardmod) == 0)) {
      printf("begin cardinality %d\n",cardinality);
      if (tfile_is_given) {
	if (times.count(cardinality) == 1) {
	  tlimit = times[cardinality];
	}
      }
      cout << "tlimit: " << tlimit << endl;
      vector<double> results;
      vector<double> times_best_found;
      double biar = 0;
      for (int trial_counter = 1; trial_counter <= n_of_trials; trial_counter++) {
	printf("begin try %d\n",trial_counter);
	
	Timer timer;
		
	bool stop = false;
	int iter = 1;
	
	if (globalBest != NULL) {
	  delete(globalBest);
	}
	globalBest = new UndirectedGraph();
	if (restartBest != NULL) {
	  delete(restartBest);
	}
	restartBest = new UndirectedGraph();
	if (currentSol != NULL) {
	  delete(currentSol);
	}
	currentSol = new UndirectedGraph();
	gho.getGreedyTree(currentSol,cardinality);
	restartBest->copy(currentSol);
	globalBest->copy(currentSol);

	globalBestValue = globalBest->weight();
	if (trial_counter == 1) {
	  biar = globalBestValue;
	}
	else {
	  if (globalBestValue < biar) {
	    biar = globalBestValue;
	  }
	}

	initializeTabuLists();
	initializeLeafsAndNeighborhood(currentSol);

	results.push_back(globalBestValue);
	times_best_found.push_back(timer.elapsed_time(Timer::VIRTUAL));	  
	
	LSMove* bestMove = NULL;

	int nic = 1;
	int init_min = 0;
	if (((*graph).vertices.size() - cardinality) < cardinality) {
	  init_min = (*graph).vertices.size() - cardinality;
	}
	else {
	  init_min = cardinality;
	}
	int init_length;
	if (init_min < ((int)(((double)(*graph).vertices.size()) / 5.0))) {
	  init_length = init_min;
	}
	else {
	  init_length = (int)(((double)(*graph).vertices.size()) / 5.0);
	}
	in_length = init_length;
	out_length = init_length;
	int max_length = (int)(((double)(*graph).vertices.size()) / 3.0);
        int increment = ((int)((max_length - in_length) / 4.0)) + 1;
	int max_unimpr_iters = increment;
	if (max_unimpr_iters < 100) {
	  max_unimpr_iters = 200;
	}

	while (!stop) {
	  
	  if ((nic % max_unimpr_iters) == 0) {
	    if (in_length + increment > max_length) {
	      in_length = init_length;
	      out_length = init_length;
	      currentSol->clear();
	      gho.getGreedyTree(currentSol,cardinality);
	      restartBest->copy(currentSol);
	      initializeTabuLists();
	      initializeLeafsAndNeighborhood(currentSol);
	    }
	    else {
	      in_length = in_length + increment;
	      out_length = out_length + increment;
	    }
	  }

	  if (bestMove != NULL) {
	    delete(bestMove);
	  }
	  if (moveType == "first_improvement") {
	    bestMove = getFirstMove(restartBest->weight(),currentSol->weight());
	  }
	  else {
	    bestMove = getBestMove(restartBest->weight(),currentSol->weight());
	  }
	  
	  if (bestMove != NULL) {
	    
	    bool do_normal_move = true;
	    
	    if (bestMove->weight_diff > 0.0) {
	      
	      if (cycleMoves == "yes") {
		
		bool improved = applyFirstBestCycleMove(currentSol);
		
		if (improved) {
		  do_normal_move = false;
		  initializeLeafsAndNeighborhood(currentSol);
		}
	      }
	    }
	    
	    if (do_normal_move) {
	      currentSol->addVertex((bestMove->in)->lVertex);
	      currentSol->addEdge((bestMove->in)->lEdge);
	      currentSol->remove((bestMove->out)->lEdge);
	      currentSol->remove((bestMove->out)->lVertex);
	      
	      adaptLeafs(bestMove,currentSol);
	      adaptNeighborhood(bestMove,currentSol);
	      adaptTabuLists((bestMove->in)->getEdge(),(bestMove->out)->getEdge());
	      
	      delete(bestMove->in);
	      delete(bestMove->out);
	      
	      currentSol->setWeight(currentSol->weight() + (bestMove->weight_diff));
	    }

	    if (currentSol->weight() < globalBest->weight()) {
	      globalBest->copy(currentSol);
	      globalBestValue = globalBest->weight();
	      if (globalBestValue < biar) {
		biar = globalBestValue;
	      }
	      printf("best %f\ttime %f\n",globalBestValue,timer.elapsed_time(Timer::VIRTUAL));
	      results[trial_counter-1] = globalBestValue;
	      times_best_found[trial_counter-1] = timer.elapsed_time(Timer::VIRTUAL);
	    }
	    else {
	      //printf("best %f\tcurrent %f\ttime %f\n",globalBestValue,currentSol->weight(),timer.elapsed_time(Timer::VIRTUAL));
	    }
	    if (currentSol->weight() < restartBest->weight()) {
	      restartBest->copy(currentSol);
	      nic = 1;
	      in_length = init_length;
	      out_length = init_length;
	      cutTabuLists();
	    }
	    else {
	      nic = nic + 1;
	    }
	  }
	  else {
	    currentSol->clear();
	    gho.getGreedyTree(currentSol,cardinality);
	    restartBest->copy(currentSol);
	    initializeTabuLists();
	    initializeLeafsAndNeighborhood(currentSol);
	  }

	  iter = iter + 1;
	  if (tfile_is_given) {
	    if (timer.elapsed_time(Timer::VIRTUAL) > tlimit) {
	      stop = true;
	    }	    
	  }
	  else {
	    if (time_limit_given && iter_limit_given) {
	      if ((timer.elapsed_time(Timer::VIRTUAL) > tlimit) || (iter > n_of_iter)) {
		stop = true;
	      }
	    }
	    else {
	      if (time_limit_given) {
		if (timer.elapsed_time(Timer::VIRTUAL) > tlimit) {
		  stop = true;
		}
	      }
	      else {
		if (iter > n_of_iter) {
		  stop = true;
		}
	      }
	    }
	  }
	}
	
		  delete(bestMove);
		  printf("end try %d\n",trial_counter);
      }      
      double r_mean = 0.0;
      double t_mean = 0.0;
      for (int i = 0; i < results.size(); i++) {
	r_mean = r_mean + results[i];
	t_mean = t_mean + times_best_found[i];
      }
      r_mean = r_mean / ((double)results.size());
      t_mean = t_mean / ((double)times_best_found.size());
      double rsd = 0.0;
      double tsd = 0.0;
      for (int i = 0; i < results.size(); i++) {
	rsd = rsd + pow(results[i]-r_mean,2.0);
	tsd = tsd + pow(times_best_found[i]-t_mean,2.0);
      }
      rsd = rsd / ((double)(results.size()-1.0));
      if (rsd > 0.0) {
	rsd = sqrt(rsd);
      }
      tsd = tsd / ((double)(times_best_found.size()-1.0));
      //cout << "tsd: " << tsd << endl;
      if (tsd > 0.0) {
	tsd = sqrt(tsd);
	//cout << "sqrt(tsd): " << tsd << endl;
      }
      printf("statistics\t%d\t%g\t%f\t%f\t%f\t%f\n",cardinality,biar,r_mean,rsd,t_mean,tsd);
      printf("end cardinality %d\n",cardinality);
		
		// eturetken 18.09.09. Write the best MST for this cardinality to the file.
		////////////////////////////////////////////////////////////////////
		if( mstfile_is_given )
		{
			string MSTFile(mstfile);
			globalBest->Write2File(concatIntToString(MSTFile, cardinality) + ".mst");
		}
		////////////////////////////////////////////////////////////////////
		
    }
  }
  
  /* delete all the vertices and edges */
  for (list<Vertex*>::iterator i = ((*graph).vertices).begin(); i != ((*graph).vertices).end(); i++) {
    delete (*i);
  }
  for (list<Edge*>::iterator i = ((*graph).edges).begin(); i != ((*graph).edges).end(); i++) {
    delete (*i);
  }	
  delete(graph);
  delete(rg);
  return EXIT_SUCCESS;
}
