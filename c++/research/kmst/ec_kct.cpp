/***************************************************************************
                          ec_kct.cpp  -  description
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
#include "TabuSearch.h"
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

/* possible values for ls: "leafs", "tsleafs", "none" */
string ls = "leafs";

/* possible values for elite_action: "yes", "no" */
string elite_action = "yes";

time_t t;
map<int,double> times;
char ifile[LINE_BUF_LEN];
char tfile[LINE_BUF_LEN];
char oput[5];
char ls_cstring[10];
bool ifile_is_given = false;
bool tfile_is_given = false;
bool cardb_is_given = false;
bool carde_is_given = false;
bool iter_limit_given = false;
bool time_limit_given = false;
bool help_needed = false;

/* output can be minimal (min) or detailed (det) */
string output = "det";

bool comments = false;

list<UndirectedGraph*> generation;


inline int stoi(string &s) {

  return atoi(s.c_str());
}

inline double stof(string &s) {

  return atof(s.c_str());
}

bool graphCompare(UndirectedGraph* t1, UndirectedGraph* t2) {

  return (t1->weight() < t2->weight());
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

  while(true) {
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
    times[card] = time;
  }
  fclose(timesFile);
}

void init_program(const int& argc, char* argv[]) {

  /* initialize random generator */
  rg = new Random((unsigned) time(&t));

  int iarg=1;

  while (iarg < argc)
  {
    if (strcmp(argv[iarg],"-i")==0) {
      strcpy(ifile,argv[++iarg]);
      ifile_is_given = true;
    }
    else if (strcmp(argv[iarg],"-tfile")==0) {
      strcpy(tfile,argv[++iarg]);
      readTimesFile();
      tfile_is_given = true;
    }
    else if (strcmp(argv[iarg],"-ls")==0) {
      strcpy(ls_cstring,argv[++iarg]);
      if (strcmp(ls_cstring,"leafs")==0) {
	ls = "leafs";
      }
      else {
	if (strcmp(ls_cstring,"tsleafs")==0) {
	  ls = "tsleafs";
	}
	else {
	  ls = "none";
	}
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
    else if (strcmp(argv[iarg],"-eliteaction")==0) {
      strcpy(oput,argv[++iarg]);
      if (strcmp(oput,"yes") == 0) {
	elite_action = "yes";
      }
      else {
	elite_action = "no";
      }
    }
    iarg++;
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

UndirectedGraph* getMinTree() {

  double minW;
  UndirectedGraph* result;
  bool started = false;
  for (list<UndirectedGraph*>::iterator i = generation.begin(); i != generation.end(); i++) {
    if (started == false) {
      started = true;
      result = (*i);
      minW = result->weight();
    }
    else {
      if (((*i)->weight()) < minW) {
	result = (*i);
	minW = result->weight();
      }
    }
  }
  return result;	
}

double getAverageFitness() {
	
  double av = 0.0;
  for (list<UndirectedGraph*>::iterator i = generation.begin(); i != generation.end(); i++) {
    av = av + (*i)->weight();
  }
  return (av/((double)(generation.size())));
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

UndirectedGraph* getMate(UndirectedGraph* aTree) {

  list<UndirectedGraph*> posMates;
  for (list<UndirectedGraph*>::iterator aT = generation.begin(); aT != generation.end(); aT++) {
    if ((!(treesDisjunct(aTree,(*aT)))) && (!(treesEqual(aTree,*aT)))) {
      posMates.push_back(*aT);
    }
  }
  UndirectedGraph* retTree = NULL;
  if (posMates.size() > 0) {
    double rand = rg->next();
    double sum = 0.0;
    list<UndirectedGraph*>::iterator ag;
    for (ag = posMates.begin(); ag != posMates.end(); ag++) {
      sum = sum + ((*ag)->weight());
    }
    double wheel = 0.0;
    ag = posMates.begin();
    int count = 0;
    while ((wheel < rand) && (ag != posMates.end())) {
      wheel = wheel + (((*ag)->weight()) / sum);
      ag++;
      count = count + 1;
    }
    ag--;
    retTree = (*ag);
  }
  posMates.clear();
  return retTree;
}

UndirectedGraph* intersectionCrossover(UndirectedGraph* t1, UndirectedGraph* t2, list<Edge*>* iset, UndirectedGraph* ug) {

  GreedyHeuristic gho(graph);
  return (gho.getICTree(t1,t2,iset,ug));
}

UndirectedGraph* unionCrossover(UndirectedGraph* t1, UndirectedGraph* t2, list<Edge*>* iset, UndirectedGraph* ug) {

  GreedyHeuristic gho(graph);
  return (gho.getUCTree(t1,t2,iset,ug));
}

UndirectedGraph* uniteOnCommonBase(UndirectedGraph* t1, UndirectedGraph* t2, list<Edge*>* is) {

  UndirectedGraph* ugh = new UndirectedGraph();
  for (list<Vertex*>::iterator v = ((*t1).vertices).begin(); v != ((*t1).vertices).end(); v++) {
    ugh->addVertex(*v);
  }
  for (list<Vertex*>::iterator v = ((*t2).vertices).begin(); v != ((*t2).vertices).end(); v++) {
    if (!(ugh->contains(*v))) {
      ugh->addVertex(*v);
    }
  }
  for (list<Edge*>::iterator e = ((*t1).edges).begin(); e != ((*t1).edges).end(); e++) {
    ugh->addEdge(*e);
  }
  for (list<Edge*>::iterator e = ((*t2).edges).begin(); e != ((*t2).edges).end(); e++) {
    if (!(ugh->contains(*e))) {
      ugh->addEdge(*e);
    }
    else {
      is->push_back(*e);
    }
  }
  return ugh;
}

void crossover() {

  list<UndirectedGraph*> newGeneration;
  list<UndirectedGraph*> toDelete;
  for (list<UndirectedGraph*>::iterator aTree = generation.begin(); aTree != generation.end(); aTree++) {
    UndirectedGraph* mate = getMate(*aTree);
    if (mate != NULL) {
      list<Edge*>* iset = new list<Edge*>;
      UndirectedGraph* ug = uniteOnCommonBase(*aTree,mate,iset);
      UndirectedGraph* icResult = intersectionCrossover(*aTree,mate,iset,ug);
      UndirectedGraph* ucResult = unionCrossover(*aTree,mate,iset,ug);
      delete(ug);
      (*iset).clear();
      delete(iset);
      bool push_back = true;
      list<UndirectedGraph*>::iterator ngm;
      if ((icResult->weight()) < (ucResult->weight())) {
	delete(ucResult);
	if ((icResult->weight()) < ((*aTree)->weight())) {
	  ngm = newGeneration.begin();
	  while ((ngm != newGeneration.end()) && (push_back == true)) {
	    if (treesEqual(icResult,(*ngm))) {
	      push_back = false;
	    }
	    ngm++;
	  }
	  if (push_back == true) {
	    newGeneration.push_back(icResult);
	  }
	  else {
	    delete(icResult);
	  }
	  toDelete.push_back(*aTree);
	}
	else {
	  delete(icResult);
	  ngm = newGeneration.begin();
	  while ((ngm != newGeneration.end()) && (push_back == true)) {
	    if (treesEqual((*aTree),(*ngm))) {
	      push_back = false;
	    }
	    ngm++;
	  }
	  if (push_back == true) {
	    newGeneration.push_back(*aTree);
	  }
	  else {
	    toDelete.push_back(*aTree);
	  }
	}
      }
      else {
	delete(icResult);
	if ((ucResult->weight()) < ((*aTree)->weight())) {
	  ngm = newGeneration.begin();
	  while ((ngm != newGeneration.end()) && (push_back == true)) {
	    if (treesEqual(ucResult,(*ngm))) {
	      push_back = false;
	    }
	    ngm++;
	  }
	  if (push_back == true) {
	    newGeneration.push_back(ucResult);
	  }
	  else {
	    delete(ucResult);
	  }
	  toDelete.push_back(*aTree);
	}
	else {
	  delete(ucResult);
	  ngm = newGeneration.begin();
	  while ((ngm != newGeneration.end()) && (push_back == true)) {
	    if (treesEqual((*aTree),(*ngm))) {
	      push_back = false;
	    }
	    ngm++;
	  }
	  if (push_back == true) {
	    newGeneration.push_back(*aTree);
	  }
	  else {
	    toDelete.push_back(*aTree);
	  }
	}
      }
    }
  }
  for (list<UndirectedGraph*>::iterator aTree = toDelete.begin(); aTree != toDelete.end(); aTree++) {
    delete(*aTree);
  }
  generation.clear();
  for (list<UndirectedGraph*>::iterator aTree = newGeneration.begin(); aTree != newGeneration.end(); aTree++) {
    generation.push_back(*aTree);
  }
  generation.sort(graphCompare);
  newGeneration.clear();
}

void mutation() {

  list<UndirectedGraph*> newGeneration;
  list<UndirectedGraph*>::iterator ngm;
  int firstcount=0;
  for (list<UndirectedGraph*>::iterator aTree = generation.begin(); aTree != generation.end(); aTree++) {
    double w = (*aTree)->weight();
    UndirectedGraph* aT = (*aTree);
    if (ls != "none") {
      if (ls == "leafs") {
	LocalSearch lso(graph,aT);
	lso.run(ls);
      }
      else {
	TabuSearch tso(graph,aT);
	tso.run("first_improvement",2*((*aT).edges.size()));
      }
    }
    if ((firstcount==0) && (elite_action == "yes")) {
      TabuSearch tso(graph,aT);
      tso.run("first_improvement",2*((*aT).edges.size()));
      firstcount=1;
    }
    bool push_back = true;
    ngm = newGeneration.begin();
    while ((ngm != newGeneration.end()) && (push_back == true)) {
      if (treesEqual(aT,(*ngm))) {
	push_back = false;
      }
      ngm++;
    }
    if (push_back == true) {
      aT->setWeight(weightOfSolution(aT));
      newGeneration.push_back(aT);
    }
    else {
      delete(aT);
    }
  }
  generation.clear();
  for (list<UndirectedGraph*>::iterator aTree = newGeneration.begin(); aTree != newGeneration.end(); aTree++) {
    generation.push_back(*aTree);
  }  
}

int random(int start, int end) {

  int ret_val = (int)(((double)start) + (( ((double)end) - ((double)start) + 1 ) * rg->next()));
  return ret_val;
}

map<Edge*,double>* generatePheromoneMap() {

  map<Edge*,double>* aMap = new map<Edge*,double>;
  for (list<Edge*>::iterator e = (*graph).edges.begin(); e != (*graph).edges.end(); e++) {
    (*aMap)[*e] = 1.0;
  }
  for (list<UndirectedGraph*>::iterator aGit = generation.begin(); aGit != generation.end(); aGit++) {
    for (list<Edge*>::iterator anEit = ((*aGit)->edges).begin(); anEit != ((*aGit)->edges).end(); anEit++) {
      (*aMap)[*anEit] = (*aMap)[*anEit] + 1.0;
    }
  }
  return aMap;
}

int main(int argc, char *argv[])
{

  cout << endl;
  Timer initialization_timer;

  init_program(argc, argv);
  graph = new UndirectedGraph(ifile);
  GreedyHeuristic gho(graph);
  double averageFitness;
  double newAverageFitness;
  UndirectedGraph* bestSoFar = new UndirectedGraph();
  double best_weightSoFar = 0.0;

  UndirectedGraph* ib = NULL;
  UndirectedGraph* bsf = NULL;

  vector<Edge*> vec;

  for (int cardinality = cardb; cardinality <= carde; cardinality++) {
    if ((cardinality == cardb) || (cardinality == carde) || ((cardinality % cardmod) == 0)) {
      printf("begin cardinality %d\n",cardinality);
      if (tfile_is_given) {
	if (times.count(cardinality) == 1) {
	  tlimit = times[cardinality];
	}
      }
      vector<double> results;
      vector<double> times_best_found;
      double biar = 0.0;
      for (int trial_counter = 1; trial_counter <= n_of_trials; trial_counter++) {
	printf("begin try %d\n",trial_counter);
	
	Timer timer;
	
	int gen_size = (int)(((double)((*graph).edges).size()) / ((double)cardinality));
	if (gen_size < 50) {
	  gen_size = 50;
	}
	if (gen_size > 200) {
	  gen_size = 200;
	}
	
	bool stop = false;
	int iter = 1;
		  
	for (list<UndirectedGraph*>::iterator aT = generation.begin(); aT != generation.end(); aT++) {
	  delete(*aT);
	}
	  
	generation.clear();
	
	/* begin produce initial generation */
	
	for (list<Edge*>::iterator se = ((*graph).edges).begin(); se != ((*graph).edges).end(); se++) {
	  vec.push_back(*se);
	}
	int random_num = 0;
	for (int i = 0; i < gen_size; i++) {
	  random_num = random(0,((*graph).edges).size()-1);
	  UndirectedGraph* aT = new UndirectedGraph();
	  gho.getRandomTree(aT,cardinality,vec[random_num]);
	  generation.push_back(aT);	
	}
	
	/* end produce initial generation */
	
	averageFitness = getAverageFitness();
	newAverageFitness = averageFitness - 1.0;
	bsf = getMinTree();
	bestSoFar->copy(bsf);
	best_weightSoFar = bestSoFar->weight();
	if (trial_counter == 1) {
	  biar = best_weightSoFar;
	}
	else {
	  if (best_weightSoFar < biar) {
	    biar = best_weightSoFar;
	  }
	}
	results.push_back(best_weightSoFar);
	times_best_found.push_back(timer.elapsed_time(Timer::VIRTUAL));	  
	
	while (!stop) {
	  
	  averageFitness = newAverageFitness;
	  
	  crossover();
	  
	  if (generation.size() > 0) {	    
	    mutation();
	    list<UndirectedGraph*> newGeneration;
	    list<UndirectedGraph*> toDelete;
	    for (list<UndirectedGraph*>::iterator t = generation.begin(); t != generation.end(); t++) {
	      (*t)->age = (*t)->age + 1;
	      if ((*t)->age > 10) {
		toDelete.push_back(*t);
	      }
	      else {
		newGeneration.push_back(*t);
	      }
	    }
	    for (list<UndirectedGraph*>::iterator aTree = toDelete.begin(); aTree != toDelete.end(); aTree++) {
	      delete(*aTree);
	    }
	    generation.clear();
	    for (list<UndirectedGraph*>::iterator aTree = newGeneration.begin(); aTree != newGeneration.end(); aTree++) {
	      generation.push_back(*aTree);
	    }
	    newGeneration.clear();
	  }
	  
	  if (generation.size() < gen_size) {
	    for (int ind = 0; ind < (gen_size - generation.size()); ind++) {
	      random_num = random(0,((*graph).edges).size()-1);
	      UndirectedGraph* aT = new UndirectedGraph();
	      gho.getRandomTree(aT,cardinality,vec[random_num]);
	      generation.push_back(aT);
	    }
	  }
	  
	  newAverageFitness = getAverageFitness();
	  ib = getMinTree();
	  double weight = ib->weight();
	  if (weight < best_weightSoFar) {
	    printf("best %f\ttime %f\n",weight,timer.elapsed_time(Timer::VIRTUAL));
	    results[trial_counter-1] = weight;
	    times_best_found[trial_counter-1] = timer.elapsed_time(Timer::VIRTUAL);
	    best_weightSoFar = weight;
	    bestSoFar->copy(ib);
	    if (best_weightSoFar < biar) {
	      biar = best_weightSoFar;
	    }
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
	
	double bestSolVal = bestSoFar->weight();
	double virt_time = timer.elapsed_time(Timer::VIRTUAL);
	double real_time = timer.elapsed_time(Timer::REAL);
	bestSoFar->clear();
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
      if (tsd > 0.0) {
	tsd = sqrt(tsd);
      }
      printf("statistics\t%d\t%g\t%f\t%f\t%f\t%f\n",cardinality,biar,r_mean,rsd,t_mean,tsd);
      printf("end cardinality %d\n",cardinality);
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
  delete(bestSoFar);
  return EXIT_SUCCESS;
}
