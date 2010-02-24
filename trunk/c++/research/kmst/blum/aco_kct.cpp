/***************************************************************************
                          aco_kct.cpp  -  description
                             -------------------
    begin                : Tue Jan 15 15:46:28 CET 2002
    copyright            : (C) 2002 by Christian Blum
    email                : cblum@ulb.ac.be
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
#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <stdio.h>
#include <math.h>
#include "Timer.h"
#include "Random.h"
#include <string>
#include "utilstuff.h"
#include "KCT_Ant.h"
#include "UndirectedGraph.h"
#include "Vertex.h"
#include "Edge.h"

#define LINE_BUF_LEN 512
#define PROG_ID_STR "Ant colony optimization for the k-cardinality tree problem, V0.1"
#define CALL_SYNTAX_STR "Parameter problems !!"
#define CALL_MISSFILE_STR "You have to specify an input file (i.e., -i instance.dat)."
#define CALL_MISSCARD_STR "You have to specify a cardinality (i.e., -cardb 10)."


fstream fout;

string output = "min";
int n_of_iter = 1000;
double time_limit = 100.0;
bool time_limit_given = false;
bool iter_limit_given = false;
bool input_file_given = false;
bool mstfile_is_given = false;
bool output_file_given = false;

int cardmod =  1;
map<int,double> times;

int n_of_trials = 1;
char i_file[LINE_BUF_LEN];
char mst_file[LINE_BUF_LEN];
char o_file[LINE_BUF_LEN];
char tfile[LINE_BUF_LEN];

UndirectedGraph* graph;

long int n_of_vertices;
long int n_of_edges;

long int cardb;
long int carde;

int cardinality;

/* parameter (partly initialized to a default value) */
string file;

long int n_of_ants = 15;

double l_rate = 0.1;
double tau_min = 0.001;
double tau_max = 0.999;

/* possible values for daemon_action: "leafs", "tsleafs", "none" */
string daemon_action = "leafs";

/* possible values for elicte_action: "no", "yes" */
string elite_action = "yes";

bool cardb_is_given = false;
bool carde_is_given = false;

bool tfile_is_given = false;

map<Edge*,double>* pheromone;

list<KCT_Ant*> ants;

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
    //cout << "card " << card << "\ttime" << time << endl;
    times[card] = time;
  }
  fclose(timesFile);
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
	else if (strcmp(argv[iarg],"-m")==0) {
		strcpy(mst_file,argv[++iarg]);
		mstfile_is_given = true;
    }
    else if (strcmp(argv[iarg],"-tfile")==0) {
      strcpy(tfile,argv[++iarg]);
      readTimesFile();
      tfile_is_given = true;
    }
    else if (strcmp(argv[iarg],"-t")==0) { 
      time_limit=atof(argv[++iarg]);
      time_limit_given = true;
    }
    else if (strcmp(argv[iarg],"-maxiter")==0) {
      n_of_iter=atoi(argv[++iarg]);
      iter_limit_given = true;
    }
    else if (strcmp(argv[iarg],"-n")==0) {
      n_of_trials=atoi(argv[++iarg]);
    }    
    else if (strcmp(argv[iarg],"-o")==0) {
      strcpy(o_file,argv[++iarg]);
      output_file_given = true;
    }
    else if (strcmp(argv[iarg],"-output")==0) {
      if (strcmp(argv[++iarg],"det")==0) {
	output = "det";
      }
      else {
	output = "min";
      }
    }
    else if (strcmp(argv[iarg],"-eliteaction")==0) {
      if (strcmp(argv[++iarg],"yes")==0) {
        elite_action = "yes";
      }
      else {
	elite_action = "no";
      }
    }
    else if (strcmp(argv[iarg],"-ls")==0) {
      if (strcmp(argv[++iarg],"leafs")==0) {
	daemon_action = "leafs";
      }
      else {
	if (strcmp(argv[iarg],"tsleafs")==0) {
	  daemon_action = "tsleafs";
	}
	else {
	  daemon_action = "none";
	}
      }
    }
    else if (strcmp(argv[iarg],"-nants")==0) {
      n_of_ants = atoi(argv[++iarg]);
    }
    else if (strcmp(argv[iarg],"-cardmod")==0) {
      cardmod = atoi(argv[++iarg]);
    }
    else if (strcmp(argv[iarg],"-lrate")==0) {
      l_rate = atof(argv[++iarg]);
    }
    else if (strcmp(argv[iarg],"-cardb")==0) {
      cardb=atoi(argv[++iarg]);
      cardb_is_given = true;
    }
    else if (strcmp(argv[iarg],"-carde")==0) {
      carde=atoi(argv[++iarg]);
      carde_is_given = true;
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
  if (output_file_given == true) {
    fout.open(o_file,std::ios::out);
    if (!fout.is_open()) {
      cout << "Problems with writing to file. Using standard output." << endl;
      output_file_given = false;
    }
  }
}

void init_pheromone_trail() {

  pheromone = new map<Edge*,double>;
  for (list<Edge*>::iterator anEdge = (graph->edges).begin(); anEdge != (graph->edges).end(); anEdge++) {
    (*pheromone)[*anEdge] = 0.5;
  }
}

void reset_pheromone_trail() {

  for (list<Edge*>::iterator anEdge = (graph->edges).begin(); anEdge != (graph->edges).end(); anEdge++) {
    (*pheromone)[*anEdge] = 0.5;
  }
}

double getBestSolutionInIterationValue() {
	
  double b = 0;
  bool started = false;
  for (list<KCT_Ant*>::iterator i = ants.begin(); i != ants.end(); i++) {
    double help = (*i)->getCurrentValue();
    if (started == true) {
      if (help < b) {
	b = help;
      }
    }
    else {
      b = help;
      started = true;
    }
  }	
  return b;
}

double getBestDaemonSolutionInIterationValue() {
	
  double b = 0;
  bool started = false;
  for (list<KCT_Ant*>::iterator i = ants.begin(); i != ants.end(); i++) {
    double help = (*i)->getCurrentDaemonValue();
    if (started == true) {
      if (help < b) {
	b = help;
      }
    }
    else {
      b = help;
      started = true;
    }
  }	
  return b;
}

double getBestSolutionValue() {
	
  double b = 0;
  bool started = false;
    for (list<KCT_Ant*>::iterator i = ants.begin(); i != ants.end(); i++) {
      double help = (*i)->getBestFoundValue();
      if (started == true) {
	if (help < b) {
	  b = help;
	}
      }
      else {
	b = help;
	started = true;
      }
  }	
  return b;
}

UndirectedGraph* getBestSolution() {

  double b = 0;
  bool started = false;
  UndirectedGraph* result;
  for (list<KCT_Ant*>::iterator i = ants.begin(); i != ants.end(); i++) {
    double help = (*i)->getBestFoundValue();
    if (started == true) {
      if (help < b) {
	b = help;
	result = (*i)->getBestFoundSolution();
      }
    }
    else {
      b = help;
      started = true;
      result = (*i)->getBestFoundSolution();
    }
  }
  UndirectedGraph* retVal = new UndirectedGraph(result);
  return retVal;
}

UndirectedGraph* getBestSolutionInIteration() {
	
  double b = 0;
  UndirectedGraph* bS;
  bool started = false;
  for (list<KCT_Ant*>::iterator i = ants.begin(); i != ants.end(); i++) {
    double help = (*i)->getCurrentValue();
    if (started == true) {
      if (help < b) {
	b = help;
	bS = (*i)->getCurrentSolution();
      }
    }
    else {
      b = help;
      started = true;
      bS = (*i)->getCurrentSolution();
    }
  }
  UndirectedGraph* retVal = new UndirectedGraph(bS);
  return retVal;
}

UndirectedGraph* getBestDaemonSolutionInIteration() {
	
  double b = 0;
  UndirectedGraph* bS;
  bool started = false;
  for (list<KCT_Ant*>::iterator i = ants.begin(); i != ants.end(); i++) {
    double help = (*i)->getCurrentDaemonValue();
    if (started == true) {
      if (help < b) {
	b = help;
	bS = (*i)->getCurrentDaemonSolution();
      }
    }
    else {
      b = help;
      started = true;
      bS = (*i)->getCurrentDaemonSolution();
    }
  }
  UndirectedGraph* retVal = new UndirectedGraph(bS);
  return retVal;
}

KCT_Ant* getBestDaemonAntInIteration() {
	
  double b = 0;
  KCT_Ant* bS;
  bool started = false;
  for (list<KCT_Ant*>::iterator i = ants.begin(); i != ants.end(); i++) {
    double help = (*i)->getCurrentDaemonValue();
    if (started == true) {
      if (help < b) {
	b = help;
	bS = (*i);
      }
    }
    else {
      b = help;
      started = true;
      bS = (*i);
    }
  }
  return bS;
}

double getSolutionAverageInIteration() {
	
  double av = 0.0;
  for (list<KCT_Ant*>::iterator i = ants.begin(); i != ants.end(); i++) {
    av = av + ((double)(*i)->getCurrentDaemonValue());
  }
  return (av/((double)n_of_ants));
}

map<Edge*,double>* translate_solution(UndirectedGraph* aTree) {

  map<Edge*,double>* ts = new map<Edge*,double>;
  for (list<Edge*>::iterator e = (graph->edges).begin(); e != (graph->edges).end(); e++) {
    if (aTree->contains(*e)) {
      (*ts)[*e] = 1.0;
    }
    else {
      (*ts)[*e] = 0.0;
    }
  }
  return ts;
}


double get_cf(UndirectedGraph* aTree) {
  
  double all = 0.0;
  double sum = 0.0;
  for (list<Edge*>::iterator e = (aTree->edges).begin(); e != (aTree->edges).end(); e++) {
    all = all + 1.0;
    sum = sum + (*pheromone)[*e];
  }
  return (sum / (((double)cardinality) * tau_max));
}

int main( int argc, char **argv ) {

  if ( argc < 2 ) {
    cout << "something wrong" << endl;
    exit(1);
  }
  else {
    read_parameters(argc,argv);
  }
  Timer initialization_timer;

  rnd = new Random((unsigned) time(&t));

  graph = new UndirectedGraph(i_file);

  init_pheromone_trail();

  register int i = 0;
  register int j = 0;
  register int k = 0;
  register int b = 0;

  cout << endl;
  for (int card_counter = cardb; card_counter <= carde; card_counter++) {
    cardinality = card_counter;
    if ((cardinality == cardb) || (cardinality == carde) || ((cardinality % cardmod) == 0)) {
      printf("begin cardinality %d\n",cardinality);

      if (tfile_is_given) {
	if (times.count(cardinality) == 1) {
	  time_limit = times[cardinality];
	}
      }
      vector<double> results;
      vector<double> times_best_found;
      double biar = 0.0;

      int n_of_ants = (int)(((double)((*graph).edges).size()) / ((double)cardinality));
      if (n_of_ants < 15) {
	n_of_ants = 15;
      }
      if (n_of_ants > 50) {
	n_of_ants = 50;
      }

      if (ants.size() > 0) {
	for (list<KCT_Ant*>::iterator anAnt = ants.begin(); anAnt != ants.end(); anAnt++) {
	  delete(*anAnt);
	}
      }
      ants.clear();

      for (i = 0; i < n_of_ants; i++) {
	KCT_Ant* ant = new KCT_Ant();
	ant->initialize(graph,rnd,pheromone,daemon_action);
	ants.push_back(ant);
      }
      
      for (int trial_counter = 1; trial_counter <= n_of_trials; trial_counter++) {
	printf("begin try %d\n",trial_counter);

	UndirectedGraph* best = NULL;
	UndirectedGraph* newSol = NULL;
	UndirectedGraph* restartBest = NULL;

	double ib_weight = 0.0;
	double rb_weight = 0.0;
	double gb_weight = 0.0;

	Timer timer;
	
	if ((!(card_counter == cardb)) || (!(trial_counter == 1))) {
	  reset_pheromone_trail();
	  for (list<KCT_Ant*>::iterator ant = ants.begin(); ant != ants.end(); ant++) {
	    (*ant)->restart_reset();
	  }
	}
	
	int iter = 1;
	double cf = 0.0;
	bool restart = false;
	bool program_stop = false;
	bool bs_update = false;
	
	while (program_stop == false) {
	  
	  for (list<KCT_Ant*>::iterator ant = ants.begin(); ant != ants.end(); ant++) {
	    for (k = 0; k < cardinality; k++) {
	      (*ant)->step(0.8);
	    }
	    (*ant)->evaluate();
	  }
	  
	  if (!(newSol == NULL)) {
	    delete(newSol);
	  }
	  if (elite_action == "no") {
	    newSol = getBestDaemonSolutionInIteration();
	  }
	  else {
	    KCT_Ant* bestAnt = getBestDaemonAntInIteration();
	    bestAnt->eliteAction();
	    newSol = new UndirectedGraph();
	    newSol->copy(bestAnt->getCurrentDaemonSolution());
	  }

	  if (iter == 1) {
	    best = new UndirectedGraph();
	    printf("best %f\ttime %f\n",newSol->weight(),timer.elapsed_time(Timer::VIRTUAL));
	    best->copy(newSol);
	    restartBest = new UndirectedGraph(newSol);
	    results.push_back(newSol->weight());
	    times_best_found.push_back(timer.elapsed_time(Timer::VIRTUAL));
	    if (trial_counter == 1) {
	      biar = newSol->weight();
	    }
	    else {
	      if (newSol->weight() < biar) {
		biar = newSol->weight();
	      }
	    }
	  }
	  else {
	    if (restart) {
	      restart = false;
	      restartBest->copy(newSol);
	      if (newSol->weight() < best->weight()) {
		printf("best %f\ttime %f\n",newSol->weight(),timer.elapsed_time(Timer::VIRTUAL));
		best->copy(newSol);
		results[trial_counter-1] = newSol->weight();
		times_best_found[trial_counter-1] = timer.elapsed_time(Timer::VIRTUAL);
		if (newSol->weight() < biar) {
		  biar = newSol->weight();
		}
	      }
	    }
	    else {
	      if (newSol->weight() < restartBest->weight()) {
		restartBest->copy(newSol);
	      }
	      if (newSol->weight() < best->weight()) {
		printf("best %f\ttime %f\n",newSol->weight(),timer.elapsed_time(Timer::VIRTUAL));
		best->copy(newSol);
		results[trial_counter-1] = newSol->weight();
		times_best_found[trial_counter-1] = timer.elapsed_time(Timer::VIRTUAL);
		if (newSol->weight() < biar) {
		  biar = newSol->weight();
		}
	      }
	    }
	  }

	  cf = get_cf(newSol);
	  //cout << "cf: " << cf << endl;
	  
	  if (bs_update && (cf > 0.99)) {
	    //cout << "doing restart" << endl;
	    bs_update = false;
	    restart = true;
	    cf = 0.0;
	    reset_pheromone_trail();
	  }
	  else {

	    if (cf > 0.99) {
	      bs_update = true;
	    }

	    if (!bs_update) {
	      if (cf < 0.7) {
		l_rate = 0.15;
		ib_weight = 2.0/3.0;
		rb_weight = 1.0/3.0;
		gb_weight = 0.0;
	      }
	      if ((cf >= 0.7) && (cf < 0.95)) {
		l_rate = 0.1;
		ib_weight = 1.0 / 3.0;
		rb_weight = 2.0 / 3.0;
		gb_weight = 0.0;
	      }
	      if (cf >= 0.95) {
		l_rate = 0.05;
		ib_weight = 0.0;
		rb_weight = 1.0;
		gb_weight = 0.0;
	      }
	    }
	    else {
	      // if bs_update = TRUE we use the best_so_far solution for updating the pheromone values
	      l_rate = 0.1;
	      ib_weight = 0.0;
	      rb_weight = 0.0;
	      gb_weight = 1.0;
	    }

	    map<Edge*,double>* trans_best = translate_solution(best);
	    map<Edge*,double>* trans_newSol = translate_solution(newSol);
	    map<Edge*,double>* trans_restartBest = translate_solution(restartBest);
	    map<Edge*,double>* new_pd = new map<Edge*,double>;
	    for (list<Edge*>::iterator e = (graph->edges).begin(); e != (graph->edges).end(); e++) {
	      (*new_pd)[*e] = 0.0;
	      (*new_pd)[*e] = (*new_pd)[*e] + (ib_weight * (*trans_newSol)[*e]) + (rb_weight * (*trans_restartBest)[*e]) + (gb_weight * (*trans_best)[*e]);
	    }
	    for (list<Edge*>::iterator e = (graph->edges).begin(); e != (graph->edges).end(); e++) {
	      (*pheromone)[*e] = (*pheromone)[*e] + (l_rate * ((*new_pd)[*e] - (*pheromone)[*e]));
	      if ((*pheromone)[*e] > tau_max) {
		(*pheromone)[*e] = tau_max;
	      }
	      if ((*pheromone)[*e] < tau_min) {
		(*pheromone)[*e] = tau_min;
	      }
	    }
	    delete(trans_best);
	    delete(trans_newSol);
	    delete(trans_restartBest);
	    delete(new_pd);
	  }

	  for (list<KCT_Ant*>::iterator i = ants.begin(); i != ants.end(); i++) {
	    (*i)->reset();
	  }
	  
	  iter = iter + 1;
	  
	  if (tfile_is_given) {
	    if (timer.elapsed_time(Timer::VIRTUAL) > time_limit) {
	      program_stop = true;
	    }	    
	  }
	  else {
	    if (time_limit_given && iter_limit_given) {
	      if ((timer.elapsed_time(Timer::VIRTUAL) > time_limit) || (iter > n_of_iter)) {
		program_stop = true;
	      }
	    }
	    else {
	      if (time_limit_given) {
		if (timer.elapsed_time(Timer::VIRTUAL) > time_limit) {
		  program_stop = true;
		}
	      }
	      else {
		if (iter > n_of_iter) {
		  program_stop = true;
		}
	      }
	    }
	  }
	}
	printf("end try %d\n",trial_counter);
		  
	// eturetken 18.09.09. Write the best MST for this cardinality to the file.
	////////////////////////////////////////////////////////////////////
	if( mstfile_is_given )
	{
	  string MSTFile(mst_file);
	  best->Write2File(concatIntToString(MSTFile, card_counter) + ".mst");
	}
	////////////////////////////////////////////////////////////////////
		  
	delete(best);
	delete(restartBest);
	delete(newSol);
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
      if (output_file_given == true) {
	fout << cardinality << "\t" << r_mean << "\t" << rsd << "\t" << t_mean << "\t" << tsd << endl;
      }
      else {
	printf("statistics\t%d\t%g\t%f\t%f\t%f\t%f\n",cardinality,biar,r_mean,rsd,t_mean,tsd);
      }
      printf("end cardinality %d\n",cardinality);
    }
  }
  if (output_file_given == true) {
    fout.close();
  }
  delete(graph);
  delete rnd;
}
