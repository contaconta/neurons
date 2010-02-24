#ifndef SVMSTEERABLEFEATURES3D_H_
#define SVMSTEERABLEFEATURES3D_H_

#include "MatDataSet.h"
#include "TwoClassFormat.h"
#include "ClassMeasurer.h"
#include "MSEMeasurer.h"
#include "QCTrainer.h"
#include "CmdLine.h"
#include "Random.h"
#include "SVMRegression.h"
#include "SVMClassification.h"
#include "KFold.h"
#include "DiskXFile.h"
#include "ClassFormatDataSet.h"
#include "MeanVarNorm.h"
#include "Timer.h"
#include <fstream>
#include <string>
#include <argp.h>
#include "utils.h"
#include <vector>
#include "stdio.h"

#include <gsl/gsl_multimin.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace Torch;
using namespace std;

class SVMSteerableFeatures3D
{

public:

  // Places in where the SVM will operate
  string training_file;

  string validation_file;

  string outputDirectory;

  string neuronFile;

  Allocator* allocator;

  bool   saveAllData;



  SVMSteerableFeatures3D(string training_file, string validation_file,
                         string outputDirectory = "/tmp/",
                         bool saveAllData = true){
    this->training_file   = training_file;
    this->validation_file = validation_file;
    this->outputDirectory = outputDirectory;
    if(!directoryExists(outputDirectory)){
      makeDirectory(outputDirectory);
    }
    this->saveAllData     = saveAllData;
    allocator = new Allocator;

    // Copies the trainig and validation files so that each thread has its owns
    int nMaxThreads = omp_get_max_threads();
    char buff[1024];
    for(int i = 0; i < nMaxThreads; i++){
      sprintf(buff,"%s%i", training_file.c_str(),i);
      copyFile(training_file, buff);
      sprintf(buff,"%s%i", validation_file.c_str(),i);
      copyFile(validation_file, buff);
    }


  }


  /** C and sk in linear, all the others are their log.*/
  void findCandSGrid
  (double &C_result,
   double &sk_result,
   double &error,
   double logC_init = 0,
   double logC_end  = 4,
   double logC_step = 1,
   double logsk_init = 0,
   double logsk_end  = 4,
   double logsk_step = 1
   );


  /** Trains the svm with a given C and a given sigma in the kernel. C and S linear*/
  void trainSVMSteerable
  ( double C,
    double sk,
    double& training_error,
    double& validation_error,
    double& b,
    int& nsv,
    int& nsvb,
    double& time,
    string filenameToStoreError = "/tmp/t_err.txt",
    int nthread = 0);

  //Wrapper for trainSVMSteerable for the gsl minimization algorithm
  static double trainSVMSteerable_f(const gsl_vector*v, void* params);

  /** Optimizes over C and s_k. C_init and sk_inin are linear coordinates.*/
  void findCandSnmsimplex2(double &C_result, double &sk_result,
                           double  C_init = 10,
                           double  sk_init = 10);



  static void extractSVM(SVM* svm, vector< double >& alpha,
                         vector< vector< double > >& supportVectors);


};




#endif
