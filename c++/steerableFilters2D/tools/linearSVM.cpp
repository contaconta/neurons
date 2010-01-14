
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
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
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



using namespace std;
using namespace Torch;

int main(int argc, char **argv) {

  if(argc!=4){
    printf("Usage: linearSVM training.txt C output.txt\n");
    exit(0);
  }
  string trainingFile(argv[1]);
  float C = atof(argv[2]);
  string outputVectorName(argv[3]);
  string name_test_file;


  Allocator *allocator = new Allocator;
  int max_load = -1;
  bool binary_mode = false;

  Kernel* kernel = new(allocator) DotKernel();
  // Kernel* kernel = new(allocator) GaussianKernel(1./(10*10));
  SVM* svm = new(allocator) SVMClassification(kernel);

  svm->setROption("C", C);
  svm->setROption("cache size", 2000);

  DataSet *data = NULL;
  data = new(allocator) MatDataSet(trainingFile.c_str(), -1, 1,
                                   false, max_load, binary_mode);

  QCTrainer trainer(svm);
  // Timer timer;
  trainer.train(data, NULL);
  svm->bCompute();
  // timer.stop();
  message("LinearKernel: %d SV with %d at bounds", svm->n_support_vectors, svm->n_support_vectors_bound);
  // message("Training time: %f", timer.getTime());
  message("The SVM has a b of %f", svm->b);

  // Testing (if defined)
  if(0){
    message("Performing the test");
    DataSet* test = new MatDataSet(name_test_file.c_str(), -1, 1, false,
                                   max_load, binary_mode);
    MeasurerList measurers;
    DiskXFile class_file("the_class_err","w");
    TwoClassFormat *class_format = new(allocator) TwoClassFormat(test);
    ClassMeasurer  *class_meas
      = new(allocator) ClassMeasurer(svm->outputs, test, class_format, &class_file);
    measurers.addNode(class_meas);
    trainer.test(&measurers);
  }


  if(1){
    std::ofstream out(outputVectorName.c_str());
    double w = 0;
    for(int nD = 0; nD < data->n_inputs; nD++){
      w = 0;
      for(int nsv = 0; nsv < svm->n_support_vectors; nsv++){
        data->setExample(svm->support_vectors[nsv]);
        w = w + svm->sv_alpha[nsv]*
          data->inputs->frames[0][nD];
      }
      out << w << std::endl ;
    }
    out.close();
    printf("Plane saved: n_support_vectors = %i\n", svm->n_support_vectors);
  }

  delete allocator;
  return(0);









}
