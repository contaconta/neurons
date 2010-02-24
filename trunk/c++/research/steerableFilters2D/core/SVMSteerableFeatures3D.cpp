#include "SVMSteerableFeatures3D.h"

void SVMSteerableFeatures3D::trainSVMSteerable
( double C,
  double sk,
  double& training_error,
  double& validation_error,
  double& b,
  int& nsv,
  int& nsvb,
  double& time,
  string filenameToStoreError,
  int nthread)
{
  // Create the data, the kernel, the svm and the trainer
  int max_load = -1;
  bool binary_mode = false;
  char buff[1024];
  sprintf(buff,"%s%i", training_file.c_str(),nthread);
  DataSet *data = NULL;
  data = new(allocator) MatDataSet(buff, -1, 1,
                                   false, max_load, binary_mode);
  Kernel* kernel = NULL;
  kernel = new(allocator) GaussianKernel(1.0/(sk*sk));
  SVM* svm = NULL;
  svm = new(allocator) SVMClassification(kernel);
  svm->setROption("cache size", 100);
  svm->setROption("C", C);
  QCTrainer* trainer = new QCTrainer(svm);
  //Timer timer;
  trainer->train(data, NULL);
  svm->bCompute();
  //timer.stop();
  nsv  = svm->n_support_vectors;
  nsvb = svm->n_support_vectors_bound;
  b    = svm->b;
  //time = timer.getTime();
  message("SVM: [sigma = %f, C= %f, SV= %d, SVB= %d, b=%f]."
          " Training time: %f, Training error: %f",
          sk, C, svm->n_support_vectors, svm->n_support_vectors_bound,
          svm->b,0, trainer->current_error);
          //timer.getTime(), trainer->current_error);

  training_error = trainer->current_error;

  // Testing (if defined)
  if(validation_file != ""){
    if(1){ //Perform the validation error "a la torch"
      message("Performing the test");
      sprintf(buff,"%s%i", validation_file.c_str(),nthread);
      DataSet* test = new MatDataSet(buff, -1, 1, false,
                                     max_load, binary_mode);
      MeasurerList measurers;
      DiskXFile class_file(filenameToStoreError.c_str(),"w");
      TwoClassFormat *class_format = new(allocator) TwoClassFormat(test);
      ClassMeasurer  *class_meas
        = new(allocator) ClassMeasurer(svm->outputs, test, class_format, &class_file);
      measurers.addNode(class_meas);
      trainer->test(&measurers);
      vector<double> error = readVectorDouble(filenameToStoreError);
      // This is stupid!!! You can not access the internal error after testing it!! The only way is to read the file!!!
      validation_error = error[0];
      printf("The validation error is %f\n", error[0]);
      delete test;
    }
    // do it in the way I am detecting
    else {
      int nSupportVectors = 0;
      int dimensionOfSupportVectors = 0;
      nSupportVectors           = svm->n_support_vectors;
      dimensionOfSupportVectors = svm->sv_sequences[0]->frame_size;
      vector< vector< double > > svectors =
        allocateMatrix(nSupportVectors, dimensionOfSupportVectors);
      vector< double > alphas(nSupportVectors);
      for(int i = 0; i < nSupportVectors; i++){
        alphas[i] = svm->sv_alpha[i];
        for(int j = 0; j < dimensionOfSupportVectors; j++){
          svectors[i][j] = svm->sv_sequences[i]->frames[0][j];
        }
      }
      sprintf(buff,"%s%i", validation_file.c_str(),nthread);
      std::ifstream torchFile(buff);
      int nSamples; int nDimensions;
      torchFile >> nSamples; torchFile >> nDimensions;
      printf("Loading torch File with %i samples and %i dimensions\n", nSamples, nDimensions);
      vector< vector< double > > testData = allocateMatrix(nSamples, nDimensions);
      for(int i = 0; i < nSamples; i++)
        for(int j = 0; j < nDimensions; j++)
          torchFile >> testData[i][j];
      torchFile.close();
      validation_error = 0;
      double res = 0;
      double expn = 0;
      printf("And now doing the evaluation\n");
      for(int s = 0; s < nSamples; s++){
        //        printf("nSample = %i\n", s);
        res = 0;
        expn = 0;
        for(int i = 0; i < alphas.size(); i++){
          expn = 0;
          for(int j = 0; j < svectors[i].size(); j++)
            expn -= (svectors[i][j] - testData[s][j])* (svectors[i][j] - testData[s][j]);
          res += alphas[i]*exp(expn/(sk*sk));
        }
        if( ((res < svm->b) && (testData[s][nDimensions-1]== 1)) ||
            ((res > svm->b) && (testData[s][nDimensions-1]==-1)) ){
          validation_error++;
        }
      }
      validation_error = validation_error/nSamples;
      printf("The validation error is %f\n", validation_error);
    }//Validation error my way
  }//Compute validation error

  if(saveAllData){
    char model_file[1024];
    sprintf(model_file, "%s/svm_%03.03e_%03.03e.svm",
            outputDirectory.c_str(), C, sk);
    DiskXFile model_(model_file, "w");
    svm->saveXFile(&model_);
    kernel->saveXFile(&model_);
    if(0){
      vector< vector< double > > all_trainings;
      if(fileExists(outputDirectory + "/all_trainings.txt")){
        all_trainings =
          loadMatrix(outputDirectory + "/all_trainings.txt");}
      all_trainings.push_back( vector< double >(8));
      all_trainings[all_trainings.size()-1][0] = C;
      all_trainings[all_trainings.size()-1][1] = sk;
      all_trainings[all_trainings.size()-1][2] = b;
      all_trainings[all_trainings.size()-1][3] = training_error;
      all_trainings[all_trainings.size()-1][4] = validation_error;
      all_trainings[all_trainings.size()-1][5] = nsv;
      all_trainings[all_trainings.size()-1][6] = nsvb;
      all_trainings[all_trainings.size()-1][7] = time;
      saveMatrix(all_trainings, outputDirectory + "/all_trainings.txt");
    }
  }

  printf("The SVM has %i support vectors and %i in the bound \n",
         svm->n_support_vectors,
         svm->n_support_vectors_bound);

  int nalph = 0;
  printf("A %f\n", (float)svm->n_alpha);
  for(int i = 0; i < svm->n_alpha; i++)
    {
      if(fabs(svm->sv_alpha[i]) > 1e-3)
        nalph++;
    }
  printf("The number of alphas != 0 is %i\n", nalph);



  // Cleanup
  delete data;
  delete trainer;
  delete svm;
  delete kernel;
}


void SVMSteerableFeatures3D::findCandSGrid
( double &C_result,
 double &sk_result,
 double &error,
 double C_init,
 double C_end,
 double C_step,
 double sk_init,
 double sk_end,
 double sk_step
)
{
  int   C_size = 1+(C_end - C_init)/C_step;
  vector< double > C_v(C_size);
  for(int i = 0; i < C_size; i++)
    C_v[i] = pow(10, C_init+C_step*i);
  int   sk_size = 1+(sk_end - sk_init)/sk_step;
  vector< double > sk_v(sk_size);
  for(int i = 0; i < sk_size; i++)
    sk_v[i] = pow(10, sk_init+sk_step*i);

  // Values to find
  vector< vector< double > > m_terr = allocateMatrix(C_size, sk_size);
  vector< vector< double > > m_verr = allocateMatrix(C_size, sk_size);
  vector< vector< double > > m_sk   = allocateMatrix(C_size, sk_size);
  vector< vector< double > > m_C    = allocateMatrix(C_size, sk_size);
  vector< vector< double > > m_sv   = allocateMatrix(C_size, sk_size);
  vector< vector< double > > m_svb  = allocateMatrix(C_size, sk_size);
  vector< vector< double > > m_time = allocateMatrix(C_size, sk_size);
  vector< vector< double > > m_b    = allocateMatrix(C_size, sk_size);

  int nthreads = 1;
#ifdef WITH_OPENMP
  nthreads = omp_get_max_threads();
#endif
  printf("SVMSteerableFeatures3D::findCandSGrid is being done with %i threads\n", nthreads);

  // For parallel code it is better to have just one for. Map to linear indexes.
  vector< double > C_v2;
  vector< double > s_k2;
  vector< int    > C_idx;
  vector< int    > s_kidx;

  for(int i = 0; i < C_v.size(); i++)
    for(int j = 0; j < sk_v.size(); j++){
      C_idx.push_back(i);
      s_kidx.push_back(j);
      C_v2.push_back(C_v[i]);
      s_k2.push_back(sk_v[j]);
    }


#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
  for(int i = 0; i < C_v2.size(); i++){
    int C_i   = C_idx[i];
    int sk_i  = s_kidx[i];
    double C  = C_v2[i];
    double sk = s_k2[i];
    double training_error, validation_error, b, time;
    int nsv, nsvb;
    char filenameToStoreTheError[1024];
    int nth = 0;
#ifdef WITH_OPENMP
    nth = omp_get_thread_num();
    printf("SVMSteerableFeatures3D::findCandSGrid C=%f s=%f nthread=%i\n",
           C, sk, omp_get_thread_num());
#endif
    sprintf(filenameToStoreTheError, "the_class_err_%i.txt", nth);

    trainSVMSteerable
      (C, sk,
       training_error, validation_error, b, nsv, nsvb, time,
       filenameToStoreTheError, nth);

    m_terr [C_i][sk_i] = training_error;
    m_verr [C_i][sk_i] = validation_error;
    m_C    [C_i][sk_i] = C;
    m_sk   [C_i][sk_i] = sk;
    m_b    [C_i][sk_i] = b;
    m_sv   [C_i][sk_i] = nsv;
    m_svb  [C_i][sk_i] = nsvb;
    m_time [C_i][sk_i] = time;
  }//C_v2

  // Get the minimum of the error
  int idxr, idxc;
  if(validation_file == ""){
    getMinInMatrix(m_terr, error, idxr, idxc);
  } else {
    getMinInMatrix(m_verr, error, idxr, idxc);
  }

  C_result  = m_C [idxr][idxc];
  sk_result = m_sk[idxr][idxc];

  if(saveAllData){
    saveMatrix(m_terr, outputDirectory + "/" + "m_terr.txt");
    saveMatrix(m_verr, outputDirectory + "/" + "m_verr.txt");
    saveMatrix(m_C,    outputDirectory + "/" + "m_C.txt"   );
    saveMatrix(m_sk,   outputDirectory + "/" + "m_sk.txt"  );
    saveMatrix(m_b,    outputDirectory + "/" + "m_b.txt"   );
    saveMatrix(m_sv,   outputDirectory + "/" + "m_sv.txt"  );
    saveMatrix(m_svb,  outputDirectory + "/" + "m_svb.txt" );
    saveMatrix(m_time, outputDirectory + "/" + "m_time.txt");
    vector< double > toSave(2);
    toSave[0] = C_result;
    toSave[1] = sk_result;
    saveVectorDouble(toSave, outputDirectory + "/CandSGrid.txt");
    saveVectorDouble(toSave, outputDirectory + "/CandS.txt");
  }

}




double SVMSteerableFeatures3D::trainSVMSteerable_f
(const gsl_vector*v, void* params)
{
  double t_e, v_e, b, time, C, sk;
  int nsv, nsvb;
  C  = gsl_vector_get(v, 0);
  sk = gsl_vector_get(v, 1);

  SVMSteerableFeatures3D* obj = (SVMSteerableFeatures3D*)params;

  obj->trainSVMSteerable(pow(10,C), pow(10,sk), t_e, v_e, b, nsv, nsvb, time);

  return v_e;
}

void SVMSteerableFeatures3D::findCandSnmsimplex2
(double &C_result, double &sk_result, double C_init, double sk_init)
{
  const gsl_multimin_fminimizer_type *T =
    gsl_multimin_fminimizer_nmsimplex;
  gsl_multimin_fminimizer *s = NULL;
  gsl_vector *ss, *x;
  gsl_multimin_function minex_func;

  size_t iter = 0;
  int status;
  double size;

  /* Starting point */
  x = gsl_vector_alloc (2);
  gsl_vector_set (x, 0, log10(C_init)  );
  gsl_vector_set (x, 1, log10(sk_init) );

  /* Set initial step sizes to 1 */
  ss = gsl_vector_alloc (2);
  gsl_vector_set_all (ss, 0.2);

  /* Initialize method and iterate */
  minex_func.n = 2;
  minex_func.f = &SVMSteerableFeatures3D::trainSVMSteerable_f;
  minex_func.params = (void *)this;

  s = gsl_multimin_fminimizer_alloc (T, 2);
  gsl_multimin_fminimizer_set (s, &minex_func, x, ss);

  string fileLog = outputDirectory + "/optPath.txt";
  FILE* f = fopen(fileLog.c_str(),"w");

  do
    {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);

      if (status)
        break;

      size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, 1e-1);

      if (status == GSL_SUCCESS)
        {
          printf ("converged to minimum at\n");
        }

      printf ("GSLMIN:: %5d %10.3e %10.3e f()= %10.3e size = %.3f\n",
              iter,
              gsl_vector_get (s->x, 0),
              gsl_vector_get (s->x, 1),
              s->fval, size);
      fprintf (f, "%5d %10.3e %10.3e %10.3e %.3f\n",
              iter,
              gsl_vector_get (s->x, 0),
              gsl_vector_get (s->x, 1),
              s->fval, size);

    }
  while (status == GSL_CONTINUE && iter < 20);

  fclose(f);

  C_result = pow(10,gsl_vector_get (s->x, 0));
  sk_result = pow(10,gsl_vector_get (s->x, 0));

  if(saveAllData){
    vector< double > toSave(2);
    toSave[0] = C_result;
    toSave[1] = sk_result;
    saveVectorDouble(toSave, outputDirectory + "/CandSnmsimplex2.txt");
    saveVectorDouble(toSave, outputDirectory + "/CandS.txt");
  }


  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free (s);
}
