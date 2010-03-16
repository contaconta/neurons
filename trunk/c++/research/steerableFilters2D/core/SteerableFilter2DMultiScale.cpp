#include "SteerableFilter2DMultiScale.h"
#include "Point2D.h"


SteerableFilter2DMultiScale::SteerableFilter2DMultiScale
(string _filename_image,
 int _M,
 float _scale_init,
 float _scale_end,
 float _scale_step,
 string filenameOutput,
 string filenameOutputTheta
 )
{
  filename_image = _filename_image;
  M = _M;
  scale_init = _scale_init;
  scale_end  = _scale_end;
  scale_step = _scale_step;

  for(float s = scale_init; s <= scale_end; s+=scale_step)
    stf.push_back( new SteerableFilter2D
                   ( filename_image, M, s));

  alphas = gsl_vector_alloc(stf.size()*M*(M+3)/2);

  directory = filename_image.substr(0,filename_image.find_last_of("/\\")+1);
  image = new Image<float>(filename_image,true);
  result = image->create_blank_image_float(filenameOutput);
  orientation = image->create_blank_image_float(filenameOutputTheta);
}

SteerableFilter2DMultiScale::SteerableFilter2DMultiScale
(string _filename_image,
 string filename_alphas,
 float _scale_init,
 float _scale_end,
 float _scale_step,
 string filenameOutput,
 string filenameOutputTheta
 )
{
  filename_image = _filename_image;
  scale_init = _scale_init;
  scale_end  = _scale_end;
  scale_step = _scale_step;

  loadAlphas(filename_alphas);

  directory = filename_image.substr(0,filename_image.find_last_of("/\\")+1);
  image = new Image<float>(filename_image,true);
  result = image->create_blank_image_float(filenameOutput);
  orientation = image->create_blank_image_float(filenameOutputTheta);
}

void SteerableFilter2DMultiScale::outputCoordinates
( string training_points,
  string filename,
  bool convert_to_radians)
{


  Cloud< Point2Dot >* training_set = new Cloud< Point2Dot >(training_points);

  // vector< Point2D* > training_set = Point2D::readFile(training_points);

  // training_set[0]->print();

  std::ofstream out(filename.c_str());
  // std::ofstream points("points.txt");
  bool torch_format = true;

  if(torch_format){
    out << training_set->points.size() << " "
        << stf[0]->derivatives.size()*stf.size() + 1 << std::endl;
  }
  vector< int > indexes(3);
  vector< float > micrometers(3);
  micrometers[2] = 0;
  indexes[2] = 0;

  for(int nP = 0; nP < training_set->points.size(); nP++){

    //Get the derivative coordinates
    micrometers[0] = training_set->points[nP]->coords[0];
    micrometers[1] = training_set->points[nP]->coords[1];
    image->micrometersToIndexes(micrometers, indexes);
    Point2Dot* tpp = dynamic_cast<Point2Dot*>(training_set->points[nP]);
    vector< double > coordinates =
      getDerivativeCoordinatesRotated(indexes[0], indexes[1],
                                      tpp->theta);
    //Output them 
    for(int i = 0 ; i < coordinates.size(); i++)
      out << coordinates[i] << " " ;

    if(torch_format)
      out << tpp->type;
    out << std::endl;
  }
  out.close();
  // points.close();
}

void SteerableFilter2DMultiScale::outputCoordinatesAllOrientations
( string training_points,
  string filename,
  bool convert_to_radians)
{


  Cloud< Point2Dot >* training_set = new Cloud< Point2Dot >(training_points);

  // vector< Point2D* > training_set = Point2D::readFile(training_points);

  // training_set[0]->print();

  std::ofstream out(filename.c_str());
  // std::ofstream points("points.txt");
  bool torch_format = true;

  if(torch_format){
    out << training_set->points.size() << " "
        << stf[0]->derivatives.size()*stf.size() + 1 << std::endl;
  }
  vector< int > indexes(3);
  vector< float > micrometers(3);
  micrometers[2] = 0;
  indexes[2] = 0;

  for(int nP = 0; nP < training_set->points.size(); nP++){

    //Get the derivative coordinates ar different orientations
    micrometers[0] = training_set->points[nP]->coords[0];
    micrometers[1] = training_set->points[nP]->coords[1];
    image->micrometersToIndexes(micrometers, indexes);
    Point2Dot* tpp = dynamic_cast<Point2Dot*>(training_set->points[nP]);
    for(float angle = 0; angle < 180; angle += 10){
      vector< double > coordinates =
        getDerivativeCoordinatesRotated(indexes[0], indexes[1],
                                        angle*M_PI/180);
      //Output them
      for(int i = 0 ; i < coordinates.size(); i++)
        out << coordinates[i] << " " ;
      if(torch_format)
        out << tpp->type;
      out << std::endl;
    }
  }
  out.close();
  // points.close();
}


vector< double >
SteerableFilter2DMultiScale::getDerivativeCoordinatesRotated
(int x, int y, double theta)
{
  vector< double > toReturn;

  gsl_vector* derivs = gsl_vector_alloc(stf[0]->alpha->size);
  gsl_vector* coords = gsl_vector_alloc(stf[0]->alpha->size);

  for(int nStf = 0; nStf < stf.size(); nStf++){
    gsl_matrix* rot = stf[nStf]->get_rotation_matrix(-theta);
    for(int i = 0; i < stf[nStf]->derivatives.size(); i++)
      gsl_vector_set(derivs, i,
                     stf[nStf]->derivatives[i]->at(x,y)
                     );

    //Do a normalization for the sigma. For each derivative order, multiply the coordinate by sigma.
    // int idx_o = 0;
    // for(int o = 1; o <= stf[nStf]->M; o++){
      // for(int i = 0; i <= o; i++){
        // gsl_vector_set(derivs, idx_o + i,
                       // gsl_vector_get(derivs, idx_o + i)*pow(stf[nStf]->sigma, o));
      // }
          // idx_o = idx_o + o + 1;
    // }

    // Do the normalization as Martens 1990
    // int idx_o = 0;
    // for(int o = 1; o <= stf[nStf]->M; o++){
      // for(int i = 0; i <= o; i++){
        // double c = 1/(sqrt((double)pow(2,o)*factorial(o-i)*factorial(i)));
        // gsl_vector_set(derivs, idx_o + i,
                       // gsl_vector_get(derivs, idx_o + i)*c);
      // }
          // idx_o = idx_o + o + 1;
    // }


    gsl_blas_dgemv(CblasTrans, 1.0, rot, derivs, 0, coords);

    for(int i = 0; i < coords->size; i++)
      toReturn.push_back(gsl_vector_get(coords, i));

    gsl_matrix_free(rot);
  }

  gsl_vector_free(derivs);
  gsl_vector_free(coords);
  return toReturn;
}



int SteerableFilter2DMultiScale::factorial(int x){
  int fac = 1;
  for (int i=2; i<=x; i++) fac *= i;
  return fac;
}

void SteerableFilter2DMultiScale::loadAlphas(string filename)
{
  FILE* f = fopen(filename.c_str(), "r");

  //This is a hack to automatically know the dimension of the vector
  char number[1024];
  int nNumbers = 0;
  while(fgets(number, 1024, f) != NULL){
    nNumbers++;
  }
  fclose(f);

  int nScales = 0;
  for(float s = scale_init; s <= scale_end; s+=scale_step)
    nScales++;

  //Gets the order of the derivatives
//   for(M = 0; M < 10; M++){
// //     printf("M = %i, %i %i\n", M, (M+3)*M/2, nNumbers);
//     if(nNumbers == nScales*(M+3)*M/2)
//       break;
//   }

  M = 4;

  f = fopen(filename.c_str(), "r");

  alphas = gsl_vector_alloc(nNumbers);
  int err = gsl_vector_fscanf(f, alphas);
  if(err == GSL_EFAILED){
    printf("Error reading the vectorx in %s\n", filename.c_str());
    exit(0);
  }
  fclose(f);
  printf("Alpha:\n");
  gsl_vector_fprintf (stdout, alphas, "%0.03f");

  //In case the steerable filters are not in there
  if(stf.size()!=nScales){
    stf.resize(0);
    for(float s = scale_init; s <= scale_end; s+=scale_step)
      stf.push_back( new SteerableFilter2D
                     ( filename_image, M, s));
  }

  //Load the coefficients on each steerable filter
  int offset = 0;
  for(int nstf = 0; nstf < stf.size(); nstf++){
    for(int i = 0; i < (M+3)*M/2; i++){
      gsl_vector_set(stf[nstf]->alpha, i,
                     gsl_vector_get(alphas, offset+i));
    }
    offset += (M+3)*M/2;
  }
}


double SteerableFilter2DMultiScale::response(double theta, int x, int y)
{
  double rs = 0;
  float s = scale_init;
  for(int i  = 0; i < stf.size(); i++){
    rs = rs + s*s*stf[i]->response(theta, x, y);
    s = s + scale_step;
  }
  return rs;
}

void SteerableFilter2DMultiScale::filter(double theta)
{
  printf("SteerableFilter2DMultiScale::filter(double theta)\n");
  for(int x = 0; x < image->width; x++){
    for(int y = 0; y < image->height; y++){
      result->put(x,y,response(theta,x,y));
    }
  }
  result->save();
}
