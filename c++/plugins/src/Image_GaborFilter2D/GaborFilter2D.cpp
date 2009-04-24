#include "GaborFilter2D.h"

#define round(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))

void GaborFilter2D::init()
{
  m_directory = m_input_image.substr(0,m_input_image.find_last_of("/\\")+1);
  m_gbFilter = 0;
  m_sz_x = -1;
  m_sz_y = -1;
}

GaborFilter2D::GaborFilter2D(string filename_image, eMode mode)
{
  m_input_image = filename_image;
  m_mode = mode;
  init();

  m_image = new Image<float>(m_input_image,true);
  printf("m_directory:%s\n", m_directory.c_str());
  printf("Image size:%dx%d\n", m_image->width, m_image->height);
  m_result = m_image->create_blank_image_float(m_directory + "result.jpg");
  m_resultNoDC = m_image->create_blank_image_float(m_directory + "resultnoDC.jpg");
}

GaborFilter2D::GaborFilter2D(Image<float>* image, Image<float>* result, eMode mode)
{
  m_input_image = "";
  m_mode = mode;
  init();

  m_image = image;
  printf("m_directory:%s\n", m_directory.c_str());
  printf("Image size:%dx%d\n", m_image->width, m_image->height);
  if(result == 0)
    m_result = m_image->create_blank_image_float(m_directory + "result.jpg");
  else
    m_result = result;
  m_resultNoDC = m_image->create_blank_image_float(m_directory + "resultnoDC.jpg");
}

gsl_matrix* GaborFilter2D::compute(
  const double sigma,
  const double wavelength,
  const double psi)
{
  update_sigmas(sigma);
  gsl_matrix* gbFilter=gsl_matrix_calloc(m_sz_x, m_sz_y);
  int step=30;
  for(int angle=0;angle<180;angle+=step)
    {
      compute_filter(angle,sigma,wavelength,psi);

      if(m_mode==MODE_AVG)
        {
          gsl_matrix_add(gbFilter,m_gbFilter);
        }
      else
        {
          // TODO : this could be optimized
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
          for (int i = 0; i < m_sz_x; i++)
            for (int j = 0; j < m_sz_y; j++)
              {
                double mij = gsl_matrix_get(m_gbFilter, i, j);
                double gij = gsl_matrix_get(gbFilter, i, j);
                gsl_matrix_set(gbFilter, i, j, GSL_MAX(mij,gij));
              }
        }
    }

  if(m_mode==MODE_AVG)
    gsl_matrix_scale(gbFilter,step/180.0);

  gsl_matrix_free(m_gbFilter);
  m_gbFilter = gbFilter;

  return m_gbFilter;
}

gsl_matrix* GaborFilter2D::compute(
  const double angle,
  const double sigma,
  const double wavelength,
  const double psi,
  bool outputCoeffs)
{
  update_sigmas(sigma);
  compute_filter(angle,sigma,wavelength,psi);

  if(outputCoeffs)
    {
      FILE* f = fopen("gaborfilter.dat", "wb");
      double value;
      for(int i = 0; i < m_sz_x; i++){
        for(int j = 0; j < m_sz_y; j++){
          value = gsl_matrix_get(m_gbFilter,i,j);
          fprintf(f, "%f ",value);
        }
        fprintf(f, "\n");
      }
      fclose(f);
    }
}

void GaborFilter2D::compute_filter(const double angle,const double sigma,const double wavelength,const double psi)
{
  int i,j;
  double x, y;
  double x_angle, y_angle;
  double gb;
  double sigma_x = sigma;
  double sigma_y = sigma; //sigma/gamma;

  for(x=-m_sz_x/2, i=0;x<=m_sz_x/2;x++, i++)
    for(y=-m_sz_y/2, j=0;y<=m_sz_y/2;y++, j++)
      {
	// Rotation 
	x_angle=x*cos(angle)+y*sin(angle);
	y_angle=-x*sin(angle)+y*cos(angle);
 
	gb = exp(-0.5*(((x_angle*x_angle)/(sigma_x*sigma_x))+((y_angle*y_angle)/(sigma_y*sigma_y))));
	gb *= cos((2*M_PI*x_angle/wavelength)+psi);

	gsl_matrix_set(m_gbFilter, i, j, gb);
      }
}

void GaborFilter2D::update_sigmas(const double sigma)
{
  assert(sigma > 0.0);
 
  int sz_x = 6*sigma; //sigma_x
  int sz_y = sz_x;

  if((sz_x & 1) == 0)
    sz_x++;

  if((sz_y & 1) == 0)
    sz_y++;

  //printf("sz_x: %d\n", sz_x);
  //printf("sz_y: %d\n", sz_y);

  if(m_sz_x != sz_x && m_sz_y != sz_y)
    {
      if(m_gbFilter!=0)
	gsl_matrix_free(m_gbFilter);

      m_gbFilter = gsl_matrix_alloc(sz_x, sz_y);
      m_sz_x = sz_x;
      m_sz_y = sz_y;
    }
}

double GaborFilter2D::response(int x, int y)
{
  double ret = 0;
  int ofs_x = m_sz_x/2;
  int ofs_y = m_sz_y/2;
  int k_x; // kernel x
  int k_y; // kernel y

  //printf("x: %d, ofs_x: %d\n", x, ofs_x);
  //printf("y: %d, ofs_y: %d\n", y, ofs_y);

  if(x<ofs_x || y<ofs_y || x>=m_image->width-ofs_x || y>=m_image->height-ofs_y)
    {
      /*  
      // Warping the image near the borders
      int idx_x, idx_y;
      for(int i = 0; i < m_sz_x; i++){
	for(int j = 0; j < m_sz_y; j++){
	  idx_x = x-ofs_x+i;
	  if(idx_x<0)
	    idx_x = 0;
	    //idx_x += m_sz_x;
	  else if(idx_x>=m_image->width)
	    //idx_x -= m_image->width;
	    idx_x = m_image->width-1;

	  idx_y = y-ofs_y+j;
	  if(idx_y<0)
	    //idx_y += m_sz_y;
	    idx_y = 0;
	  else if(idx_y>=m_image->height)
	    //idx_y -= m_image->height;
	    idx_y = m_image->height-1;

	  k_x = m_sz_x-1-i;
	  k_y = m_sz_y-1-i;
	  ret += gsl_matrix_get(m_gbFilter,k_x,k_y)*m_image->at(idx_x, idx_y);
	}
      }
      */
    }
    else
      {
	// Normal convolution
	for(int i = 0; i < m_sz_x; i++){
	  for(int j = 0; j < m_sz_y; j++){
	    k_x = m_sz_x-1-i;
	    k_y = m_sz_y-1-j;
	    ret += gsl_matrix_get(m_gbFilter,k_x,k_y)*m_image->at(x-ofs_x+i,y-ofs_y+j);
	  }
	}
      }
  return ret;
}

void GaborFilter2D::filter()
{
  double res;
  double mean = 0;
  #pragma omp parallel for
  for(int x = 0; x < m_image->width; x++){
    #pragma omp parallel for
    for(int y = 0; y < m_image->height; y++){
      res = response(x,y);
      m_result->put(x,y,res);
      mean +=res;
    }
  }
  mean /= (m_image->width*m_image->height);
  //printf("Mean : %f\n", mean);
  m_result->save();

  // Remove DC component
#pragma omp parallel for
  for(int x = 0; x < m_image->width; x++){
#pragma omp parallel for
    for(int y = 0; y < m_image->height; y++){
      res = m_result->at(x,y);
      res -= mean;
      m_resultNoDC->put(x,y,res);
    }
  }

  m_resultNoDC->save();
}
