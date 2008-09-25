#include "Mask.h"

Polynomial* Mask::hermite_polynomial_p(int order)
{
  vector< double > coeffs(1);
  coeffs[0]=1.0;
  vector< double > coeffs_d(2);
  coeffs_d[0] = 0.0;
  coeffs_d[1] = 2;

  Polynomial* otr = new Polynomial(coeffs_d);
  Polynomial* p = new Polynomial(coeffs);

  for(int i = 0; i < order; i++){
    //p = p' - 2*x*p;
    Polynomial* p_dev = p->derivative();
    Polynomial* p_otr = p->multiply_polynomial(otr);
    p = p_otr->substract_polynomial(p_dev);
  }

  return p;
}


vector<float> Mask::hermitian_mask(int order, float sigma, bool limit_value){
  //First we get the polynomial that would multiply the gaussian kernel.
  Polynomial* p = hermite_polynomial_p(order);

  //This calculates half of the mask. It would be symetric or antysimetric.
  vector< double > value(1);
  double norm_coeff = 1.0/(sigma*
                           sqrt(M_PI*
                                pow(2,order)*
                                factorial_n(order)
                                )
                           );
  value[0] = norm_coeff*p->evaluate(0);
  int i = 1;
  do{
    value.push_back(norm_coeff*
                    p->evaluate(-i/sigma)*
                    exp(-double(i*i)/(sigma*sigma)));
    i = i+1;
  }
  while( (exp(-double(i*i)/(sigma*sigma)) > 1e-3) && (value.size() <= 50)*limit_value ||
         (1-limit_value)*(value.size() <= 50)
       );

  float simmetry;
  if( order%2 != 0)
    simmetry = -1;
  else
    simmetry = 1;

  vector<float> mask(2*value.size()-1);
  mask[value.size()-1]= value[0];
  for(int i = 1; i < value.size(); i++){
    mask[value.size()-1+i] = value[i];
    mask[value.size()-1-i] = simmetry*value[i];
  }
  return mask;
}


vector<float> Mask::gaussian_mask(int order, float sigma, bool limit_value){
  //First we get the polynomial that would multiply the gaussian kernel.
  vector< double > coeffs(1);
  coeffs[0]=1.0;
  vector< double > coeffs_d(2);
  coeffs_d[0] = 0.0;
  coeffs_d[1] = -1/(sigma*sigma);

  Polynomial* otr = new Polynomial(coeffs_d);
  Polynomial* p = new Polynomial(coeffs);

  for(int i = 0; i < order; i++){
    //p = p' - x/sigma p;
    Polynomial* p_dev = p->derivative();
    Polynomial* p_otr = p->multiply_polynomial(otr);
//     delete p;
    p = p_dev->add_polynomial(p_otr);
  }
  // p->print();

  //This calculates half of the mask. It would be symetric or antysimetric.
  vector< double > value(1);
  double norm_coeff = 1.0/(sqrt(2.0*M_PI)*sigma);
  value[0] = norm_coeff*p->evaluate(0);
  int i = 1;
  do{
    value.push_back(norm_coeff*
                    p->evaluate(i)*
                    exp(-double(i*i)/(2.0*sigma*sigma)));
    i = i+1;
//     value.push_back(p->evaluate(i++));
  }
  while( (exp(-double(i*i)/(2.0*sigma*sigma)) > 1e-3)  && (value.size() <= 50)*limit_value ||
         (1-limit_value)*(value.size() <= 50)
       );
//    while( (value.size() <= 100));

//   p->print();
//   for(int i = 0; i < value.size(); i++)
//     printf("%f ", value[i]);
//   printf("\n");

  float simmetry;
  if( order%2 != 0)
    simmetry = -1;
  else
    simmetry = 1;

  vector<float> mask(2*value.size()-1);
  mask[value.size()-1]= value[0];
  for(int i = 1; i < value.size(); i++){
    mask[value.size()-1+i] = value[i];
    mask[value.size()-1-i] = simmetry*value[i];
  }
  return mask;
}

vector<float> Mask::gaussian_mask_orthogonal(int order, float sigma, bool limit_value){
  //First we get the polynomial that would multiply the gaussian kernel.
  vector< double > coeffs(1);
  coeffs[0]=1.0;
  vector< double > coeffs_d(2);
  coeffs_d[0] = 0.0;
  coeffs_d[1] = -1/(sigma*sigma);

  Polynomial* otr = new Polynomial(coeffs_d);
  Polynomial* p = new Polynomial(coeffs);

  for(int i = 0; i < order; i++){
    //p = p' - x/sigma p;
    Polynomial* p_dev = p->derivative();
    Polynomial* p_otr = p->multiply_polynomial(otr);
//     delete p;
    p = p_dev->add_polynomial(p_otr);
  }
  // p->print();

  //This calculates half of the mask. It would be symetric or antysimetric.
  vector< double > value(1);
  double norm_coeff = 1.0/(sqrt(2.0*M_PI)*sigma);
  value[0] = norm_coeff*p->evaluate(0);
  int i = 1;
  do{
    value.push_back(norm_coeff*
                    p->evaluate(i)*
                    exp(-double(i*i)/(4.0*sigma*sigma)));
    i = i+1;
//     value.push_back(p->evaluate(i++));
  }
  while( (exp(-double(i*i)/(4.0*sigma*sigma)) > 1e-3) 
         // && (value.size() <= 50)*limit_value ||
         // (1-limit_value)*(value.size() <= 50)
       );
//    while( (value.size() <= 100));

//   p->print();
//   for(int i = 0; i < value.size(); i++)
//     printf("%f ", value[i]);
//   printf("\n");

  float simmetry;
  if( order%2 != 0)
    simmetry = -1;
  else
    simmetry = 1;

  vector<float> mask(2*value.size()-1);
  mask[value.size()-1]= value[0];
  for(int i = 1; i < value.size(); i++){
    mask[value.size()-1+i] = value[i];
    mask[value.size()-1-i] = simmetry*value[i];
  }

  double en = energy1DGaussianMaskOrthogonal(order, sigma);
  for(int i = 0; i < mask.size(); i++)
    mask[i] /= en;

  return mask;
}


double Mask::energy1DGaussianMask(int dx, double sigma)
{
  double en = 1;
  en = sqrt(
            double(dfactorial_n(2*dx-1)) /
            (pow(2,dx+1)*pow(sigma,2*dx+1)*sqrt(M_PI))
            );
  return en;

}


double Mask::energy1DGaussianMaskOrthogonal(int dx, double sigma)
{
  double en = 1;
  en = sqrt(
            double(factorial_n(dx)) /
            (pow(sigma,2*dx+1)*sqrt(2*M_PI))
            );
  return en;

}


double Mask::energy2DGaussianMask(int dx, int dy, double sigma)
{
  double en = 1;
  en = sqrt(
            dfactorial_n(2*dx-1)*dfactorial_n(2*dy-1) /
            (pow(2,dx+dy+2)*pow(sigma,2*(dx+dy+1))*M_PI)
            );
  return en;
}

double Mask::energy2DGaussianMaskOrthogonal(int dx, int dy, double sigma)
{
  double en = 1;
  en = sqrt(
            double(factorial_n(dx)*factorial_n(dy)) /
            (pow(sigma,2*dx+2*dy+2)*2*M_PI)
            );
  return en;
}
