
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
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////


#ifndef MASKS_H_
#define MASKS_H_

#include <math.h>
#include "utils.h"
#include "polynomial.h"

/** Class that contains static methods to obtain the masks used to blur, calculate image derivatives, Hermite coefficients and so on from images or cubes of data.
*/

class Mask
{
public:

  Mask(){}

  /** Returns a Hermite 'physics' polynomial of the given order. The Polynomial is obtained with the following recursive equation:
      /f[
        H_{n+1}(x) = 2xH_n(x) - H^'(x)
      /f]
      with $H_1(x) = 1$.
   */
  static Polynomial* hermite_polynomial_p(int order);


  /** Creates a 'hermite' mask. It corresponds to the function:
      \begin{equation}
      \frac{1}{\sqrt{2^n}n!} H_n(\frac{-x}{\sigma})\frac{1}{\sigma \sqrt{\Pi}}e^{-\frac{-x^2}{2}}
      \end{equation}
      Where H_n is the (physics) hermitian polynomial of order n.
  */
  static vector<float> hermitian_mask(int order, float sigma, bool limit_value = false);


  /** Creates a Gaussian mask of 60 elements. If limit_value = true, the size of the mask is truncated if its value is bellow 1e-4*/
  static vector<float> gaussian_mask(int order, float sigma, bool limit_value = false);

  /** Creates an orthogonal gaussian masks. This responds to the formula:
      \begin{equation}
      \frac{1}{\sqrt{2 \Pi \sigma^2}}\frac{(-1)^n}{\sigma^n}}H_n(\frac\({-x}{\sigma}\)e^{-\frac{x^2}{4\sigma^2}}
      \end{equation}
  */
  static vector<float> gaussian_mask_orthogonal(int order, float sigma, bool limit_value = true);

  /** Returns the energy of the oglp (orthogonal gaussian-like polynomial. It implements:
      \begin{equation}
      \sqrt{\frac{dx!}{\sigma^{2dx+1}\sqrt{2\pi}}}
      \end{equation}
  */
  static double energy1DGaussianMaskOrthogonal(int dx, double sigma);

  /** Returns the energy of the oglp (orthogonal gaussian-like polynomial. It implements:
      \begin{equation}
      \sqrt{\frac{dx!dy!}{\sigma^{2dx+2dy+2}2\pi}}
      \end{equation}
  */
  static double energy2DGaussianMaskOrthogonal(int dx, int dy, double sigma);



  /** Returns the energy of the derivatives of a gaussian mask. It implements the formula:
      \begin{equation}
      \sqrt{\frac{(2d_x-1)!!(2d_y-1)!!}{2^{d_x+d_y+2}\sigma^{2(d_x+d_y+1)\Pi}}}
      \end{equation}
  */
  static double energy2DGaussianMask(int dx, int dy, double sigma);

  /** Returns the energy of the derivatives of a gaussian mask. It implements the formula:
      \begin{equation}
      \sqrt{\frac{(2d_x-1)!!}{2^{d_x+1}\sigma^{2d_x+1) \sqrt(\Pi)}}}
      \end{equation}
  */
  static double energy1DGaussianMask(int dx, double sigma);


};






#endif

