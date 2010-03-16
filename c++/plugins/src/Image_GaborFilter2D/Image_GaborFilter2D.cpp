
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
// Written and (C) by Aurelien Lucchi                                  //
// Contact <aurelien.lucchi@gmail.com> for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <gmodule.h>
#include <stdio.h>
#include <Object.h>
#include "Image.h"

extern "C"
{
#include "GaborFilter2D.h"
#include "GaborFilter2D.cpp"

  Image< float >* input_img = 0;
  Image< float >* output_img = 0;

  double sigma = 4;
  double wavelength = 8;
  int mode = 0;

  void load_parameters()
  {
    string filename = "Image_GaborFilter2D.txt";
    ifstream in(filename.c_str());
    string str;
    size_t found;
    while(getline(in,str))
      {
        // trim the string
        std::remove(str.begin(), str.end(), ' ');

        std::cout << "str:" << str << std::endl;

        found=str.find("sigma");
        if (found!=string::npos)
          {
            std::istringstream iss(str.substr(6));
            iss >> sigma;
            std::cout << "sigma:" << sigma << std::endl;
          }
        else
          {
            found=str.find("wavelength");
            if (found!=string::npos)
              {
                std::istringstream iss(str.substr(11));
                iss >> wavelength;
                std::cout << "wavelength:" << wavelength << std::endl;
              }          
            else
              {
                found=str.find("mode");
                if (found!=string::npos)
                  {
                    std::istringstream iss(str.substr(5));
                    iss >> mode;
                    std::cout << "mode:" << mode << std::endl;
                  }
              }
          }
      }
    in.close();
  }

  G_MODULE_EXPORT const bool plugin_init()
  {

    printf("init Image plugin\n");    
    return true;
  }

  G_MODULE_EXPORT const bool plugin_run(vector<Object*>& objects)
  {
    printf("Image plugin running\n");
    load_parameters();
    int nbImage = 0;
    for(vector<Object*>::iterator itObject = objects.begin();
        itObject != objects.end(); itObject++)
      {
        if((*itObject)->className()=="Image")
          {
            if(nbImage == 0)
              {
                input_img = (Image<float>*)*itObject;
                printf("input img - width: %d, height: %d\n",input_img->width, input_img->height);
                nbImage++;
              }
            else
              {
                output_img = (Image<float>*)*itObject;
                printf("output img - width: %d, height: %d\n",output_img->width, output_img->height);
                GaborFilter2D gb(input_img, output_img, (GaborFilter2D::eMode)mode);

                printf("Computing Gabor filter output\n");
                gb.compute(sigma, wavelength,0.0);
                gb.filter();
                printf("Replacing image\n");

                output_img->reloadTexture();
              }
          }
      }

    return true;
  }

  G_MODULE_EXPORT const bool plugin_quit()
  {

    printf("Exit\n");
    return true;
  }
}
