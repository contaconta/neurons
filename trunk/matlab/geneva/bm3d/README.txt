------------------------------------------------------------------


  BM3D demo software for image/video restoration and enhancement  
                   Public release v1.7.5 (7 July 2010) 

------------------------------------------------------------------

Copyright (c) 2006-2010 Tampere University of Technology. 
All rights reserved.
This work should be used for nonprofit purposes only.

Authors:                     Kostadin Dabov
                             Alessandro Foi

BM3D web page:               http://www.cs.tut.fi/~foi/GCF-BM3D


------------------------------------------------------------------
Contents
------------------------------------------------------------------

The package comprises these functions

*) BM3D.m        : BM3D grayscale-image denoising [1]
*) CBM3D.m       : CBM3D RGB-image denoising [2]
*) VBM3D.m       : VBM3D grayscale-video denoising [3]
*) CVBM3D.m      : CVBM3D RGB-video denoising
*) BM3DSHARP.m   : BM3D-SHARP grayscale-image sharepening & 
                   denoising [4]
*) BM3DDEB.m     : BM3D-DEB grayscale-image deblurring [5]

For help on how to use these scripts, you can e.g. use "help BM3D"
or "help CBM3D".

Each demo calls MEX-functions that allow to change all possible 
parameters used in the algorithm from within the corresponding 
M-file.


------------------------------------------------------------------
Installation
------------------------------------------------------------------

Unzip both BM3D.zip (contains codes) and BM3D_images.zip (contains 
test images) in a folder that is in the MATLAB path.


------------------------------------------------------------------
Requirements
------------------------------------------------------------------

*) MS Windows (32- or 64-bit), Linux (32-bit or 64-bit)
    or Mac OS X (32-bit)
    note: CVBM3D supports only 32-bit and 64-bit Windows
*) Matlab v.6.5 or later with installed:
  -- Image Processing Toolbox (for visualization with "imshow"),

------------------------------------------------------------------
What's new in this release
------------------------------------------------------------------

v1.7.5
 + Changed the strong-noise parameters ("vn" profile) in BM3D.m,
   as proposed in [6].

v1.7.4
 + Added support for Matlab running on Mac OSX 64-bit

v1.7.3
 + Fixed a problem with writing to AVI files in CVBM3D
 + Fixed a problem with VBM3D when the input is a 3-D matrix

v1.7.2
 + Fixed the output of CVBM3D to be in range [0,255] instead of 
   in range [0,1]

v1.7.1
 + Fixed a bug in VBM3D.m introduced in v1.7 that concerns the
   declipping

v1.7
 + Added CVBM3D.m script that performs denoising on RGB-videos with
   AWGN
 + Fixed VBM3D.m to use declipping in the case when noisy AVI file
   is provided

v1.6
 + Made few fixes to the "getTransfMatrix" internal function.
   If used with default parameters, BM3D no longer requires
   neither Wavelet, PDE, nor Signal Processing toolbox.
 + Added support for x86_64 Linux

v1.5.1
 + Fixed bugs for older versions of Matlab
 + Added support for 32-bit Linux
 + improved the structure of the VBM3D.m script

v1.5
 + Added x86_64 version of the MEX-files that run on 64-bit Matlab 
   under Windows
 + Added a missing function in BM3DDEB.m
 + Improves some of the comments in the codes
 + Fixed a bug in VBM3D when only a input noisy video is provided

v1.4.1
 + Fixed a bug in the grayscale-image deblurring codes and made
   these codes compatible with Matlab 7 or newer versions.

v1.4
 + Added grayscale-image deblurring

v1.3
 + Added grayscale-image joint sharpening and denoising

v1.2.1
 + Fixed the output of the VBM3D to be the final Wiener estimate 
   rather than the intermediate basic estimate
 + Fixed a problem when the original video is provided as a 3D
   matrix

v1.2.
 + Added grayscale-video denoising files

v1.1.3. 
 + Added support for Linux x86-compatible platforms

v1.1.2. 
 + Fixed bugs related with Matlab v.6.1

v1.1.1. 
 + Fixed bugs related with Matlab v.6 (e.g., "isfloat" was not 
   available and "imshow" did not work with single precision)
 + Improved the usage examples shown by executing "help BM3D"
   or "help CBM3D" MATLAB commands

v1.1. 
 + Fixed a bug in comparisons of the image sizes, which was
   causing problems when executing "CBM3D(1,z,sigma);"
 + Fixed a bug that was causing a crash when the input images are
   of type "uint8"
 + Fixed a problem that has caused some versions of imshow to 
   report an error
 + Fixed few typos in the comments of the functions
 + Made the parameters of the BM3D and the C-BM3D the same

v1.0. Initial version.


------------------------------------------------------------------
Publications
------------------------------------------------------------------

[1] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Image 
denoising by sparse 3D transform-domain collaborative filtering," 
IEEE Trans. Image Process., vol. 16, no. 8, August 2007.

[2] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Color 
image denoising via sparse 3D collaborative filtering with 
grouping constraint in luminance-chrominance space," Proc. IEEE
Int. Conf. Image Process., ICIP 2007, San Antonio, TX, USA, 
September 2007.

[3] K. Dabov, A. Foi, and K. Egiazarian, "Video denoising by 
sparse 3D transform-domain collaborative filtering," Proc.
European Signal Process. Conf., EUSIPCO 2007, Poznan, Poland,
September 2007.

[4] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Joint 
image sharpening and denoising by 3D transform-domain 
collaborative filtering," Proc. 2007 Int. TICSP Workshop Spectral 
Meth. Multirate Signal Process., SMMSP 2007, Moscow, Russia, 
September 2007.

[5] K. Dabov, A. Foi, and K. Egiazarian, "Image restoration by 
sparse 3D transform-domain collaborative filtering," Proc.
SPIE Electronic Imaging, January 2008.

[6] Y. Hou, C. Zhao, D. Yang, and Y. Cheng, 'Comment on "Image 
Denoising by Sparse 3D Transform-Domain Collaborative Filtering"'
accepted for publication, IEEE Trans. Image Process., July, 2010.

------------------------------------------------------------------
Disclaimer
------------------------------------------------------------------

Any unauthorized use of these routines for industrial or profit-
oriented activities is expressively prohibited. By downloading 
and/or using any of these files, you implicitly agree to all the 
terms of the TUT limited license:
http://www.cs.tut.fi/~foi/GCF-BM3D/legal_notice.html


------------------------------------------------------------------
Feedback
------------------------------------------------------------------

If you have any comment, suggestion, or question, please do
contact Kostadin Dabov at: dabov _at_ cs.tut.fi

