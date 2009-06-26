Setup your workstation:
* Install gcc and g++ (tested with 4.3)
* Install CMake
* Install OpenCV. There is a linux package for most distributions.
sudo apt-get install gcc g++ cmake libcv-dev
* Copy the file FindOpenCV.cmake in the misc directory to the Modules directory of your CMake installation (where also FindOpenGL.cmake can be found). This will enable CMake to find OpenCV on your machine. Make sure that you are the owner of the file or you will have to use sudo ccmake .

-------------------------------------------------------------------------------

Compile the code
* First of all, you have to generate the Makefile with cmake. Go to the src directory and type :
ccmake .
Then follow the instruction to generate the Makefile (press c to configure then g to generate)
* To compile the code, you can use the build script available in the scr directory

-------------------------------------------------------------------------------

Testing
A few matlab scripts are available in src/test :
* showImg.m is loading an image and compute the integral image. You can then select part of the image and it will return the haar response for the select area.
* enumerateLearners.m enumerate the weak learners for a specified type
* rectangleFeature.m returns the haar response for a selected position in an image
