Compile the code
1) First of all, you have to generate the Makefile with cmake. Go to the src directory and type :
ccmake .
Then follow the instruction to generate the Makefile (press c to configure then g to generate)
2) To compile the code, you can use the build script available in the scr directory

Testing
A few matlab scripts are available in src/test :
* showImg.m is loading an image and compute the integral image. You can then select part of the image and it will return the haar response for the select area.
* enumerateLearners.m enumerate the weak learners for a specified type
* rectangleFeature.m returns the haar response for a selected position in an image
